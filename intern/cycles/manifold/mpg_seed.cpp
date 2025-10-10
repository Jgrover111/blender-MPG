/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_seed.h"

#include "kernel/bvh/bvh.h"
#include "kernel/bvh/util.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/geom/object.h"
#include "kernel/geom/triangle.h"
#include "kernel/light/light.h"
#include "kernel/sample/mapping.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/svm/types.h"
#include "kernel/types.h"

#include "util/math_base.h"
#include "util/math_float4.h"
#include "util/math_intersect.h"
#include "util/hash.h"

#include <cfloat>
#include <cmath>

CCL_NAMESPACE_BEGIN

float mpg_rebuild_seed_pdf(const MpgSeedRay &seed)
{
  const float raw_pdf = seed.seed_pdf_raw;
  if (!(isfinite_safe(raw_pdf) && raw_pdf > 0.0f)) {
    return 0.0f;
  }

  float resample_factor = seed.seed_resample_factor;
  if (!(isfinite_safe(resample_factor) && resample_factor > 0.0f)) {
    const int total_trials = seed.trial_count;
    if (total_trials <= 0) {
      return 0.0f;
    }
    resample_factor = float(total_trials);
  }

  const float normalized_pdf = raw_pdf * resample_factor;
  if (!(isfinite_safe(normalized_pdf) && normalized_pdf > 0.0f)) {
    return 0.0f;
  }

  return normalized_pdf;
}

float mpg_seed_branch_probability(const int guided_attempt_budget,
                                  const int fallback_attempt_budget,
                                  MpgSeedBranch branch)
{
  const float guided_weight = (guided_attempt_budget > 0) ? float(guided_attempt_budget) : 0.0f;
  const float fallback_weight = (fallback_attempt_budget > 0) ? float(fallback_attempt_budget) : 0.0f;

  switch (branch) {
    case MPG_SEED_BRANCH_GUIDED:
      if (guided_weight <= 0.0f) {
        return 0.0f;
      }
      if (fallback_weight <= 0.0f) {
        return 1.0f;
      }
      break;
    case MPG_SEED_BRANCH_FALLBACK:
      if (fallback_weight <= 0.0f) {
        return 0.0f;
      }
      if (guided_weight <= 0.0f) {
        return 1.0f;
      }
      break;
    default:
      return 0.0f;
  }

  const float weight_sum = guided_weight + fallback_weight;
  if (weight_sum <= 0.0f) {
    return 0.0f;
  }

  const float probability = (branch == MPG_SEED_BRANCH_GUIDED) ?
                                (guided_weight / weight_sum) :
                                (fallback_weight / weight_sum);
  return fminf(fmaxf(probability, 0.0f), 1.0f);
}

static inline bool has_specular_bsdf_at_hit(KernelGlobals kg,
                                            const Ray &ray,
                                            const Intersection &isect,
                                            bool &has_smooth_normals)
{
  has_smooth_normals = false;
  ShaderData spec_sd = {};
  shader_setup_from_ray(kg, &spec_sd, &ray, const_cast<Intersection *>(&isect));

  const ConstIntegratorState integrator_state = nullptr;
  /* We only need closures, not emission; this is fast enough and CPU-safe. */
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
      kg, integrator_state, &spec_sd, nullptr, PATH_RAY_CAMERA, true);

  has_smooth_normals = (spec_sd.shader & SHADER_SMOOTH_NORMAL) != 0;

  for (int i = 0; i < spec_sd.num_closure; ++i) {
    const ShaderClosure *c = &spec_sd.closure[i];
    if (!CLOSURE_IS_BSDF(c->type)) {
      continue;
    }
    const bool is_micro    = CLOSURE_IS_BSDF_MICROFACET(c->type);
    const bool is_singular = CLOSURE_IS_BSDF_SINGULAR(c->type);
    if (is_singular) {
      return true;
    }
    if (is_micro) {
      const MicrofacetBsdf *mf = reinterpret_cast<const MicrofacetBsdf *>(c);
      const float a = fmaxf(mf->alpha_x, mf->alpha_y);
      if (a <= 0.02f) {
        return true; /* razor-sharp microfacet behaves like specular for MPG v1 */
      }
    }
  }
  return false;
}

float3 mpg_surface_ray_offset(KernelGlobals kg,
                              const ShaderData &sd,
                              const float3 ray_P,
                              const float3 ray_D)
{
  if (!(sd.type & PRIMITIVE_TRIANGLE)) {
    return ray_P;
  }

  float3 verts[3];
  if (sd.type == PRIMITIVE_TRIANGLE) {
    triangle_vertices(kg, sd.prim, verts);
  }
  else {
    kernel_assert(sd.type == PRIMITIVE_MOTION_TRIANGLE);
    motion_triangle_vertices(kg, sd.object, sd.prim, sd.time, verts);
  }

  float3 local_ray_P = ray_P;
  float3 local_ray_D = ray_D;

  if (!(sd.object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    const Transform itfm = object_get_inverse_transform(kg, &sd);
    local_ray_P = transform_point(&itfm, local_ray_P);
    local_ray_D = transform_direction(&itfm, local_ray_D);
  }

  if (ray_triangle_intersect_self(local_ray_P, local_ray_D, verts)) {
    return ray_P;
  }

  return ray_offset(ray_P, sd.Ng);
}

bool mpg_generate_seed(KernelGlobals kg,
                       const ShaderData &sd,
                       const ShaderClosure &bsdf,
                       const GuideSummary &guide,
                       const MpgOptions &options,
                       const uint32_t path_flag,
                       const int bounce,
                       const RNGState &rng_state,
                       MpgSeedRay &seed,
                       MpgFailureCode &failure_code,
                       const int rng_branch_offset)
{
  seed = MpgSeedRay();
  failure_code = MPG_FAILURE_NONE;

  enum class SeedLobe {
    Reflection,
    Transmission,
    Dual
  };

  const auto classify_seed_lobe = [](const ShaderClosure &closure) {
    if (CLOSURE_IS_BSDF_TRANSPARENT(closure.type) ||
        CLOSURE_IS_BSDF_TRANSMISSION(closure.type))
    {
      return SeedLobe::Transmission;
    }
    if (CLOSURE_IS_GLASS(closure.type)) {
      return SeedLobe::Dual;
    }
    return SeedLobe::Reflection;
  };

  const SeedLobe seed_lobe = classify_seed_lobe(bsdf);

  float3 shading_normal = sd.N;
  bool shading_normal_valid = !is_zero(shading_normal);
  if (shading_normal_valid) {
    shading_normal = safe_normalize(shading_normal);
    shading_normal_valid = !is_zero(shading_normal);
  }

  float3 geometric_normal = sd.Ng;
  bool geometric_normal_valid = !is_zero(geometric_normal);
  if (geometric_normal_valid) {
    geometric_normal = safe_normalize(geometric_normal);
    geometric_normal_valid = !is_zero(geometric_normal);
  }

  if (!shading_normal_valid && geometric_normal_valid) {
    shading_normal = geometric_normal;
    shading_normal_valid = true;
  }
  if (!geometric_normal_valid && shading_normal_valid) {
    geometric_normal = shading_normal;
    geometric_normal_valid = true;
  }
  if (!shading_normal_valid && !geometric_normal_valid) {
    shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    shading_normal_valid = true;
    geometric_normal = shading_normal;
    geometric_normal_valid = true;
  }

  float3 reflection_normal = geometric_normal_valid ? geometric_normal : shading_normal;
  float3 transmission_normal = geometric_normal_valid ? geometric_normal : shading_normal;

  float3 guided_axis = guide.mean_dir;
  bool guided_axis_valid = !is_zero(guided_axis);
  if (guided_axis_valid) {
    guided_axis = safe_normalize(guided_axis);
    guided_axis_valid = !is_zero(guided_axis);
  }

  const float transmission_dot_wi = geometric_normal_valid ? dot(transmission_normal, sd.wi) : 0.0f;
  const float transmission_hemisphere_sign = geometric_normal_valid ?
                                                 ((transmission_dot_wi >= 0.0f) ? -1.0f : 1.0f) :
                                                 1.0f;
  const float hemisphere_epsilon = 1.0e-5f;

  bool prefer_transmission = false;
  bool prefer_transmission_from_guide = false;

  if (guided_axis_valid && geometric_normal_valid) {
    const float dot_axis_transmission = dot(guided_axis, transmission_normal);
    if (fabsf(dot_axis_transmission) > hemisphere_epsilon) {
      if (fabsf(transmission_dot_wi) > hemisphere_epsilon) {
        prefer_transmission = (dot_axis_transmission * transmission_dot_wi < 0.0f);
      }
      else {
        prefer_transmission = (dot_axis_transmission < 0.0f);
      }
      prefer_transmission_from_guide = true;
    }
  }

  if (!prefer_transmission_from_guide) {
    if (seed_lobe == SeedLobe::Transmission) {
      prefer_transmission = true;
    }
    else if (seed_lobe == SeedLobe::Dual) {
      bool decided = false;
      if (guided_axis_valid && geometric_normal_valid) {
        const float ref_dot_trans = dot(guided_axis, transmission_normal);
        const float ref_dot_refl = dot(guided_axis, reflection_normal);
        if (fabsf(ref_dot_trans) > hemisphere_epsilon || fabsf(ref_dot_refl) > hemisphere_epsilon) {
          prefer_transmission = fabsf(ref_dot_trans) > fabsf(ref_dot_refl);
          decided = true;
        }
      }
      if (!decided) {
        /* Fall back to the view direction when the guide does not provide a stable mean yet. */
        const float3 reference_dir = (!is_zero(guide.mean_dir)) ? guide.mean_dir : sd.wi;
        const bool reference_valid = !is_zero(reference_dir);
        if (reference_valid) {
          const float3 reference = safe_normalize(reference_dir);
          if (!is_zero(reference)) {
            const float ref_dot_trans = dot(reference, transmission_normal);
            const float ref_dot_refl = dot(reference, reflection_normal);
            prefer_transmission = fabsf(ref_dot_trans) > fabsf(ref_dot_refl);
          }
        }
      }
    }
  }

  bool using_transmission_hemisphere = prefer_transmission && geometric_normal_valid;

  /* Determine the dominant seed direction from the guided mean with optional jitter. */
  float3 axis = guided_axis_valid ? guided_axis : guide.mean_dir;
  if (is_zero(axis)) {
    /* Bootstrap seeds rely on a stable geometric frame. Ignore shading normal
     * perturbations when no guided mean is available to mirror the reference
     * solver and to avoid exploring directions that immediately graze the
     * receiver. */
    axis = sd.Ng;
    if (is_zero(axis)) {
      axis = sd.N;
    }
  }

  bool axis_valid = !is_zero(axis);
  if (axis_valid) {
    axis = safe_normalize(axis);
    axis_valid = !is_zero(axis);
  }

  if (axis_valid && geometric_normal_valid) {
    const float dot_axis_transmission = dot(axis, transmission_normal);
    if (fabsf(dot_axis_transmission) > hemisphere_epsilon &&
        (dot_axis_transmission * transmission_hemisphere_sign) >= 0.0f)
    {
      using_transmission_hemisphere = true;
      prefer_transmission = true;
    }
  }

  float3 hemisphere_normal = using_transmission_hemisphere ? transmission_normal : reflection_normal;
  bool hemisphere_valid = !is_zero(hemisphere_normal);
  if (!hemisphere_valid && using_transmission_hemisphere) {
    hemisphere_normal = reflection_normal;
    hemisphere_valid = !is_zero(hemisphere_normal);
    using_transmission_hemisphere = false;
  }
  if (!hemisphere_valid) {
    hemisphere_normal = reflection_normal;
    hemisphere_valid = !is_zero(hemisphere_normal);
    using_transmission_hemisphere = false;
  }

  float hemisphere_sign = 1.0f;
  if (using_transmission_hemisphere) {
    hemisphere_sign = transmission_hemisphere_sign;
  }

  const auto matches_hemisphere = [&](const float3 &direction) {
    if (!hemisphere_valid) {
      return true;
    }
    if (using_transmission_hemisphere) {
      const float dot_ng_dir = dot(direction, transmission_normal);
      return dot_ng_dir * hemisphere_sign >= 0.0f;
    }
    return dot(direction, hemisphere_normal) >= 0.0f;
  };

  if (axis_valid && !matches_hemisphere(axis)) {
    if (!using_transmission_hemisphere) {
      axis = -axis;
      axis = safe_normalize(axis);
      axis_valid = !is_zero(axis);
    }
    else {
      axis_valid = false;
    }
  }

  const bool has_direction_relaxed = (guide.rbar > 1.0e-4f);
  /* Bootstrap mode (no directional signal yet) explores a uniform sphere distribution,
   * mirroring the mpg_try_connect gating thresholds so relaxed gating alone keeps
   * directional seeds narrow whenever the guide provides a stable mean. */
  const bool bootstrap_seed = !has_direction_relaxed;
  const bool use_uniform_fallback = !axis_valid;
  const bool use_uniform_sphere_sampling = bootstrap_seed || use_uniform_fallback;

  const int max_supported_bounces = clamp(options.max_bounces, 1, 2);
  const bool allow_double_bounce = (max_supported_bounces >= 2);

  float geom_single_weight = 1.0f;
  float geom_double_weight = allow_double_bounce ? 1.0f : 0.0f;
  float geom_sum = geom_single_weight + geom_double_weight;
  if (!(geom_sum > 0.0f)) {
    geom_sum = 1.0f;
    geom_double_weight = 0.0f;
  }
  float geom_pdf_single = geom_single_weight / geom_sum;
  float geom_pdf_double = allow_double_bounce ? (geom_double_weight / geom_sum) : 0.0f;

  float guided_pdf_single = geom_pdf_single;
  float guided_pdf_double = geom_pdf_double;

  const bool guided_distribution_available = allow_double_bounce && axis_valid &&
                                             !use_uniform_sphere_sampling;

  if (guided_distribution_available) {
    float guided_single_weight = prefer_transmission ? 0.0f : 1.0f;
    float guided_double_weight = prefer_transmission ? 1.0f : 0.0f;
    const float guided_sum = guided_single_weight + guided_double_weight;
    if (guided_sum > 0.0f) {
      guided_pdf_single = guided_single_weight / guided_sum;
      guided_pdf_double = guided_double_weight / guided_sum;
    }
    else {
      guided_pdf_single = geom_pdf_single;
      guided_pdf_double = geom_pdf_double;
    }
  }

  float pdf_single = geom_pdf_single;
  float pdf_double = geom_pdf_double;
  if (allow_double_bounce) {
    const float bounce_alpha = guided_distribution_available ? 0.5f : 1.0f;
    pdf_single = bounce_alpha * geom_pdf_single + (1.0f - bounce_alpha) * guided_pdf_single;
    pdf_double = bounce_alpha * geom_pdf_double + (1.0f - bounce_alpha) * guided_pdf_double;
  }
  else {
    pdf_single = 1.0f;
    pdf_double = 0.0f;
  }

  float pdf_sum = pdf_single + pdf_double;
  if (!(isfinite_safe(pdf_sum) && pdf_sum > 0.0f)) {
    pdf_single = 1.0f;
    pdf_double = 0.0f;
    pdf_sum = 1.0f;
  }
  pdf_single /= pdf_sum;
  pdf_double = allow_double_bounce ? (pdf_double / pdf_sum) : 0.0f;

  const uint32_t branch_offset_u = (rng_branch_offset >= 0) ? uint(rng_branch_offset) :
                                                                  uint(-rng_branch_offset);
  const uint32_t bounce_seed = hash_uint3(
      rng_state.rng_pixel, uint(rng_state.sample), rng_state.rng_offset + branch_offset_u);
  const float bounce_rand = uint_to_float_excl(bounce_seed);

  int selected_bounce_count = 1;
  float selected_bounce_pdf = pdf_single;

  const bool allow_single = pdf_single > 0.0f;
  const bool allow_double_pdf = allow_double_bounce && pdf_double > 0.0f;

  if (!allow_single && allow_double_pdf) {
    selected_bounce_count = 2;
    selected_bounce_pdf = pdf_double;
  }
  else if (allow_single && allow_double_pdf) {
    if (bounce_rand >= pdf_single) {
      selected_bounce_count = 2;
      selected_bounce_pdf = pdf_double;
    }
  }
  else if (!allow_single && !allow_double_pdf) {
    selected_bounce_count = allow_double_bounce ? 2 : 1;
    selected_bounce_pdf = allow_double_bounce ? pdf_double : pdf_single;
  }

  if (!(isfinite_safe(selected_bounce_pdf) && selected_bounce_pdf > 0.0f)) {
    failure_code = MPG_FAILURE_INVALID_PDF;
    return false;
  }

  seed.bounce_count = selected_bounce_count;
  seed.bounce_pdf_raw = selected_bounce_pdf;
  seed.bounce_pdf = fmaxf(selected_bounce_pdf, 1.0e-16f);

  const float min_cone_angle = 0.00872664626f; /* ~0.5 degrees. */
  float jitter = fmaxf(options.angular_jitter, min_cone_angle);
  float3 seed_direction = zero_float3();
  bool seed_valid = false;
  Intersection isect = {};
  MpgFailureCode last_failure = MPG_FAILURE_SEED;
  enum class SeedTrialBranch {
    Guided,
    Fallback,
  };

  SeedTrialBranch successful_branch = SeedTrialBranch::Guided;
  int guided_trials = 0;
  int fallback_trials = 0;
  float accepted_seed_pdf = 0.0f;
  float accepted_branch_pdf = 0.0f;
  float accepted_direction_pdf = 0.0f;

  const int uniform_attempt_budget = bootstrap_seed ? 32 : (use_uniform_fallback ? 32 : 16);
  const int guided_attempt_budget = use_uniform_sphere_sampling ? 0 : uniform_attempt_budget;
  const int fallback_attempt_budget = use_uniform_sphere_sampling ? uniform_attempt_budget : 32;
  int total_attempt_budget = guided_attempt_budget + fallback_attempt_budget;
  if (total_attempt_budget <= 0) {
    total_attempt_budget = 1;
  }
  const int seed_branch_count = total_attempt_budget;
  const float guided_branch_probability =
      mpg_seed_branch_probability(guided_attempt_budget, fallback_attempt_budget, MPG_SEED_BRANCH_GUIDED);
  const float fallback_branch_probability =
      mpg_seed_branch_probability(guided_attempt_budget, fallback_attempt_budget, MPG_SEED_BRANCH_FALLBACK);
  const float fallback_one_minus_cos = one_minus_cos(0.6f * M_PI_F);

  auto try_seed_sample = [&](const float3 &candidate_direction,
                             const float candidate_pdf,
                             const float candidate_branch_pdf,
                             const float candidate_direction_pdf,
                             Intersection &out_isect,
                             const SeedTrialBranch branch) -> bool {
    int &branch_trials = (branch == SeedTrialBranch::Guided) ? guided_trials : fallback_trials;
    ++branch_trials;
    if (is_zero(candidate_direction) || candidate_pdf <= 0.0f) {
      last_failure = MPG_FAILURE_INVALID_SEED_PDF;
      return false;
    }

    /* Bootstrap seeds and uniform fallbacks still probe broadly but must respect the
     * hemisphere test so rays never shoot across Ng. Directional seeds continue to obey
     * the same filtering. */
    if (!matches_hemisphere(candidate_direction)) {
      last_failure = MPG_FAILURE_SEED;
      return false;
    }

    const float3 normalized_direction = normalize(candidate_direction);

    Ray ray;
    ray.P = mpg_surface_ray_offset(kg, sd, sd.P, normalized_direction);
    ray.D = normalized_direction;
    ray.tmin = 1.0e-4f;
    ray.tmax = FLT_MAX;
    ray.time = sd.time;
    ray.self.prim = sd.prim;
    ray.self.object = sd.object;
    ray.self.light_prim = PRIM_NONE;
    ray.self.light_object = OBJECT_NONE;

    Intersection candidate_isect = {};
    if (!scene_intersect(kg, &ray, PATH_RAY_ALL_VISIBILITY, &candidate_isect)) {
      last_failure = MPG_FAILURE_SEED;
      return false;
    }

    if (!(candidate_isect.type & PRIMITIVE_TRIANGLE)) {
      last_failure = MPG_FAILURE_GEOMETRY;
      return false;
    }

    bool has_smooth_normals = false;
    if (!has_specular_bsdf_at_hit(kg, ray, candidate_isect, has_smooth_normals)) {
      last_failure = MPG_FAILURE_NO_SPECULAR;
      return false;
    }

    out_isect = candidate_isect;
    seed_direction = normalized_direction;
    accepted_seed_pdf = candidate_pdf;
    accepted_branch_pdf = candidate_branch_pdf;
    accepted_direction_pdf = candidate_direction_pdf;
    seed.use_smooth_normals = has_smooth_normals && !using_transmission_hemisphere;
    successful_branch = branch;
    return true;
  };

  for (int attempt = 0; attempt < total_attempt_budget && !seed_valid; ++attempt) {
    const float3 rand = path_branched_rng_3D(kg,
                                            &rng_state,
                                            attempt + rng_branch_offset,
                                            seed_branch_count,
                                            PRNG_SURFACE_BSDF);

    SeedTrialBranch branch = SeedTrialBranch::Fallback;
    float branch_pdf = 1.0f;
    const bool has_guided_branch = guided_branch_probability > 0.0f;
    const bool has_fallback_branch = fallback_branch_probability > 0.0f;

    if (has_guided_branch && has_fallback_branch) {
      const float branch_sample = rand.x;
      const bool choose_guided = branch_sample < guided_branch_probability;
      branch = choose_guided ? SeedTrialBranch::Guided : SeedTrialBranch::Fallback;
      branch_pdf = choose_guided ? guided_branch_probability : fallback_branch_probability;
    }
    else if (has_guided_branch) {
      branch = SeedTrialBranch::Guided;
      branch_pdf = 1.0f;
    }
    else if (has_fallback_branch) {
      branch = SeedTrialBranch::Fallback;
      branch_pdf = 1.0f;
    }

    float direction_pdf = 0.0f;
    float3 candidate_direction = zero_float3();
    const float2 rand_dir = make_float2(rand.y, rand.z);

    if (branch == SeedTrialBranch::Guided) {
      float unused_cos = 0.0f;
      candidate_direction =
          sample_uniform_cone(axis, one_minus_cos(jitter), rand_dir, &unused_cos, &direction_pdf);
    }
    else {
      const bool fallback_uses_uniform_sphere = bootstrap_seed || !axis_valid;
      if (!fallback_uses_uniform_sphere) {
        float unused_cos = 0.0f;
        candidate_direction = sample_uniform_cone(
            axis, fallback_one_minus_cos, rand_dir, &unused_cos, &direction_pdf);
      }
      else {
        candidate_direction = sample_uniform_sphere(rand_dir);
        direction_pdf = M_1_4PI_F;
      }
    }

    const float candidate_pdf = direction_pdf * branch_pdf;
    seed_valid = try_seed_sample(
        candidate_direction, candidate_pdf, branch_pdf, direction_pdf, isect, branch);
  }

  if (!seed_valid) {
    failure_code = last_failure;
    return false;
  }

  const int total_trials = guided_trials + fallback_trials;
  if (total_trials <= 0) {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }
  const int accepted_trials = (successful_branch == SeedTrialBranch::Guided) ? guided_trials :
                                                                        fallback_trials;
  if (accepted_trials <= 0) {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }

  const float resample_factor = float(total_trials);
  const float normalized_pdf = accepted_seed_pdf * resample_factor;
  if (!(isfinite_safe(normalized_pdf) && normalized_pdf > 0.0f)) {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }
  /* Sample an emitter using the Cycles light sampling routine. */
  const float3 rand_light = path_state_rng_3D(kg, &rng_state, PRNG_LIGHT);
  LightSample light_sample;
  if (!light_sample_from_position(kg,
                                  rand_light,
                                  sd.time,
                                  sd.P,
                                  sd.N,
                                  light_link_receiver_nee(kg, &sd),
                                  sd.flag,
                                  bounce,
                                  path_flag,
                                  &light_sample))
  {
    failure_code = MPG_FAILURE_SEED;
    return false;
  }

  if (light_sample.pdf <= 0.0f) {
    failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
    return false;
  }

  seed.direction = seed_direction;
  seed.seed_pdf_raw = accepted_seed_pdf;
  seed.seed_branch_pdf = accepted_branch_pdf;
  seed.seed_direction_pdf = accepted_direction_pdf;
  seed.trial_count = total_trials;
  seed.accepted_trial_count = accepted_trials;
  seed.guided_trial_count = guided_trials;
  seed.fallback_trial_count = fallback_trials;
  seed.seed_resample_factor = resample_factor;
  seed.branch = (successful_branch == SeedTrialBranch::Guided) ? MPG_SEED_BRANCH_GUIDED :
                                                                    MPG_SEED_BRANCH_FALLBACK;
  seed.seed_pdf = fmaxf(normalized_pdf, 1.0e-16f);
  seed.light_sample = light_sample;
  seed.path_flag = path_flag;
  /* Keep the light endpoint provided by the Cycles light sampler. For distant/background
   * lights it stays at infinity, while finite lights already reference the actual vertex.
   * Only normalize the direction so downstream code can derive a stable segment. */
  const bool has_light_direction = !is_zero(light_sample.D);
  if (has_light_direction) {
    seed.light_sample.D = normalize(light_sample.D);
  }

  seed.object = isect.object;
  seed.prim = isect.prim;
  seed.bary_u = isect.u;
  seed.bary_v = isect.v;
  return true;
}

CCL_NAMESPACE_END