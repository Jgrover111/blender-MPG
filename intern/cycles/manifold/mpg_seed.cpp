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

#include <cfloat>
#include <cmath>

CCL_NAMESPACE_BEGIN

float mpg_rebuild_seed_pdf(const MpgSeedRay &seed)
{
  const float raw_pdf = seed.seed_pdf_raw;
  if (!(isfinite_safe(raw_pdf) && raw_pdf > 0.0f)) {
    return 0.0f;
  }

  const int acceptance_trials = seed.accepted_trial_count;
  if (acceptance_trials <= 0) {
    return 0.0f;
  }

  const float normalized_pdf = raw_pdf * float(acceptance_trials);
  if (!(isfinite_safe(normalized_pdf) && normalized_pdf > 0.0f)) {
    return 0.0f;
  }

  return normalized_pdf;
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

  float3 reflection_normal = shading_normal;
  float3 transmission_normal = geometric_normal;

  bool prefer_transmission = (seed_lobe == SeedLobe::Transmission);
  if (seed_lobe == SeedLobe::Dual) {
    if (!prefer_transmission) {
      /* Use the guided axis (when available) to decide whether we should bias the seed toward
       * the refractive or reflective hemisphere. Falling back to the view direction provides a
       * deterministic choice when the guide has no dominant direction yet. */
      const float3 reference_dir = (!is_zero(guide.mean_dir)) ? guide.mean_dir : sd.wi;
      const bool reference_valid = !is_zero(reference_dir);
      if (reference_valid) {
        const float ref_dot_trans = dot(reference_dir, transmission_normal);
        const float ref_dot_refl = dot(reference_dir, reflection_normal);
        prefer_transmission = fabsf(ref_dot_trans) > fabsf(ref_dot_refl);
      }
    }
  }

  bool using_transmission_hemisphere = prefer_transmission && geometric_normal_valid;
  float3 hemisphere_normal = using_transmission_hemisphere ? transmission_normal : reflection_normal;
  bool hemisphere_valid = !is_zero(hemisphere_normal);
  if (!hemisphere_valid) {
    hemisphere_normal = reflection_normal;
    hemisphere_valid = !is_zero(hemisphere_normal);
    using_transmission_hemisphere = false;
  }

  float hemisphere_sign = 1.0f;
  if (using_transmission_hemisphere) {
    const float dot_ng_wi = dot(transmission_normal, sd.wi);
    hemisphere_sign = (dot_ng_wi >= 0.0f) ? -1.0f : 1.0f;
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

  /* Determine the dominant seed direction from the guided mean with optional jitter. */
  float3 axis = guide.mean_dir;
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

  const float min_cone_angle = 0.00872664626f; /* ~0.5 degrees. */
  float jitter = fmaxf(options.angular_jitter, min_cone_angle);
  float candidate_pdf = 0.0f;
  float3 seed_direction = zero_float3();
  const int seed_branch_count = use_uniform_sphere_sampling ? 32 : 16;
  const int max_seed_attempts = bootstrap_seed ? 32 : (use_uniform_fallback ? 32 : 16);
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

  auto try_seed_sample = [&](const float3 &candidate_direction,
                             const float candidate_pdf,
                             Intersection &out_isect,
                             const SeedTrialBranch branch) -> bool {
    int &branch_trials = (branch == SeedTrialBranch::Guided) ? guided_trials : fallback_trials;
    ++branch_trials;
    if (is_zero(candidate_direction) || candidate_pdf <= 0.0f) {
      last_failure = MPG_FAILURE_INVALID_SEED_PDF;
      return false;
    }

    /* Bootstrap seeds and uniform fallbacks explore the full sphere and let the solver
     * decide the valid manifold branch. Directional seeds must still respect the
     * reflection/transmission hemisphere filtering. */
    const bool skip_hemisphere_test = bootstrap_seed ||
                                      (use_uniform_fallback &&
                                       branch == SeedTrialBranch::Fallback);
    if (!skip_hemisphere_test && !matches_hemisphere(candidate_direction)) {
      last_failure = MPG_FAILURE_SEED;
      return false;
    }

    const float3 normalized_direction = normalize(candidate_direction);

    Ray ray;
    ray.P = mpg_surface_ray_offset(kg, sd, sd.P, normalized_direction);
    ray.D = normalized_direction;
    ray.tmin = 0.0f;
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
    seed.use_smooth_normals = has_smooth_normals && !using_transmission_hemisphere;
    successful_branch = branch;
    return true;
  };

  for (int attempt = 0; attempt < max_seed_attempts && !seed_valid; ++attempt) {
    const float2 rand = path_branched_rng_2D(kg,
                                             &rng_state,
                                             attempt + rng_branch_offset,
                                             seed_branch_count,
                                             PRNG_SURFACE_BSDF);

    const SeedTrialBranch branch = use_uniform_sphere_sampling ?
                                       SeedTrialBranch::Fallback :
                                       SeedTrialBranch::Guided;

    if (use_uniform_sphere_sampling) {
      seed_direction = sample_uniform_sphere(rand);
      candidate_pdf = M_1_4PI_F;
    }
    else {
      float unused_cos = 0.0f;
      seed_direction = sample_uniform_cone(
          axis, one_minus_cos(jitter), rand, &unused_cos, &candidate_pdf);
    }

    seed_valid = try_seed_sample(seed_direction, candidate_pdf, isect, branch);
  }

  if (!seed_valid && !use_uniform_fallback) {
    const int fallback_branch_count = 32;
    const int fallback_attempts = 32;
    const int attempt_offset = max_seed_attempts;
    const float fallback_one_minus_cos = one_minus_cos(0.6f * M_PI_F);

    for (int attempt = 0; attempt < fallback_attempts && !seed_valid; ++attempt) {
      const float2 rand = path_branched_rng_2D(kg,
                                               &rng_state,
                                               attempt + attempt_offset + rng_branch_offset,
                                               fallback_branch_count,
                                               PRNG_SURFACE_BSDF);

      const bool fallback_uses_uniform_sphere = bootstrap_seed || !axis_valid;
      if (!fallback_uses_uniform_sphere) {
        float unused_cos = 0.0f;
        seed_direction = sample_uniform_cone(
            axis, fallback_one_minus_cos, rand, &unused_cos, &candidate_pdf);
      }
      else {
        seed_direction = sample_uniform_sphere(rand);
        candidate_pdf = M_1_4PI_F;
      }

      seed_valid =
          try_seed_sample(seed_direction, candidate_pdf, isect, SeedTrialBranch::Fallback);
    }
  }

  if (!seed_valid) {
    failure_code = last_failure;
    return false;
  }

  const int total_trials = guided_trials + fallback_trials;
  const int accepted_trials = (successful_branch == SeedTrialBranch::Guided) ? guided_trials :
                                                                        fallback_trials;
  if (accepted_trials <= 0) {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }
  /* The candidate density must be renormalized by the acceptance probability of the branch that
   * produced a valid specular hit. Guided and fallback cones are independent proposals, so only
   * the attempts made with the successful branch affect the normalization. */
  const int branch_trials = accepted_trials;
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
  seed.trial_count = total_trials;
  seed.accepted_trial_count = branch_trials;
  seed.seed_pdf = mpg_rebuild_seed_pdf(seed);
  if (seed.seed_pdf <= 0.0f) {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }
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