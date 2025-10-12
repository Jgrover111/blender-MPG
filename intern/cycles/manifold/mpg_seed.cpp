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

#include "util/color.h"
#include "util/math_base.h"
#include "util/math_float4.h"
#include "util/math_intersect.h"
#include "util/hash.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

CCL_NAMESPACE_BEGIN

float mpg_rebuild_seed_pdf(const MpgSeedRay &seed)
{
  const float raw_pdf = seed.seed_pdf_raw;
  if (!(isfinite_safe(raw_pdf) && raw_pdf > 0.0f)) {
    return 0.0f;
  }

  const float expected_trials = seed.seed_resample_factor;
  if (!(isfinite_safe(expected_trials) && expected_trials > 0.0f)) {
    return 0.0f;
  }

  const float inv_expected_trials = 1.0f / fmaxf(expected_trials, 1.0e-16f);
  if (!(isfinite_safe(inv_expected_trials) && inv_expected_trials > 0.0f)) {
    return 0.0f;
  }

  const float normalized_pdf = raw_pdf * inv_expected_trials;
  if (!(isfinite_safe(normalized_pdf) && normalized_pdf > 0.0f)) {
    return 0.0f;
  }

  const float bounce_pdf = seed.bounce_pdf;
  const float bounce_pdf_raw = seed.bounce_pdf_raw;

  float bounce_factor = 1.0f;
  if (isfinite_safe(bounce_pdf_raw) && bounce_pdf_raw > 0.0f) {
    if (!(isfinite_safe(bounce_pdf) && bounce_pdf > 0.0f)) {
      return 0.0f;
    }

    const float inv_bounce_pdf_raw = 1.0f / fmaxf(bounce_pdf_raw, 1.0e-16f);
    if (!(isfinite_safe(inv_bounce_pdf_raw) && inv_bounce_pdf_raw > 0.0f)) {
      return 0.0f;
    }

    bounce_factor = bounce_pdf * inv_bounce_pdf_raw;
  }
  else {
    if (!(isfinite_safe(bounce_pdf) && bounce_pdf > 0.0f)) {
      return 0.0f;
    }

    bounce_factor = bounce_pdf;
  }

  const float combined_pdf = normalized_pdf * bounce_factor;
  if (!(isfinite_safe(combined_pdf) && combined_pdf > 0.0f)) {
    return 0.0f;
  }

  return combined_pdf;
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

static float mpg_uniform_cone_pdf(const float one_minus_cos_angle)
{
  if (one_minus_cos_angle > 0.0f) {
    return M_1_2PI_F / one_minus_cos_angle;
  }
  return 1.0f;
}

static float mpg_integrate_cone_hemisphere_partial(const float theta_start,
                                                   const float theta_end,
                                                   const float beta)
{
  if (!(theta_end > theta_start)) {
    return 0.0f;
  }

  const float sin_beta = sinf(beta);
  if (fabsf(sin_beta) < 1.0e-7f) {
    return 0.0f;
  }

  const float cot_beta = cosf(beta) / sin_beta;
  const int integration_steps = 32;
  const float step = (theta_end - theta_start) / float(integration_steps);
  float accumulated = 0.0f;

  for (int i = 0; i < integration_steps; ++i) {
    const float t = (float(i) + 0.5f) / float(integration_steps);
    const float theta = theta_start + (theta_end - theta_start) * t;
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);
    if (fabsf(sin_theta) < 1.0e-7f) {
      continue;
    }
    const float cot_theta = cos_theta / sin_theta;
    float cos_argument = -cot_theta * cot_beta;
    cos_argument = fminf(fmaxf(cos_argument, -1.0f), 1.0f);
    const float delta_phi = 2.0f * acosf(cos_argument);
    accumulated += sin_theta * delta_phi;
  }

  return accumulated * step;
}

static float mpg_uniform_cone_hemisphere_acceptance(const float3 &axis,
                                                    const float one_minus_cos_angle,
                                                    const float3 &hemisphere_axis)
{
  if (is_zero(axis) || is_zero(hemisphere_axis)) {
    return 1.0f;
  }

  const float3 axis_normalized = safe_normalize(axis);
  const float3 hemisphere_normalized = safe_normalize(hemisphere_axis);
  if (is_zero(axis_normalized) || is_zero(hemisphere_normalized)) {
    return 1.0f;
  }

  const float cos_theta_max = fminf(fmaxf(1.0f - one_minus_cos_angle, -1.0f), 1.0f);
  const float theta_max = acosf(cos_theta_max);
  const float cone_area = 2.0f * M_PI_F * (1.0f - cos_theta_max);
  if (!(cone_area > 0.0f)) {
    return (dot(axis_normalized, hemisphere_normalized) >= 0.0f) ? 1.0f : 0.0f;
  }

  const float cos_beta = fminf(fmaxf(dot(axis_normalized, hemisphere_normalized), -1.0f), 1.0f);
  const float beta = acosf(cos_beta);
  const float half_pi = 0.5f * M_PI_F;

  if (beta + theta_max <= half_pi) {
    return 1.0f;
  }
  if (beta >= half_pi + theta_max) {
    return 0.0f;
  }

  const float sin_beta = sinf(beta);
  if (fabsf(sin_beta) < 1.0e-7f) {
    return (cos_beta >= 0.0f) ? 1.0f : 0.0f;
  }

  float accepted_area = 0.0f;

  if (beta < half_pi) {
    const float theta_full = fminf(theta_max, fmaxf(half_pi - beta, 0.0f));
    if (theta_full > 0.0f) {
      accepted_area += 2.0f * M_PI_F * (1.0f - cosf(theta_full));
    }
    const float theta_partial_start = theta_full;
    if (theta_partial_start < theta_max) {
      accepted_area += mpg_integrate_cone_hemisphere_partial(theta_partial_start, theta_max, beta);
    }
  }
  else {
    const float theta_partial_start = fmaxf(beta - half_pi, 0.0f);
    if (theta_partial_start < theta_max) {
      accepted_area += mpg_integrate_cone_hemisphere_partial(theta_partial_start, theta_max, beta);
    }
  }

  float acceptance = accepted_area / cone_area;
  acceptance = fminf(fmaxf(acceptance, 0.0f), 1.0f);
  return acceptance;
}

static bool mpg_compute_dielectric_reflection_probability(KernelGlobals kg,
                                                          const ShaderData &sd,
                                                          const ShaderClosure &bsdf,
                                                          const float3 &fallback_normal,
                                                          float &reflection_probability)
{
  if (!CLOSURE_IS_BSDF_MICROFACET(bsdf.type)) {
    return false;
  }

  const MicrofacetBsdf *microfacet = reinterpret_cast<const MicrofacetBsdf *>(&bsdf);
  float3 fresnel_normal = microfacet->N;
  if (is_zero(fresnel_normal)) {
    fresnel_normal = fallback_normal;
  }
  if (is_zero(fresnel_normal)) {
    fresnel_normal = sd.Ng;
  }
  if (is_zero(fresnel_normal)) {
    fresnel_normal = sd.N;
  }
  if (is_zero(fresnel_normal)) {
    return false;
  }

  fresnel_normal = safe_normalize(fresnel_normal);
  if (is_zero(fresnel_normal)) {
    return false;
  }

  const float cos_theta_i = clamp(dot(fresnel_normal, sd.wi), -1.0f, 1.0f);

  Spectrum reflectance = zero_spectrum();
  Spectrum transmittance = zero_spectrum();
  float cos_theta_t = 0.0f;

  MicrofacetBsdf temp_bsdf = *microfacet;
  microfacet_fresnel(kg, &temp_bsdf, cos_theta_i, &cos_theta_t, &reflectance, &transmittance);

  const float reflectance_avg = fmaxf(average(reflectance), 0.0f);
  const float transmittance_avg = fmaxf(average(transmittance), 0.0f);
  const float probability_sum = reflectance_avg + transmittance_avg;

  if (!(probability_sum > 0.0f)) {
    reflection_probability = 1.0f;
    return true;
  }

  reflection_probability = fminf(fmaxf(reflectance_avg / probability_sum, 0.0f), 1.0f);
  return true;
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

  float reflection_probability = 1.0f;
  float transmission_probability = 0.0f;

  if (seed_lobe == SeedLobe::Transmission) {
    reflection_probability = 0.0f;
    transmission_probability = 1.0f;
  }
  else if (seed_lobe == SeedLobe::Dual) {
    if (prefer_transmission_from_guide) {
      reflection_probability = prefer_transmission ? 0.0f : 1.0f;
      transmission_probability = 1.0f - reflection_probability;
    }
    else {
      float fresnel_reflection = 1.0f;
      if (mpg_compute_dielectric_reflection_probability(
              kg, sd, bsdf, shading_normal, fresnel_reflection))
      {
        reflection_probability = fresnel_reflection;
        transmission_probability = 1.0f - reflection_probability;
      }
      else {
        reflection_probability = prefer_transmission ? 0.0f : 1.0f;
        transmission_probability = 1.0f - reflection_probability;
      }
    }
  }
  else {
    reflection_probability = 1.0f;
    transmission_probability = 0.0f;
  }

  reflection_probability = fminf(fmaxf(reflection_probability, 0.0f), 1.0f);
  transmission_probability = fminf(fmaxf(transmission_probability, 0.0f), 1.0f);
  float probability_sum = reflection_probability + transmission_probability;
  if (!(probability_sum > 0.0f)) {
    reflection_probability = 1.0f;
    transmission_probability = 0.0f;
    probability_sum = 1.0f;
  }
  reflection_probability /= probability_sum;
  transmission_probability /= probability_sum;

  if (seed_lobe == SeedLobe::Dual && !prefer_transmission_from_guide) {
    const float diff = transmission_probability - reflection_probability;
    if (diff > 1.0e-5f) {
      prefer_transmission = true;
    }
    else if (diff < -1.0e-5f) {
      prefer_transmission = false;
    }
  }

#ifdef WITH_CYCLES_DEBUG
  DCHECK(isfinite_safe(reflection_probability));
  DCHECK(isfinite_safe(transmission_probability));
  const float probability_total = reflection_probability + transmission_probability;
  DCHECK(fabsf(probability_total - 1.0f) <= 1.0e-5f);
#endif

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
      prefer_transmission = true;
    }
  }

  const bool reflection_hemisphere_valid = !is_zero(reflection_normal);
  const bool transmission_hemisphere_valid = !is_zero(transmission_normal);

  float3 axis_reflection = axis;
  bool axis_reflection_valid = axis_valid;
  if (axis_reflection_valid) {
    axis_reflection = safe_normalize(axis_reflection);
    axis_reflection_valid = !is_zero(axis_reflection);
  }
  if (axis_reflection_valid && reflection_hemisphere_valid) {
    float dot_axis_reflection = dot(axis_reflection, reflection_normal);
    if (fabsf(dot_axis_reflection) > hemisphere_epsilon && dot_axis_reflection < 0.0f) {
      axis_reflection = safe_normalize(-axis_reflection);
      axis_reflection_valid = !is_zero(axis_reflection);
      dot_axis_reflection = dot(axis_reflection, reflection_normal);
    }
    if (axis_reflection_valid && dot_axis_reflection < 0.0f) {
      axis_reflection_valid = false;
    }
  }

  float3 axis_transmission = axis;
  bool axis_transmission_valid = axis_valid;
  if (axis_transmission_valid) {
    axis_transmission = safe_normalize(axis_transmission);
    axis_transmission_valid = !is_zero(axis_transmission);
  }
  if (axis_transmission_valid && transmission_hemisphere_valid) {
    float dot_axis_transmission = dot(axis_transmission, transmission_normal);
    if (fabsf(dot_axis_transmission) > hemisphere_epsilon &&
        (dot_axis_transmission * transmission_hemisphere_sign) < 0.0f)
    {
      axis_transmission = safe_normalize(-axis_transmission);
      axis_transmission_valid = !is_zero(axis_transmission);
      dot_axis_transmission = dot(axis_transmission, transmission_normal);
    }
    if (axis_transmission_valid &&
        (dot_axis_transmission * transmission_hemisphere_sign) < 0.0f)
    {
      axis_transmission_valid = false;
    }
  }

  const bool has_direction_relaxed = (guide.rbar > 1.0e-4f);
  /* Bootstrap mode (no directional signal yet) explores a uniform sphere distribution,
   * mirroring the mpg_try_connect gating thresholds so relaxed gating alone keeps
   * directional seeds narrow whenever the guide provides a stable mean. */
  const bool bootstrap_seed = !has_direction_relaxed;
  const bool use_uniform_fallback = !axis_valid;
  const bool use_uniform_sphere_sampling = bootstrap_seed || use_uniform_fallback;

  const bool fallback_reflection_enforces_hemisphere =
      (bootstrap_seed || !axis_reflection_valid) && reflection_hemisphere_valid;
  const bool fallback_transmission_enforces_hemisphere =
      (bootstrap_seed || !axis_transmission_valid) && transmission_hemisphere_valid;

  float3 fallback_uniform_reflection_axis = zero_float3();
  if (fallback_reflection_enforces_hemisphere) {
    fallback_uniform_reflection_axis = reflection_normal;
    if (!is_zero(fallback_uniform_reflection_axis)) {
      fallback_uniform_reflection_axis = safe_normalize(fallback_uniform_reflection_axis);
    }
    else {
      fallback_uniform_reflection_axis = zero_float3();
    }
  }

  float3 fallback_uniform_transmission_axis = zero_float3();
  if (fallback_transmission_enforces_hemisphere) {
    float3 transmission_axis = transmission_normal;
    if (transmission_hemisphere_sign < 0.0f) {
      transmission_axis = -transmission_axis;
    }
    if (!is_zero(transmission_axis)) {
      fallback_uniform_transmission_axis = safe_normalize(transmission_axis);
    }
    else {
      fallback_uniform_transmission_axis = zero_float3();
    }
  }

  const bool fallback_uniform_reflection_axis_valid =
      !is_zero(fallback_uniform_reflection_axis);
  const bool fallback_uniform_transmission_axis_valid =
      !is_zero(fallback_uniform_transmission_axis);

  const auto matches_branch_hemisphere = [&](const float3 &direction,
                                             const MpgSeedScatter scatter_branch) {
    if (scatter_branch == MPG_SEED_SCATTER_REFRACTION) {
      if (!transmission_hemisphere_valid) {
        return true;
      }
      const float dot_ng_dir = dot(direction, transmission_normal);
      return dot_ng_dir * transmission_hemisphere_sign >= 0.0f;
    }
    if (!reflection_hemisphere_valid) {
      return true;
    }
    return dot(direction, reflection_normal) >= 0.0f;
  };

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
  MpgSeedScatter successful_scatter_branch = MPG_SEED_SCATTER_NONE;
  int guided_trials = 0;
  int fallback_trials = 0;
  float accepted_seed_pdf = 0.0f;
  float accepted_branch_pdf = 0.0f;
  float accepted_direction_pdf = 0.0f;
  float accepted_scatter_pdf = 0.0f;
  float3 accepted_direction_normalized = zero_float3();
  uint8_t accepted_tau_bits = 0;
  uint8_t accepted_tau_count = 0;

  const int uniform_attempt_budget = bootstrap_seed ? 32 : (use_uniform_fallback ? 32 : 16);
  const int guided_attempt_budget = use_uniform_sphere_sampling ? 0 : uniform_attempt_budget;
  const int fallback_attempt_budget = use_uniform_sphere_sampling ? uniform_attempt_budget : 32;
  int total_attempt_budget = guided_attempt_budget + fallback_attempt_budget;
  if (total_attempt_budget <= 0) {
    total_attempt_budget = 1;
  }
  const int repeat_trial_budget = std::max(options.max_seed_repeat_trials, 0);
  int seed_branch_count = total_attempt_budget + repeat_trial_budget;
  if (seed_branch_count <= 0) {
    seed_branch_count = 1;
  }
  const float guided_branch_probability =
      mpg_seed_branch_probability(guided_attempt_budget, fallback_attempt_budget, MPG_SEED_BRANCH_GUIDED);
  const float fallback_branch_probability =
      mpg_seed_branch_probability(guided_attempt_budget, fallback_attempt_budget, MPG_SEED_BRANCH_FALLBACK);
  const float jitter_one_minus_cos = one_minus_cos(jitter);
  const float fallback_one_minus_cos = one_minus_cos(0.6f * M_PI_F);

  float3 reflection_hemisphere_axis = reflection_normal;
  bool reflection_hemisphere_axis_valid = reflection_hemisphere_valid;
  if (reflection_hemisphere_axis_valid) {
    reflection_hemisphere_axis = safe_normalize(reflection_hemisphere_axis);
    reflection_hemisphere_axis_valid = !is_zero(reflection_hemisphere_axis);
  }

  float3 transmission_hemisphere_axis = transmission_normal;
  bool transmission_hemisphere_axis_valid = transmission_hemisphere_valid;
  if (transmission_hemisphere_axis_valid) {
    if (transmission_hemisphere_sign < 0.0f) {
      transmission_hemisphere_axis = -transmission_hemisphere_axis;
    }
    transmission_hemisphere_axis = safe_normalize(transmission_hemisphere_axis);
    transmission_hemisphere_axis_valid = !is_zero(transmission_hemisphere_axis);
  }

  const auto cone_acceptance = [&](const float3 &axis_dir,
                                   const bool axis_valid,
                                   const bool hemisphere_valid,
                                   const float3 &hemisphere_axis,
                                   const float one_minus_cos_angle) {
    if (!axis_valid || !hemisphere_valid) {
      return 1.0f;
    }
    return mpg_uniform_cone_hemisphere_acceptance(axis_dir, one_minus_cos_angle, hemisphere_axis);
  };

  const float reflection_guided_acceptance = cone_acceptance(
      axis_reflection, axis_reflection_valid, reflection_hemisphere_axis_valid, reflection_hemisphere_axis, jitter_one_minus_cos);
  const float transmission_guided_acceptance = cone_acceptance(
      axis_transmission, axis_transmission_valid, transmission_hemisphere_axis_valid, transmission_hemisphere_axis, jitter_one_minus_cos);
  const float reflection_fallback_acceptance = cone_acceptance(
      axis_reflection, axis_reflection_valid, reflection_hemisphere_axis_valid, reflection_hemisphere_axis, fallback_one_minus_cos);
  const float transmission_fallback_acceptance = cone_acceptance(
      axis_transmission, axis_transmission_valid, transmission_hemisphere_axis_valid, transmission_hemisphere_axis, fallback_one_minus_cos);
  const float reflection_uniform_acceptance = cone_acceptance(fallback_uniform_reflection_axis,
                                                              fallback_uniform_reflection_axis_valid,
                                                              reflection_hemisphere_axis_valid,
                                                              reflection_hemisphere_axis,
                                                              1.0f);
  const float transmission_uniform_acceptance = cone_acceptance(fallback_uniform_transmission_axis,
                                                                fallback_uniform_transmission_axis_valid,
                                                                transmission_hemisphere_axis_valid,
                                                                transmission_hemisphere_axis,
                                                                1.0f);

  const uint8_t base_tau_count = (selected_bounce_count > 0) ? static_cast<uint8_t>(selected_bounce_count) : 0;
  const float uniqueness_threshold = 1.0e-4f;

  auto trial_signatures_equivalent = [&](const Intersection &a,
                                         const float3 &dir_a,
                                         const uint8_t tau_bits_a,
                                         const uint8_t tau_count_a,
                                         const Intersection &b,
                                         const float3 &dir_b,
                                         const uint8_t tau_bits_b,
                                         const uint8_t tau_count_b) -> bool {
    if (tau_count_a != tau_count_b || tau_bits_a != tau_bits_b) {
      return false;
    }

    if (a.prim != b.prim || a.object != b.object || a.type != b.type) {
      return false;
    }

    const float bary_tolerance = 1.0e-5f;
    if (fabsf(a.u - b.u) > bary_tolerance || fabsf(a.v - b.v) > bary_tolerance) {
      return false;
    }

    const float ref_t = fabsf(b.t);
    const float relative_tolerance = 1.0e-4f * fmaxf(ref_t, 1.0f);
    if (fabsf(a.t - b.t) > relative_tolerance) {
      return false;
    }

    if (is_zero(dir_a) || is_zero(dir_b)) {
      return false;
    }

    const float dot_dir = fmaxf(-1.0f, fminf(1.0f, dot(dir_a, dir_b)));
    if (fabsf(dot_dir - 1.0f) >= uniqueness_threshold) {
      return false;
    }

    return true;
  };

  const uint32_t rejection_seed_offset = rng_state.rng_offset + branch_offset_u + 0x9e3779b9u;

  auto sample_conditioned_cone = [&](const float3 &axis_dir,
                                     const bool axis_valid,
                                     const float one_minus_cos_angle,
                                     const float acceptance,
                                     const bool hemisphere_valid,
                                     const float3 &hemisphere_axis,
                                     const float2 &initial_rand,
                                     const int sample_index,
                                     float3 &out_direction,
                                     float &out_pdf) -> bool {
    out_direction = zero_float3();
    out_pdf = 0.0f;

    if (!axis_valid) {
      return false;
    }

    if (!hemisphere_valid) {
      float unused_cos = 0.0f;
      out_direction = sample_uniform_cone(axis_dir, one_minus_cos_angle, initial_rand, &unused_cos, &out_pdf);
      return true;
    }

    if (!(acceptance > 0.0f)) {
      return false;
    }

    const float base_pdf = mpg_uniform_cone_pdf(one_minus_cos_angle);
    const float conditioned_pdf = base_pdf / acceptance;
    const int max_attempts = 16;
    float2 rand_dir = initial_rand;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
      float unused_cos = 0.0f;
      float unused_pdf = 0.0f;
      const float3 candidate = sample_uniform_cone(axis_dir, one_minus_cos_angle, rand_dir, &unused_cos, &unused_pdf);
      if (dot(candidate, hemisphere_axis) >= 0.0f) {
        out_direction = candidate;
        out_pdf = conditioned_pdf;
        return true;
      }

      const uint32_t attempt_seed = hash_uint4(rng_state.rng_pixel,
                                               uint(rng_state.sample),
                                               rejection_seed_offset,
                                               uint(sample_index) * 0x51633u + uint(attempt) + 1u);
      rand_dir = make_float2(uint_to_float_excl(attempt_seed),
                             uint_to_float_excl(hash_uint(attempt_seed ^ 0xa511e9b3u)));
    }

    return false;
  };

  auto sample_trial = [&](const int sample_index,
                          const float3 &rand_sample,
                          SeedTrialBranch &branch,
                          float &branch_pdf,
                          float3 &candidate_direction,
                          float &direction_pdf,
                          MpgSeedScatter &scatter_branch,
                          float &scatter_pdf) {
    branch = SeedTrialBranch::Fallback;
    branch_pdf = 1.0f;
    direction_pdf = 0.0f;
    candidate_direction = zero_float3();

    const bool has_guided_branch = guided_branch_probability > 0.0f;
    const bool has_fallback_branch = fallback_branch_probability > 0.0f;

    if (has_guided_branch && has_fallback_branch) {
      const float branch_sample = rand_sample.x;
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

    const float2 rand_dir = make_float2(rand_sample.y, rand_sample.z);

    scatter_branch = MPG_SEED_SCATTER_REFLECTION;
    scatter_pdf = 1.0f;

    const bool has_reflection_branch = reflection_probability > 0.0f;
    const bool has_transmission_branch = transmission_probability > 0.0f;

    if (has_reflection_branch && has_transmission_branch) {
      const uint32_t scatter_seed = hash_uint3(rng_state.rng_pixel,
                                               uint(rng_state.sample),
                                               rng_state.rng_offset + branch_offset_u +
                                                   uint(sample_index) * 0x51633u);
      const float scatter_rand = uint_to_float_excl(scatter_seed);
      const bool choose_reflection = scatter_rand < reflection_probability;
      scatter_branch = choose_reflection ? MPG_SEED_SCATTER_REFLECTION : MPG_SEED_SCATTER_REFRACTION;
      scatter_pdf = choose_reflection ? reflection_probability : transmission_probability;
    }
    else if (has_transmission_branch) {
      scatter_branch = MPG_SEED_SCATTER_REFRACTION;
      scatter_pdf = 1.0f;
    }
    else {
      scatter_branch = MPG_SEED_SCATTER_REFLECTION;
      scatter_pdf = 1.0f;
    }

    const bool branch_is_transmission = (scatter_branch == MPG_SEED_SCATTER_REFRACTION);
    const float3 branch_axis = branch_is_transmission ? axis_transmission : axis_reflection;
    const bool branch_axis_valid = branch_is_transmission ? axis_transmission_valid : axis_reflection_valid;
    const float3 branch_fallback_axis = branch_is_transmission ?
                                            fallback_uniform_transmission_axis :
                                            fallback_uniform_reflection_axis;
    const bool branch_fallback_axis_valid = branch_is_transmission ?
                                                fallback_uniform_transmission_axis_valid :
                                                fallback_uniform_reflection_axis_valid;
    const bool branch_uses_uniform_sphere = bootstrap_seed || !branch_axis_valid;

    const bool branch_hemisphere_valid = branch_is_transmission ? transmission_hemisphere_axis_valid :
                                                             reflection_hemisphere_axis_valid;
    const float3 branch_hemisphere_axis = branch_is_transmission ? transmission_hemisphere_axis :
                                                                  reflection_hemisphere_axis;

    if (branch == SeedTrialBranch::Guided) {
      const float branch_acceptance = branch_is_transmission ? transmission_guided_acceptance :
                                                               reflection_guided_acceptance;
      if (!sample_conditioned_cone(branch_axis,
                                   branch_axis_valid,
                                   jitter_one_minus_cos,
                                   branch_acceptance,
                                   branch_hemisphere_valid,
                                   branch_hemisphere_axis,
                                   rand_dir,
                                   sample_index,
                                   candidate_direction,
                                   direction_pdf))
      {
        direction_pdf = 0.0f;
        candidate_direction = zero_float3();
        return;
      }
    }
    else {
      if (!branch_uses_uniform_sphere) {
        const float branch_acceptance = branch_is_transmission ? transmission_fallback_acceptance :
                                                                 reflection_fallback_acceptance;
        if (!sample_conditioned_cone(branch_axis,
                                     branch_axis_valid,
                                     fallback_one_minus_cos,
                                     branch_acceptance,
                                     branch_hemisphere_valid,
                                     branch_hemisphere_axis,
                                     rand_dir,
                                     sample_index,
                                     candidate_direction,
                                     direction_pdf))
        {
          direction_pdf = 0.0f;
          candidate_direction = zero_float3();
          return;
        }
      }
      else if (branch_fallback_axis_valid) {
        const float branch_acceptance = branch_is_transmission ? transmission_uniform_acceptance :
                                                                 reflection_uniform_acceptance;
        if (!sample_conditioned_cone(branch_fallback_axis,
                                     true,
                                     1.0f,
                                     branch_acceptance,
                                     branch_hemisphere_valid,
                                     branch_hemisphere_axis,
                                     rand_dir,
                                     sample_index,
                                     candidate_direction,
                                     direction_pdf))
        {
          direction_pdf = 0.0f;
          candidate_direction = zero_float3();
          return;
        }
      }
      else {
        candidate_direction = sample_uniform_sphere(rand_dir);
        direction_pdf = M_1_4PI_F;
      }
    }
  };

  auto try_seed_sample = [&](const float3 &candidate_direction,
                             const float candidate_pdf,
                             const float candidate_branch_pdf,
                             const float candidate_direction_pdf,
                             const MpgSeedScatter scatter_branch,
                             const float candidate_scatter_pdf,
                             Intersection &out_isect,
                             const SeedTrialBranch branch,
                             const bool record_accept,
                             float3 &out_normalized_direction,
                             uint8_t &out_tau_bits,
                             uint8_t &out_tau_count) -> bool {
    int &branch_trials = (branch == SeedTrialBranch::Guided) ? guided_trials : fallback_trials;
    ++branch_trials;
    out_normalized_direction = zero_float3();
    out_tau_bits = 0;
    out_tau_count = 0;
    if (is_zero(candidate_direction) || candidate_pdf <= 0.0f || candidate_scatter_pdf <= 0.0f) {
      last_failure = MPG_FAILURE_INVALID_SEED_PDF;
      return false;
    }

    /* Bootstrap seeds and uniform fallbacks still probe broadly but must respect the
     * hemisphere test so rays never shoot across Ng. Directional seeds continue to obey
     * the same filtering. */
    if (!matches_branch_hemisphere(candidate_direction, scatter_branch)) {
      last_failure = MPG_FAILURE_SEED;
      return false;
    }

    const float3 normalized_direction = safe_normalize(candidate_direction);
    if (is_zero(normalized_direction)) {
      last_failure = MPG_FAILURE_INVALID_SEED_PDF;
      return false;
    }
    out_normalized_direction = normalized_direction;
    out_tau_count = base_tau_count;
    if (base_tau_count > 0 && scatter_branch == MPG_SEED_SCATTER_REFRACTION) {
      out_tau_bits = 1u;
    }

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
    if (record_accept) {
      seed_direction = normalized_direction;
      accepted_seed_pdf = candidate_pdf;
      accepted_branch_pdf = candidate_branch_pdf;
      accepted_direction_pdf = candidate_direction_pdf;
      accepted_scatter_pdf = candidate_scatter_pdf;
      seed.use_smooth_normals = has_smooth_normals && (scatter_branch != MPG_SEED_SCATTER_REFRACTION);
      successful_branch = branch;
      successful_scatter_branch = scatter_branch;
      accepted_direction_normalized = normalized_direction;
      accepted_tau_bits = out_tau_bits;
      accepted_tau_count = out_tau_count;
    }
    return true;
  };

  for (int attempt = 0; attempt < total_attempt_budget && !seed_valid; ++attempt) {
    const float3 rand = path_branched_rng_3D(
        kg, &rng_state, attempt + rng_branch_offset, seed_branch_count, PRNG_SURFACE_BSDF);

    SeedTrialBranch branch = SeedTrialBranch::Fallback;
    float branch_pdf = 1.0f;
    float direction_pdf = 0.0f;
    float3 candidate_direction = zero_float3();
    MpgSeedScatter scatter_branch = MPG_SEED_SCATTER_REFLECTION;
    float scatter_pdf = 1.0f;

    sample_trial(attempt, rand, branch, branch_pdf, candidate_direction, direction_pdf, scatter_branch, scatter_pdf);

    const float candidate_pdf = direction_pdf * branch_pdf * scatter_pdf;
    float3 candidate_normalized_direction = zero_float3();
    uint8_t candidate_tau_bits = 0;
    uint8_t candidate_tau_count = 0;
    seed_valid = try_seed_sample(candidate_direction,
                                 candidate_pdf,
                                 branch_pdf,
                                 direction_pdf,
                                 scatter_branch,
                                 scatter_pdf,
                                 isect,
                                 branch,
                                 true,
                                 candidate_normalized_direction,
                                 candidate_tau_bits,
                                 candidate_tau_count);
    if (seed_valid) {
      accepted_direction_normalized = candidate_normalized_direction;
      accepted_tau_bits = candidate_tau_bits;
      accepted_tau_count = candidate_tau_count;
    }
  }

  if (!seed_valid) {
    failure_code = last_failure;
    return false;
  }

  const int guided_trials_initial = guided_trials;
  const int fallback_trials_initial = fallback_trials;
  const Intersection accepted_isect = isect;

  int equivalent_successes = 0;

  for (int extra = 0; extra < repeat_trial_budget; ++extra) {
    const int attempt_index = total_attempt_budget + extra;
    const float3 rand = path_branched_rng_3D(
        kg, &rng_state, attempt_index + rng_branch_offset, seed_branch_count, PRNG_SURFACE_BSDF);

    SeedTrialBranch branch = SeedTrialBranch::Fallback;
    float branch_pdf = 1.0f;
    float direction_pdf = 0.0f;
    float3 candidate_direction = zero_float3();
    MpgSeedScatter scatter_branch = MPG_SEED_SCATTER_REFLECTION;
    float scatter_pdf = 1.0f;

    sample_trial(attempt_index, rand, branch, branch_pdf, candidate_direction, direction_pdf, scatter_branch, scatter_pdf);

    const float candidate_pdf = direction_pdf * branch_pdf * scatter_pdf;
    Intersection repeat_isect = {};
    float3 repeat_normalized_direction = zero_float3();
    uint8_t repeat_tau_bits = 0;
    uint8_t repeat_tau_count = 0;
    const bool repeat_success = try_seed_sample(candidate_direction,
                                                candidate_pdf,
                                                branch_pdf,
                                                direction_pdf,
                                                scatter_branch,
                                                scatter_pdf,
                                                repeat_isect,
                                                branch,
                                                false,
                                                repeat_normalized_direction,
                                                repeat_tau_bits,
                                                repeat_tau_count);

    if (!repeat_success) {
      continue;
    }

    if (branch != successful_branch || scatter_branch != successful_scatter_branch) {
      continue;
    }

    if (!trial_signatures_equivalent(repeat_isect,
                                     repeat_normalized_direction,
                                     repeat_tau_bits,
                                     repeat_tau_count,
                                     accepted_isect,
                                     accepted_direction_normalized,
                                     accepted_tau_bits,
                                     accepted_tau_count)) {
      continue;
    }

    ++equivalent_successes;
  }

  const int total_trials = guided_trials + fallback_trials;
  if (total_trials <= 0) {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }
  const int accepted_trials_initial = (successful_branch == SeedTrialBranch::Guided) ?
                                          guided_trials_initial :
                                          fallback_trials_initial;
  if (accepted_trials_initial <= 0) {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }

  const int success_count = equivalent_successes + 1;
  const float total_trials_f = float(total_trials);
  const float success_count_f = float(success_count);
  if (!(isfinite_safe(total_trials_f) && total_trials_f > 0.0f && success_count_f > 0.0f)) {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }

  const float expected_trials = total_trials_f / success_count_f;
  if (!(isfinite_safe(expected_trials) && expected_trials > 0.0f)) {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }

  const float acceptance_probability = 1.0f / fmaxf(expected_trials, 1.0e-16f);
  if (!(isfinite_safe(acceptance_probability) && acceptance_probability > 0.0f)) {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }

  const float normalized_pdf = accepted_seed_pdf * acceptance_probability;
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
  seed.direction_normalized = accepted_direction_normalized;
  const float combined_seed_pdf_raw = accepted_seed_pdf * seed.bounce_pdf_raw;
  const float combined_seed_pdf = normalized_pdf * seed.bounce_pdf;

  if (!(isfinite_safe(combined_seed_pdf_raw) && combined_seed_pdf_raw > 0.0f) ||
      !(isfinite_safe(combined_seed_pdf) && combined_seed_pdf > 0.0f))
  {
    failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return false;
  }

  seed.seed_pdf_raw = combined_seed_pdf_raw;
  seed.seed_branch_pdf = accepted_branch_pdf;
  seed.seed_direction_pdf = accepted_direction_pdf;
  seed.seed_scatter_pdf = fmaxf(accepted_scatter_pdf, 1.0e-16f);
  seed.trial_count = total_trials;
  seed.accepted_trial_count = accepted_trials_initial;
  seed.guided_trial_count = guided_trials;
  seed.fallback_trial_count = fallback_trials;
  seed.seed_resample_factor = fmaxf(expected_trials, 1.0e-16f);
  seed.branch = (successful_branch == SeedTrialBranch::Guided) ? MPG_SEED_BRANCH_GUIDED :
                                                                    MPG_SEED_BRANCH_FALLBACK;
  seed.scatter = successful_scatter_branch;
  seed.tau_bits = accepted_tau_bits;
  seed.tau_count = accepted_tau_count;
  seed.seed_pdf = fmaxf(combined_seed_pdf, 1.0e-16f);
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