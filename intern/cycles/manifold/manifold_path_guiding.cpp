/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Manifold Path Guiding (MPG) - Cycles implementation
 *
 * This file matches the structure of the Mitsuba MPG reference implementation:
 * https://github.com/mollnn/manifold-path-guiding/tree/main/mitsuba/src/integrators/MPG
 *
 * File corresponds to: manifold_path_guiding.cpp in the Mitsuba reference
 *
 * This file consolidates all MPG implementation for easier comparison with Mitsuba:
 * - Main entry point (mpg_try_connect)
 * - Seed generation
 * - Newton solvers (single and double bounce)
 * - PDF evaluation
 *
 * Original Cycles implementation split across:
 * - mpg.cpp (main entry)
 * - mpg_seed.cpp (seed generation)
 * - mpg_solve.cpp (Newton solvers)
 * - mpg_pdf.cpp (PDF evaluation)
 */

#include "manifold/manifold_path_guiding.h"


#include "kernel/bvh/bvh.h"
#include "kernel/bvh/util.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/geom/object.h"
#include "kernel/geom/triangle.h"
#include "kernel/light/light.h"
#include "kernel/light/distribution.h"
#include "kernel/light/tree.h"
#include "kernel/light/common.h"
#include "kernel/light/sample.h"
#include "kernel/sample/mapping.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/device/cpu/globals.h"
#include "kernel/integrator/path_state.h"
#include "kernel/svm/types.h"
#include "kernel/types.h"

#include "util/color.h"
#include "util/log.h"
#include "util/math.h"
#include "util/math_base.h"
#include "util/math_float4.h"
#include "util/math_intersect.h"
#include "util/hash.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#ifdef WITH_CYCLES_DEBUG
#  include "util/log.h"
#endif

CCL_NAMESPACE_BEGIN


/* ========================================================================
 * PDF Evaluation
 * From: mpg_pdf.cpp
 * ======================================================================== */


float mpg_light_sample_pdf_solid(KernelGlobals kg,
                                 const ShaderData &sd,
                                 const LightSample &light_sample)
{
  (void)kg;
  (void)sd;

  if (!(isfinite_safe(light_sample.pdf) && light_sample.pdf > 0.0f)) {
    return 0.0f;
  }

  float pdf_solid = light_sample.pdf;

  if (light_sample.t != FLT_MAX) {
    bool needs_conversion = false;

    switch (light_sample.type) {
      case LIGHT_POINT:
      case LIGHT_SPOT:
        needs_conversion = true;
        break;
      case LIGHT_AREA:
        if (light_sample.prim != PRIM_NONE) {
          const ccl_global KernelLight *klight = &kernel_data_fetch(lights, light_sample.prim);
          /* Area lights that sample from an elliptical shape keep the PDF in area
           * measure even when the spread is zero. Convert those (and any spread
           * configurations) back to solid angle to match Mitsuba's measure. */
          const bool is_elliptical = (klight->area.invarea < 0.0f);
          const bool has_spread = (klight->area.tan_half_spread != 0.0f);
          needs_conversion = (has_spread || is_elliptical);
        }
        break;
      default:
        break;
    }

    if (needs_conversion) {
      float3 light_dir = light_sample.D;
      if (is_zero(light_dir)) {
        return 0.0f;
      }
      light_dir = normalize(light_dir);

      const float jacobian = light_pdf_area_to_solid_angle(
          light_sample.Ng, -light_dir, light_sample.t);

      if (!(isfinite_safe(jacobian) && jacobian > 0.0f)) {
        return 0.0f;
      }

      pdf_solid *= jacobian;
    }
  }

  return pdf_solid;
}

bool mpg_evaluate_pdf(KernelGlobals kg,
                      const ShaderData &sd,
                      const ShaderClosure &bsdf,
                      const GuideSummary &guide,
                      const MpgSeedRay &seed,
                      const MpgSolverOutput &solution,
                      float &pdf)
{
  (void)bsdf;
  (void)guide;

  pdf = 0.0f;

  const float p_seed_rebuilt = mpg_rebuild_seed_pdf(seed);
  if (!(isfinite_safe(p_seed_rebuilt) && p_seed_rebuilt > 0.0f)) {
    return false;
  }
  const float p_seed = fmaxf(p_seed_rebuilt, 1.0e-16f);

  const int vertex_count = solution.specular_vertex_count;
  if (vertex_count <= 0) {
    return false;
  }

  LightSample light_sample = seed.light_sample;
  const float pdf_selection = light_sample.pdf_selection;
  if (!(isfinite_safe(pdf_selection) && pdf_selection > 0.0f)) {
    return false;
  }

  light_sample.pdf /= pdf_selection;
  light_sample.pdf_selection = 1.0f;

  const MpgSpecularVertex &exit_vertex = solution.specular_vertices[vertex_count - 1];
  const bool exit_is_refraction = exit_vertex.is_refraction;
#ifdef WITH_CYCLES_DEBUG
  DCHECK(solution.is_refraction == exit_is_refraction);
#endif
  uint32_t updated_path_flag = seed.path_flag;
  if (exit_is_refraction) {
    updated_path_flag |= PATH_RAY_MIS_HAD_TRANSMISSION;
  }
  light_sample_update(kg, &light_sample, exit_vertex.position, exit_vertex.normal, updated_path_flag);

  light_sample.pdf *= pdf_selection;
  light_sample.pdf_selection = pdf_selection;

  float p_light = mpg_light_sample_pdf_solid(kg, sd, light_sample);
  if (!isfinite_safe(p_light) || p_light <= 0.0f) {
    return false;
  }
  p_light = fmaxf(p_light, 1.0e-16f);

  float J = fabsf(solution.jacobian_total);
  if (!isfinite_safe(J) || J <= 0.0f) {
    return false;
  }
  J = fmaxf(J, 1.0e-16f);

  const float pdf_product = p_seed * p_light * J;
  if (!isfinite_safe(pdf_product) || pdf_product <= 0.0f) {
    return false;
  }

#ifdef WITH_CYCLES_DEBUG
  if (p_seed > 0.0f && isfinite_safe(p_seed) &&
      isfinite_safe(seed.seed_pdf) && seed.seed_pdf > 0.0f)
  {
    const float tolerance = fmaxf(fabsf(seed.seed_pdf), 1.0e-16f) * 1.0e-4f;
    DCHECK(fabsf(seed.seed_pdf - p_seed) <= tolerance);
  }
#endif

  pdf = fmaxf(pdf_product, 1.0e-16f);
  return true;
}


/* ========================================================================
 * Seed Generation
 * From: mpg_seed.cpp
 * ======================================================================== */


/* Debug printing toggles - set categories to true to enable specific debug output */
struct MPG_DEBUG {
  /* Seed generation and acceptance/rejection */
  static constexpr bool SEED = false;
};

/* Bit manipulation functions for full-path tau encoding (matching Mitsuba MPG reference) */

ccl_device_forceinline void set_chaintype_bit(uint8_t &tau, int position, bool is_refraction)
{
  /* Clear bit at position, then set it if refraction */
  tau &= ~(1u << position);
  if (is_refraction) {
    tau |= (1u << position);
  }
}

ccl_device_forceinline bool get_chaintype_bit(uint8_t tau, int position)
{
  /* Extract bit at position: 1 = refraction, 0 = reflection */
  return ((tau >> position) & 1u) != 0u;
}

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
    const bool is_glass    = CLOSURE_IS_GLASS(c->type);

    /* Glass BSDFs are specular surfaces that MPG can use as intermediate vertices.
     * CLOSURE_IS_BSDF_SINGULAR only includes transparent/portal, not glass. */
    if (is_singular || is_glass) {
      return true;
    }
    if (is_micro) {
      const MicrofacetBsdf *mf = reinterpret_cast<const MicrofacetBsdf *>(c);
      const float a = fmaxf(mf->alpha_x, mf->alpha_y);
      /* Per Codex finding: Match solver threshold (1e-6) to avoid accepting seeds
       * that will later be rejected. The solver only supports near-delta microfacets,
       * so seed generation must use the same criterion. */
      if (a <= 1e-6f) {
        return true; /* near-delta microfacet behaves like perfect specular */
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
    /* For non-specular BSDFs (diffuse, glossy), return Dual to explore both
     * reflection and refraction paths through specular surfaces in the scene.
     * MPG is about finding specular surfaces from diffuse starting points,
     * so we shouldn't constrain the search based on the starting BSDF type. */
    return SeedLobe::Dual;
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
  float3 transmission_hemisphere_normal = transmission_normal;
  bool transmission_hemisphere_valid = !is_zero(transmission_hemisphere_normal);
  if (transmission_hemisphere_valid && geometric_normal_valid) {
    if (transmission_dot_wi < 0.0f) {
      transmission_hemisphere_normal = -transmission_hemisphere_normal;
    }
  }
  if (transmission_hemisphere_valid) {
    transmission_hemisphere_normal = safe_normalize(transmission_hemisphere_normal);
    transmission_hemisphere_valid = !is_zero(transmission_hemisphere_normal);
  }
  const float hemisphere_epsilon = 1.0e-5f;

  /* Compute reflection vs transmission probability following Mitsuba reference.
   * For dual-lobe materials (glass), use Fresnel equations to determine probability.
   * This is simpler and more physically accurate than complex heuristics. */
  float reflection_probability = 0.5f;
  float transmission_probability = 0.5f;

  if (seed_lobe == SeedLobe::Transmission) {
    reflection_probability = 0.0f;
    transmission_probability = 1.0f;
  }
  else if (seed_lobe == SeedLobe::Reflection) {
    reflection_probability = 1.0f;
    transmission_probability = 0.0f;
  }
  else if (seed_lobe == SeedLobe::Dual) {
    /* Use Fresnel equations to compute reflection probability, matching Mitsuba reference.
     * This provides physically-based probability based on incident angle and IOR. */
    float fresnel_reflection = 0.5f;
    if (mpg_compute_dielectric_reflection_probability(
            kg, sd, bsdf, shading_normal, fresnel_reflection))
    {
      reflection_probability = fresnel_reflection;
      transmission_probability = 1.0f - fresnel_reflection;
    }
    /* If Fresnel computation fails, use 50/50 split as fallback */
  }

  /* Normalize probabilities to ensure they sum to 1.0 */
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

  /* Determine preferred scattering mode based on computed probabilities.
   * Used later for directional sampling guidance. */
  bool prefer_transmission = (transmission_probability > reflection_probability);
  bool prefer_transmission_decided = (fabsf(transmission_probability - reflection_probability) > 1.0e-5f);

#if 0 // WITH_CYCLES_DEBUG disabled for performance
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

  if (axis_valid && transmission_hemisphere_valid) {
    const float dot_axis_transmission = dot(axis, transmission_hemisphere_normal);
    if (fabsf(dot_axis_transmission) > hemisphere_epsilon && dot_axis_transmission >= 0.0f) {
      prefer_transmission = true;
      prefer_transmission_decided = true;
    }
  }

  const bool reflection_hemisphere_valid = !is_zero(reflection_normal);

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
    float dot_axis_transmission = dot(axis_transmission, transmission_hemisphere_normal);
    if (fabsf(dot_axis_transmission) > hemisphere_epsilon && dot_axis_transmission < 0.0f) {
      axis_transmission = safe_normalize(-axis_transmission);
      axis_transmission_valid = !is_zero(axis_transmission);
      dot_axis_transmission = dot(axis_transmission, transmission_hemisphere_normal);
    }
    if (axis_transmission_valid && dot_axis_transmission < 0.0f) {
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
    float3 transmission_axis = transmission_hemisphere_normal;
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
      const float dot_ng_dir = dot(direction, transmission_hemisphere_normal);
      return dot_ng_dir >= 0.0f;
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

  const bool prefer_double_for_refraction = allow_double_pdf &&
                                            prefer_transmission &&
                                            bootstrap_seed;

  if (!allow_single && allow_double_pdf) {
    selected_bounce_count = 2;
    selected_bounce_pdf = pdf_double;
  }
  else if (allow_single && allow_double_pdf) {
    if (prefer_double_for_refraction || bounce_rand >= pdf_single) {
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

  /* Match Mitsuba reference: use uniform hemisphere/sphere sampling without tight cone restrictions.
   * The reference implementation doesn't use angular jitter or tight cones - it samples uniformly
   * and relies on the guide distribution (ChainDistribution) for directional bias. */
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

  /* Increase attempt budgets to match Mitsuba's more generous sampling strategy.
   * Mitsuba allows up to 1e6 trials; we use 256 as a practical compromise for performance.
   * This gives seeds much better chance of finding specular surfaces. */
  const int uniform_attempt_budget = bootstrap_seed ? 256 : (use_uniform_fallback ? 256 : 128);
  const int guided_attempt_budget = use_uniform_sphere_sampling ? 0 : uniform_attempt_budget;
  const int fallback_attempt_budget = use_uniform_sphere_sampling ? uniform_attempt_budget : 256;
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

  /* Hemisphere axes for uniform sampling (Mitsuba reference approach) */
  float3 reflection_hemisphere_axis = reflection_normal;
  bool reflection_hemisphere_axis_valid = reflection_hemisphere_valid;
  if (reflection_hemisphere_axis_valid) {
    reflection_hemisphere_axis = safe_normalize(reflection_hemisphere_axis);
    reflection_hemisphere_axis_valid = !is_zero(reflection_hemisphere_axis);
  }

  float3 transmission_hemisphere_axis = transmission_hemisphere_normal;
  bool transmission_hemisphere_axis_valid = transmission_hemisphere_valid;
  if (transmission_hemisphere_axis_valid) {
    transmission_hemisphere_axis = safe_normalize(transmission_hemisphere_axis);
    transmission_hemisphere_axis_valid = !is_zero(transmission_hemisphere_axis);
  }

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

    /* Match Mitsuba reference: use uniform hemisphere sampling instead of tight cones.
     * The reference doesn't use cone-based directional constraints - it samples uniformly
     * over the appropriate hemisphere and relies on the learned distribution for bias. */
    const bool branch_is_transmission = (scatter_branch == MPG_SEED_SCATTER_REFRACTION);
    const bool branch_hemisphere_valid = branch_is_transmission ? transmission_hemisphere_axis_valid :
                                                             reflection_hemisphere_axis_valid;
    const float3 branch_hemisphere_axis = branch_is_transmission ? transmission_hemisphere_axis :
                                                                  reflection_hemisphere_axis;

    if (branch_hemisphere_valid) {
      /* Sample uniformly over hemisphere aligned with the scattering normal */
      sample_uniform_hemisphere(branch_hemisphere_axis, rand_dir, &candidate_direction, &direction_pdf);
    }
    else {
      /* No valid hemisphere - sample full sphere uniformly */
      candidate_direction = sample_uniform_sphere(rand_dir);
      direction_pdf = M_1_4PI_F;  /* 1/(4π) for uniform sphere */
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
if constexpr (MPG_DEBUG::SEED) {
      /* Debug: Print what surface we hit and why it was rejected */
      ShaderData debug_sd = {};
      shader_setup_from_ray(kg, &debug_sd, &ray, const_cast<Intersection *>(&candidate_isect));
      const ConstIntegratorState integrator_state = nullptr;
      surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
          kg, integrator_state, &debug_sd, nullptr, PATH_RAY_CAMERA, true);

      printf("MPG SEED REJECTION: Hit non-specular surface\n");
      printf("  Ray origin: (%.6f, %.6f, %.6f)\n", ray.P.x, ray.P.y, ray.P.z);
      printf("  Ray direction: (%.6f, %.6f, %.6f)\n", ray.D.x, ray.D.y, ray.D.z);
      printf("  Hit position: (%.6f, %.6f, %.6f)\n", debug_sd.P.x, debug_sd.P.y, debug_sd.P.z);
      printf("  Hit object: %d, prim: %d\n", candidate_isect.object, candidate_isect.prim);
      printf("  Number of closures: %d\n", debug_sd.num_closure);

      for (int i = 0; i < debug_sd.num_closure; ++i) {
        const ShaderClosure *c = &debug_sd.closure[i];
        const bool is_bsdf = CLOSURE_IS_BSDF(c->type);
        const bool is_glass = CLOSURE_IS_GLASS(c->type);
        const bool is_singular = CLOSURE_IS_BSDF_SINGULAR(c->type);
        const bool is_micro = CLOSURE_IS_BSDF_MICROFACET(c->type);

        printf("  Closure %d: type=%d, is_bsdf=%d, is_glass=%d, is_singular=%d, is_micro=%d\n",
               i, (int)c->type, is_bsdf, is_glass, is_singular, is_micro);

        if (is_micro) {
          const MicrofacetBsdf *mf = reinterpret_cast<const MicrofacetBsdf *>(c);
          printf("    Microfacet: alpha_x=%.6f, alpha_y=%.6f, ior=%.6f\n",
                 mf->alpha_x, mf->alpha_y, mf->ior);
        }
      }
}
      last_failure = MPG_FAILURE_NO_SPECULAR;
      return false;
    }

    /* Per Codex finding: Set tau_bits based on ACTUAL surface BSDF type, not scatter_branch proposal.
     * The scatter_branch is a probabilistic guess, but tau hints must encode what actually happened.
     * Query the hit surface to determine if it's refractive or reflective. */
    if (base_tau_count > 0) {
      ShaderData hit_sd = {};
      shader_setup_from_ray(kg, &hit_sd, &ray, const_cast<Intersection *>(&candidate_isect));
      const ConstIntegratorState integrator_state = nullptr;
      surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
          kg, integrator_state, &hit_sd, nullptr, PATH_RAY_CAMERA, true);

      /* Determine if the hit surface actually supports refraction.
       * Check for glass (both reflection+transmission) or pure transmission closures. */
      bool surface_has_refraction = false;
      bool surface_has_reflection_only = false;

      for (int i = 0; i < hit_sd.num_closure; ++i) {
        const ShaderClosure *closure = &hit_sd.closure[i];
        if (CLOSURE_IS_BSDF(closure->type)) {
          if (CLOSURE_IS_BSDF_SINGULAR(closure->type) || CLOSURE_IS_BSDF_MICROFACET(closure->type)) {
            /* Check if this closure supports transmission/refraction */
            if (CLOSURE_IS_GLASS(closure->type) || CLOSURE_IS_BSDF_TRANSMISSION(closure->type)) {
              surface_has_refraction = true;
            }
            /* Glass supports both, but pure glossy/diffuse are reflection only.
             * If it's not transmission and not glass, it's reflection only. */
            else {
              surface_has_reflection_only = true;
            }
          }
        }
      }

      /* Determine actual scatter type based on what the surface supports */
      bool is_refraction = (scatter_branch == MPG_SEED_SCATTER_REFRACTION);
      if (surface_has_refraction && !surface_has_reflection_only) {
        /* Surface only supports refraction (pure transmission or glass with no other closures) */
        is_refraction = true;
      }
      else if (surface_has_reflection_only && !surface_has_refraction) {
        /* Surface only supports reflection (no transmission) */
        is_refraction = false;
      }

      /* Set bit 0 for primary bounce based on actual surface type */
      set_chaintype_bit(out_tau_bits, 0, is_refraction);

      /* For double-bounce refraction, set bit 1 for secondary bounce */
      if (base_tau_count >= 2 && is_refraction) {
        set_chaintype_bit(out_tau_bits, 1, true);
      }
    }

    out_isect = candidate_isect;
    if (record_accept) {
if constexpr (MPG_DEBUG::SEED) {
      /* Debug: Print successful seed acceptance */
      ShaderData debug_sd = {};
      shader_setup_from_ray(kg, &debug_sd, &ray, const_cast<Intersection *>(&candidate_isect));
      printf("MPG SEED ACCEPTED: Found specular surface\n");
      printf("  Ray origin: (%.6f, %.6f, %.6f)\n", ray.P.x, ray.P.y, ray.P.z);
      printf("  Ray direction: (%.6f, %.6f, %.6f)\n", ray.D.x, ray.D.y, ray.D.z);
      printf("  Hit position: (%.6f, %.6f, %.6f)\n", debug_sd.P.x, debug_sd.P.y, debug_sd.P.z);
      printf("  Hit object: %d, scatter: %d, smooth_normals: %d\n",
             candidate_isect.object, (int)scatter_branch, has_smooth_normals);
}
      seed_direction = normalized_direction;
      accepted_seed_pdf = candidate_pdf;
      accepted_branch_pdf = candidate_branch_pdf;
      accepted_direction_pdf = candidate_direction_pdf;
      accepted_scatter_pdf = candidate_scatter_pdf;
      /* Force flat shading for ALL MPG paths to ensure tangent basis orthogonality.
       * For smooth normals, the interpolated normal may not be orthogonal to the
       * geometric tangent basis (dXdu, dXdv), causing Newton solver divergence.
       * Mitsuba MPG uses geometric normals for manifold constraints. */
      seed.use_smooth_normals = false;
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


/* ========================================================================
 * Newton Manifold Solvers (Single and Double Bounce)
 * From: mpg_solve.cpp
 * ======================================================================== */


/* Thread-local storage for current sample number (set by mpg_try_connect) */
static thread_local int g_current_sample = -1;

/* Forward declaration for debug helper (defined after MPG_DEBUG struct) */
static inline bool debug_print_enabled();

/* Debug printing toggles - set categories to true to enable specific debug output */
struct MPG_DEBUG {
  /* Sample number filter: Only print debug output for this sample number.
   * Set to -1 to print all samples, or a specific number (e.g., 127 for sample 127).
   * Example: SAMPLE_FILTER = 127 means only print on sample 127 (0-indexed). */
  static constexpr int SAMPLE_FILTER = 127;  // -1 = all samples, or set to specific sample number

  /* Base flags - these are AND'ed with sample filter automatically via helper functions below */
  static constexpr bool SEED_BASE = true;
  static constexpr bool GEOMETRY_BASE = true;
  static constexpr bool PARAMS_BASE = true;
  static constexpr bool NEWTON_BASE = true;
  static constexpr bool NEWTON_DETAIL_BASE = true;
  static constexpr bool EVAL_FAIL_BASE = true;
  static constexpr bool SUCCESS_BASE = true;

  /* Helper functions that combine flag with sample filter - use these in if statements */
  static inline bool SEED() { return SEED_BASE && debug_print_enabled(); }
  static inline bool GEOMETRY() { return GEOMETRY_BASE && debug_print_enabled(); }
  static inline bool PARAMS() { return PARAMS_BASE && debug_print_enabled(); }
  static inline bool NEWTON() { return NEWTON_BASE && debug_print_enabled(); }
  static inline bool NEWTON_DETAIL() { return NEWTON_DETAIL_BASE && debug_print_enabled(); }
  static inline bool EVAL_FAIL() { return EVAL_FAIL_BASE && debug_print_enabled(); }
  static inline bool SUCCESS() { return SUCCESS_BASE && debug_print_enabled(); }
};

/* Helper to check if debug printing is enabled for current sample */
static inline bool debug_print_enabled() {
  return (MPG_DEBUG::SAMPLE_FILTER == -1) || (g_current_sample == MPG_DEBUG::SAMPLE_FILTER);
}

/* Setter for sample number (called from mpg.cpp to avoid thread_local extern issues) */
void mpg_set_current_sample(int sample) {
  g_current_sample = sample;
}

namespace {
struct SpecularSurfaceGeometry {
  float3 verts[3];
  float3 normals[3];
  float3 dPdu;
  float3 dPdv;
};

struct SpecularParameters {
  bool is_refraction = false;
  float base_eta = 1.0f;
  bool backfacing = false;  /* True if ray hits back face (exiting for closed objects) */
  bool has_microfacet = false;
  MicrofacetBsdf microfacet = {};
  FresnelDielectricTint fresnel_dielectric_tint = {};
  FresnelConductor fresnel_conductor = {};
  FresnelGeneralizedSchlick fresnel_generalized_schlick = {};
  FresnelF82Tint fresnel_f82_tint = {};
  Spectrum singular_reflection_weight = zero_spectrum();
  bool use_singular_reflection_weight = false;
  float3 normal = zero_float3();
  bool has_normal = false;
  bool has_conductor_fresnel = false;
  ComplexIOR<Spectrum> conductor_ior = {};
};

struct SpecularEval {
  float3 point = zero_float3();
  float3 normal = zero_float3();
  float3 dir_ds = zero_float3();
  float3 dir_sl = zero_float3();
  float distance_ds = 0.0f;
  float distance_sl = 0.0f;
  float3 dXdu = zero_float3();
  float3 dXdv = zero_float3();
  float3 dNdu = zero_float3();
  float3 dNdv = zero_float3();
  float3 residual = zero_float3();
  float cos_theta_i = 0.0f;
  float cos_theta_t = 0.0f;
  float eta = 1.0f;
  bool tir = false;
  bool refractive = false;
  LightSample light_sample = {};  /* Stored for analytical Jacobian computation */
  /* Per Codex: BSDF frame tangents (orthogonal to BSDF frame normal).
   * Mitsuba uses frame.s and frame.t, not geometric dPdu/dPdv. */
  float3 tangent_u = zero_float3();
  float3 tangent_v = zero_float3();
};

float3 surface_point_from_barycentric(const SpecularSurfaceGeometry &geometry, const float u, const float v);
float3 surface_normal_from_barycentric(const SpecularSurfaceGeometry &geometry, const float u, const float v);
float3 combine_vertex_normals(const SpecularSurfaceGeometry &geometry, const float u, const float v);

/* Bit manipulation functions for full-path tau encoding (matching Mitsuba MPG reference) */

ccl_device_forceinline void set_chaintype_bit(uint8_t &tau, int position, bool is_refraction)
{
  /* Clear bit at position, then set it if refraction */
  tau &= ~(1u << position);
  if (is_refraction) {
    tau |= (1u << position);
  }
}

ccl_device_forceinline bool get_chaintype_bit(uint8_t tau, int position)
{
  /* Extract bit at position: 1 = refraction, 0 = reflection */
  return ((tau >> position) & 1u) != 0u;
}

float3 compute_distant_visibility_endpoint(const LightSample &light_sample, const float3 &origin)
{
  if (light_sample.t != FLT_MAX) {
    return light_sample.P;
  }

  float3 dir = light_sample.D;
  if (is_zero(dir)) {
    return origin;
  }

  dir = normalize(dir);
  return origin + dir * MPG_DISTANT_LIGHT_VISIBILITY_DISTANCE;
}

bool build_primary_shading_data(const ShaderData &receiver_sd,
                                const SpecularSurfaceGeometry &geometry,
                                const MpgSeedRay &seed,
                                const float u,
                                const float v,
                                ShaderData &out_sd)
{
  out_sd = ShaderData();
  out_sd.P = surface_point_from_barycentric(geometry, u, v);
  out_sd.time = receiver_sd.time;
  out_sd.prim = seed.prim;
  out_sd.object = seed.object;

  /* Use mesh normals directly without flipping based on ray direction.
   * For flat shading, use geometry.normals[0] (preserves mesh normal from load_surface_geometry).
   * For smooth shading, interpolate vertex normals.
   * This ensures thin glass surfaces maintain their opposing normals. */
  float3 geom_normal;
  if (seed.use_smooth_normals) {
    geom_normal = combine_vertex_normals(geometry, u, v);
  }
  else {
    geom_normal = geometry.normals[0];
  }

  if (!is_zero(geom_normal)) {
    geom_normal = safe_normalize(geom_normal);
  }
  else {
    /* Fallback to geometric normal only if mesh normal is degenerate */
    geom_normal = safe_normalize(cross(geometry.dPdu, geometry.dPdv));
  }

  if (is_zero(geom_normal)) {
    return false;
  }

  /* Do NOT flip normal based on ray direction - use mesh normal as-is */
  out_sd.Ng = geom_normal;
  out_sd.N = geom_normal;
  return true;
}

void copy_microfacet_to_parameters(const MicrofacetBsdf *microfacet, SpecularParameters &params)
{
  params.has_microfacet = (microfacet != nullptr);
  if (!params.has_microfacet) {
    return;
  }

  params.microfacet = *microfacet;
  params.microfacet.fresnel = nullptr;

  const MicrofacetFresnel fresnel_type = static_cast<MicrofacetFresnel>(microfacet->fresnel_type);
  switch (fresnel_type) {
    case MicrofacetFresnel::DIELECTRIC_TINT:
      if (microfacet->fresnel != nullptr) {
        params.fresnel_dielectric_tint = *reinterpret_cast<const FresnelDielectricTint *>(
            microfacet->fresnel);
        params.microfacet.fresnel = &params.fresnel_dielectric_tint;
      }
      break;
    case MicrofacetFresnel::CONDUCTOR:
      if (microfacet->fresnel != nullptr) {
        params.fresnel_conductor = *reinterpret_cast<const FresnelConductor *>(microfacet->fresnel);
        params.microfacet.fresnel = &params.fresnel_conductor;
      }
      break;
    case MicrofacetFresnel::GENERALIZED_SCHLICK:
      if (microfacet->fresnel != nullptr) {
        params.fresnel_generalized_schlick =
            *reinterpret_cast<const FresnelGeneralizedSchlick *>(microfacet->fresnel);
        params.microfacet.fresnel = &params.fresnel_generalized_schlick;
      }
      break;
    case MicrofacetFresnel::F82_TINT:
      if (microfacet->fresnel != nullptr) {
        params.fresnel_f82_tint = *reinterpret_cast<const FresnelF82Tint *>(microfacet->fresnel);
        params.microfacet.fresnel = &params.fresnel_f82_tint;
      }
      break;
    case MicrofacetFresnel::NONE:
    case MicrofacetFresnel::DIELECTRIC:
      /* No additional data to copy. */
      break;
  }
}

Spectrum evaluate_specular_weight(KernelGlobals kg,
                                  const SpecularParameters &params,
                                  const float3 &dir_ds,
                                  const float3 &dir_sl,
                                  const float cos_theta_i_hint,
                                  const float cos_theta_t_hint)
{
  if (params.has_microfacet) {
    const MicrofacetBsdf &mf = params.microfacet;
    float3 oriented_normal = mf.N;
    if (cos_theta_i_hint < 0.0f) {
      oriented_normal = -oriented_normal;
    }
    else if (dot(oriented_normal, dir_ds) < 0.0f) {
      oriented_normal = -oriented_normal;
    }

    const float cos_NI = dot(oriented_normal, dir_ds);
    const float cos_NO = dot(oriented_normal, dir_sl);

    if (!(fabsf(cos_NI) > 1e-7f && fabsf(cos_NO) > 1e-7f)) {
      return zero_spectrum();
    }
    /* For reflection, both rays must be on same side of normal (same sign cosines).
     * For refraction, they must be on opposite sides (opposite sign cosines).
     * This matches Mitsuba reference behavior. */
    const bool same_side = (cos_NI * cos_NO > 0.0f);
    if (params.is_refraction) {
      if (same_side) {
        return zero_spectrum();  /* Refraction requires opposite sides */
      }
    }
    else {
      if (!same_side) {
        return zero_spectrum();  /* Reflection requires same side */
      }
    }

    Spectrum F_refl=zero_spectrum(), F_trans=zero_spectrum();
    microfacet_fresnel(kg, &mf, cos_NI, nullptr, &F_refl, &F_trans);
    const Spectrum fw = params.is_refraction ? F_trans : F_refl;
    return mf.weight * fw;
  }
  else {
    const bool has_cos_i = isfinite_safe(cos_theta_i_hint);
    const bool has_cos_t = isfinite_safe(cos_theta_t_hint);
    float3 normal = params.normal;
    if (!params.has_normal) {
      normal = params.has_microfacet ? params.microfacet.N : normal;
    }
    if (is_zero(normal)) {
      return zero_spectrum();
    }
    normal = normalize(normal);

    float3 oriented_normal = normal;
    if (has_cos_i && cos_theta_i_hint < 0.0f) {
      oriented_normal = -oriented_normal;
    }
    else if (dot(oriented_normal, dir_ds) > 0.0f) {
      oriented_normal = -oriented_normal;
    }

    const float cos_theta_i_signed = has_cos_i ? cos_theta_i_hint : -dot(oriented_normal, dir_ds);
    const float cos_theta_i = fabsf(cos_theta_i_signed);
    const float cos_theta_o = dot(oriented_normal, dir_sl);

    /* For reflection, both rays on same side (same sign product).
     * For refraction, opposite sides (negative product). */
    if (params.is_refraction) {
      if (!(cos_theta_i > 0.0f) || fabsf(cos_theta_o) < 1e-10f || cos_theta_i_signed * cos_theta_o >= 0.0f) {
        return zero_spectrum();
      }
    }
    else {
      if (!(cos_theta_i > 0.0f) || fabsf(cos_theta_o) < 1e-10f || cos_theta_i_signed * cos_theta_o <= 0.0f) {
        return zero_spectrum();
      }
    }

    if (params.has_conductor_fresnel) {
      if (!(cos_theta_i > 1e-7f)) {
        return zero_spectrum();
      }
      return fresnel_conductor(cos_theta_i, params.conductor_ior);
    }

    const float eta = fmaxf(params.base_eta, 1.0e-6f);
    if (!(eta > 0.0f)) {
      return zero_spectrum();
    }

    float cos_theta_t = has_cos_t ? fabsf(cos_theta_t_hint) : 0.0f;
    float relative_eta = eta;
    if (params.is_refraction) {
      const float dot_incident_normal = dot(normal, dir_ds);
      /* Consistent with compute_specular: grazing incidence treated as entering */
      const bool entering = (dot_incident_normal <= 1e-7f);
      relative_eta = entering ? (1.0f / eta) : eta;
      if (!has_cos_t) {
        const float sin2_theta_i = fmaxf(0.0f, 1.0f - cos_theta_i * cos_theta_i);
        const float sin2_theta_t = relative_eta * relative_eta * sin2_theta_i;
        if (sin2_theta_t >= 1.0f) {
          return zero_spectrum();
        }
        cos_theta_t = sqrtf(fmaxf(0.0f, 1.0f - sin2_theta_t));
      }
    }

    if (!(cos_theta_i > 1e-7f)) {
      return zero_spectrum();
    }

    if (!params.is_refraction && params.use_singular_reflection_weight) {
      return params.singular_reflection_weight;
    }

    float cos_theta_t_eval = cos_theta_t;
    const float F = params.is_refraction ?
                        fresnel_dielectric(cos_theta_i, relative_eta, &cos_theta_t_eval) :
                        fresnel_dielectric_cos(cos_theta_i, eta);
    Spectrum result = params.is_refraction ? make_spectrum(1.0f - F) : make_spectrum(F);
    if (params.is_refraction) {
      const float eta_scale = relative_eta * relative_eta;
      if (!(eta_scale > 0.0f)) {
        return zero_spectrum();
      }
      result *= eta_scale;
    }
    return result;
  }
}

float3 combine_vertex_normals(const SpecularSurfaceGeometry &geometry, const float u, const float v)
{
  const float w = 1.0f - u - v;
  float3 n = geometry.normals[0] * w + geometry.normals[1] * u + geometry.normals[2] * v;
  if (is_zero(n)) {
    return normalize(cross(geometry.dPdu, geometry.dPdv));
  }
  return normalize(n);
}

float3 compute_normal_derivative(const SpecularSurfaceGeometry &geometry,
                                 const float u,
                                 const float v,
                                 const float3 &normal,
                                 const bool du)
{
  const float w = 1.0f - u - v;
  const float3 raw = geometry.normals[0] * w + geometry.normals[1] * u + geometry.normals[2] * v;
  const float norm_raw = len(raw);
  if (norm_raw < 1e-10f) {
    return zero_float3();
  }
  const float3 d_raw = du ? (geometry.normals[1] - geometry.normals[0]) :
                           (geometry.normals[2] - geometry.normals[0]);
  const float3 projection = normal * dot(normal, d_raw);
  return (d_raw - projection) / norm_raw;
}

float3 reflect_dir(const float3 &dir_in, const float3 &normal)
{
  return normalize(dir_in - 2.0f * dot(dir_in, normal) * normal);
}

bool refract_dir(const float3 &dir_in,
                 const float3 &normal,
                 const float eta,
                 float3 &dir_out,
                 float &cos_theta_i,
                 float &cos_theta_t)
{
  cos_theta_i = -dot(dir_in, normal);
  const float eta_ratio = eta;
  const float sin2_theta_i = fmaxf(0.0f, 1.0f - cos_theta_i * cos_theta_i);
  const float sin2_theta_t = eta_ratio * eta_ratio * sin2_theta_i;
  if (sin2_theta_t > 1.0f) {
    return false;
  }
  cos_theta_t = sqrtf(fmaxf(0.0f, 1.0f - sin2_theta_t));
  dir_out = normalize(eta_ratio * dir_in + (eta_ratio * cos_theta_i - cos_theta_t) * normal);
  return true;
}

float3 refract_dir(const float3 &dir_in,
                   const float3 &normal,
                   const float eta,
                   bool &tir,
                   float &cos_theta_i,
                   float &cos_theta_t)
{
  float3 dir_out = zero_float3();
  if (!refract_dir(dir_in, normal, eta, dir_out, cos_theta_i, cos_theta_t)) {
    tir = true;
    return dir_out;
  }
  tir = false;
  return dir_out;
}

float3 compute_specular(const float3 &dir_ds,
                        const float3 &normal,
                        const SpecularParameters &params,
                        bool &tir,
                        float &cos_theta_i,
                        float &cos_theta_t,
                        float &eta_used)
{
  /* Per Codex finding: Use SD_BACKFACING flag for entering/exiting determination.
   * The dot(normal, dir_ds) test breaks when the normal has been flipped to align with
   * the BSDF frame, because both vectors can point in the same direction even when exiting.
   * SD_BACKFACING is set by shader_setup_from_ray based on dot(Ng, wi) < 0 using the
   * original geometric normal, so it's reliable regardless of BSDF frame orientation. */
  const bool exiting = params.backfacing;

  const float dot_w_n = dot(normal, dir_ds);

if (MPG_DEBUG::PARAMS()) {
  if (params.is_refraction) {
    printf("MPG DEBUG compute_specular: Mitsuba-style refraction\n");
    printf("  normal: (%.6f, %.6f, %.6f)\n", normal.x, normal.y, normal.z);
    printf("  dir_ds: (%.6f, %.6f, %.6f)\n", dir_ds.x, dir_ds.y, dir_ds.z);
    printf("  dot(normal, dir_ds): %.6f\n", dot_w_n);
    printf("  backfacing flag: %s\n", params.backfacing ? "TRUE" : "FALSE");
  }
}

if (MPG_DEBUG::PARAMS()) {
  if (params.is_refraction) {
    printf("  → Determined: %s (dot_w_n=%.6f)\n", exiting ? "EXITING" : "ENTERING", dot_w_n);
  }
}

  /* Orient normal to point toward the medium the ray came from.
   * Flip normal when exiting (coming from inside glass). */
  float3 oriented_normal = exiting ? -normal : normal;

  if (!params.is_refraction) {
    /* For reflection, use propagation direction (toward surface) */
    const float3 incoming_reflect = dir_ds;
    tir = false;
    cos_theta_i = -dot(incoming_reflect, oriented_normal);
    cos_theta_t = cos_theta_i;
    eta_used = 1.0f;
    return reflect_dir(incoming_reflect, oriented_normal);
  }

  /* Compute eta ratio for reverse ray tracing per Mitsuba's refract().
   * Eta = n_incident / n_transmitted for Snell's law.
   * Exiting (glass→air): eta = n_glass/n_air = IOR (e.g., 1.5)
   * Entering (air→glass): eta = n_air/n_glass = 1/IOR (e.g., 0.667) */
  const float safe_base_eta = fmaxf(params.base_eta, 1e-6f);
  const float eta_ratio = exiting ? safe_base_eta : (1.0f / safe_base_eta);
  const float safe_eta_ratio = fmaxf(eta_ratio, 1e-6f);

if (MPG_DEBUG::PARAMS()) {
  if (params.is_refraction) {
    printf("  → IOR: base=%.6f, eta_ratio=%s ? %.6f : %.6f = %.6f\n",
           safe_base_eta, exiting ? "base" : "1/base",
           safe_base_eta, 1.0f / safe_base_eta, safe_eta_ratio);
  }
}

  float3 dir = refract_dir(dir_ds, oriented_normal, safe_eta_ratio, tir, cos_theta_i, cos_theta_t);
  eta_used = safe_eta_ratio;
  if (tir) {
    cos_theta_t = 0.0f;
    return dir;
  }

  return dir;
}

float3 derivative_normalized(const float3 &vector, const float3 &d_vector)
{
  const float len_v = len(vector);
  if (len_v < 1e-8f) {
    return zero_float3();
  }
  const float3 v_hat = vector / len_v;
  return (d_vector * len_v - vector * (dot(vector, d_vector) / len_v)) / (len_v * len_v);
}

void compute_light_sample_direction(const LightSample &light_sample,
                                    const float3 &point,
                                    float3 &direction,
                                    float &distance)
{
  if (light_sample.t == FLT_MAX) {
    direction = is_zero(light_sample.D) ? make_float3(0.0f, 0.0f, 1.0f) : safe_normalize(light_sample.D);
    distance = MPG_DISTANT_LIGHT_VISIBILITY_DISTANCE;
    return;
  }

  float3 delta = light_sample.P - point;
  distance = len(delta);
  if (distance > 0.0f) {
    direction = delta / distance;
  }
  else {
    direction = is_zero(delta) ? make_float3(0.0f, 0.0f, 1.0f) : safe_normalize(delta);
    distance = 0.0f;
  }
}

float3 compute_light_sample_direction_derivative(const LightSample &light_sample,
                                                 const float3 &point,
                                                 const float3 &d_point)
{
  /* For directional/infinite lights (sun, background), direction is fixed
   * regardless of specular vertex position. Derivative is zero, matching
   * Mitsuba's fixed_direction flag behavior. */
  if (light_sample.t == FLT_MAX) {
    return zero_float3();
  }

  /* For finite lights (point, area), direction changes with vertex position.
   * Compute derivative of normalized direction vector. */
  return derivative_normalized(light_sample.P - point, -d_point);
}

void evaluate_specular(const ShadingPoint &D,
                       const MpgSeedRay &seed,
                       const SpecularSurfaceGeometry &geometry,
                       const SpecularParameters &params,
                       const float u,
                       const float v,
                       SpecularEval &eval)
{
  const float w = 1.0f - u - v;
  eval.point = geometry.verts[0] * w + geometry.verts[1] * u + geometry.verts[2] * v;
  eval.dXdu = geometry.dPdu;
  eval.dXdv = geometry.dPdv;
  /* Direction FROM reference TO specular point (current_vertex - previous_vertex).
   * For path x_prev→x_cur→x_next: dir_ds = x_cur - x_prev (forward along path).
   * Used with wi = -dir_ds to get Mitsuba convention: wi = x_prev - x_cur (backward).
   * For refraction: dir_ds represents light arrival direction at specular vertex. */
  eval.dir_ds = eval.point - D.position;
  eval.distance_ds = len(eval.dir_ds);
  eval.dir_ds = (eval.distance_ds > 0.0f) ? (eval.dir_ds / eval.distance_ds) :
                                           make_float3(0.0f, 0.0f, 1.0f);

  compute_light_sample_direction(seed.light_sample, eval.point, eval.dir_sl, eval.distance_sl);
  eval.light_sample = seed.light_sample;  /* Store for analytical Jacobian computation */

  /* Use normals following Mitsuba's ManifoldVertex construction approach.
   * Per Codex finding: Mitsuba builds each vertex from bsdf()->frame and only flips
   * the geometric normal to match the shading frame. We prioritize BSDF frame normal
   * when available (from params.normal), falling back to interpolated/face normals.
   *
   * For smooth shading: interpolate vertex normals
   * For flat shading: use the face normal (already computed and stored in geometry.normals)
   * For BSDF frame: use params.normal if available (highest priority) */

  if (params.has_normal) {
    /* Use BSDF frame normal - highest priority per Mitsuba's approach.
     * This comes from microfacet->N or shading_normal from specular_parameters_from_surface. */
    eval.normal = params.normal;
if (MPG_DEBUG::PARAMS()) {
    printf("MPG DEBUG evaluate_specular: Using BSDF frame normal from params\n");
    printf("  params.normal: (%.6f, %.6f, %.6f)\n",
           params.normal.x, params.normal.y, params.normal.z);
}
    /* Per Codex: Propagate frame derivatives when BSDF frame varies across surface.
     * - For microfacet BSDF frames (constant): derivatives are zero
     * - For BSDF frames from smooth shading (varying): compute derivatives from geometry
     * Mitsuba's ManifoldVertex propagates frame derivatives; we need to match this. */
    if (params.has_microfacet && !is_zero(params.microfacet.N)) {
      /* Microfacet BSDF frame is constant across surface - zero derivatives correct */
      eval.dNdu = zero_float3();
      eval.dNdv = zero_float3();
    }
    else if (seed.use_smooth_normals) {
      /* BSDF frame from smooth-shaded geometry - compute derivatives to match Mitsuba */
      eval.dNdu = compute_normal_derivative(geometry, u, v, eval.normal, true);
      eval.dNdv = compute_normal_derivative(geometry, u, v, eval.normal, false);
    }
    else {
      /* Flat shading - normal is constant across face */
      eval.dNdu = zero_float3();
      eval.dNdv = zero_float3();
    }
  }
  else if (seed.use_smooth_normals) {
    /* Fallback to interpolated vertex normals */
    eval.normal = combine_vertex_normals(geometry, u, v);
    if (is_zero(eval.normal)) {
      /* Fallback to face normal if interpolation gives zero */
      eval.normal = geometry.normals[0];
    }
    eval.dNdu = compute_normal_derivative(geometry, u, v, eval.normal, true);
    eval.dNdv = compute_normal_derivative(geometry, u, v, eval.normal, false);
  }
  else {
    /* Use precomputed face normal from load_surface_geometry */
    eval.normal = geometry.normals[0];
if (MPG_DEBUG::PARAMS()) {
    printf("MPG DEBUG evaluate_specular: Using flat shading normal\n");
    printf("  geometry.normals[0]: (%.6f, %.6f, %.6f)\n",
           geometry.normals[0].x, geometry.normals[0].y, geometry.normals[0].z);
    printf("  eval.normal set to: (%.6f, %.6f, %.6f)\n",
           eval.normal.x, eval.normal.y, eval.normal.z);
}
    eval.dNdu = zero_float3();
    eval.dNdv = zero_float3();
  }

  if (!is_zero(eval.normal)) {
    eval.normal = safe_normalize(eval.normal);
    /* Use geometric normals as-is from mesh. No flipping based on BSDF or ray direction.
     * The entering/exiting determination in compute_specular() uses dot(normal, ray) which
     * works correctly with the actual geometric normals from the mesh. */

    /* Per Codex: Construct BSDF frame tangents orthogonal to BSDF frame normal.
     * Mitsuba uses frame.s and frame.t (orthogonal to frame.n), not geometric dPdu/dPdv.
     * This ensures the half-vector constraint projection matches Mitsuba when the BSDF
     * frame normal differs from the geometric normal. */
    make_orthonormals(eval.normal, &eval.tangent_u, &eval.tangent_v);
  }

  /* Per Codex: Mitsuba's refract(w, n, eta) expects w to be the direction AWAY from
   * the surface (wi = x_prev - x_cur), but dir_ds is x_cur - x_prev (toward surface).
   * Pass -dir_ds to match Mitsuba's convention for entering/exiting determination. */
  float cos_theta_i = 0.0f, cos_theta_t = 0.0f, eta = 1.0f;
  const float3 spec_dir = compute_specular(
      -eval.dir_ds, eval.normal, params, eval.tir, cos_theta_i, cos_theta_t, eta);
  eval.refractive = params.is_refraction;
  eval.eta = eta;
  eval.cos_theta_i = cos_theta_i;
  eval.cos_theta_t = cos_theta_t;

  /* Mitsuba's half-vector constraint formulation (proven to converge to 1e-4).
   * Instead of directional constraint (dir_sl - spec_dir), use generalized half-vector
   * projected onto surface tangent frame. This is the constraint used in Mitsuba reference.
   *
   * Note: Half-vector uses BSDF convention (directions away from surface), while refraction
   * computation uses propagation direction (toward surface). These are intentionally different. */
  const float3 wi = -eval.dir_ds;  /* Incident direction away from surface (X→C) */
  const float3 wo = eval.dir_sl;    /* Outgoing direction away from surface (X→L) */

  /* Half-vector eta determination per Mitsuba reference.
   * CRITICAL: Must test dot(wi, geometric_normal) dynamically, not use precomputed flag.
   * CRITICAL: Align geometric normal to BSDF frame before testing, per Mitsuba.
   *
   * Mitsuba's logic (manifold_path_guiding.h):
   *   Float eta = v[i].eta;  // base IOR
   *   // Align gn to BSDF frame: masked(gn, dot(n, gn) < 0) *= -1
   *   if (dot(wi, v[i].gn) < 0.f) {  // Test against ALIGNED geometric normal
   *       eta = rcp(eta);
   *   }
   *
   * The half-vector eta convention:
   * - Entering (dot(wi, aligned_gn) >= 0): eta = base_eta (e.g., 1.5 for glass)
   * - Exiting (dot(wi, aligned_gn) < 0): eta = 1/base_eta (e.g., 0.667 for glass) */
  float h_eta = 1.0f;
  if (eval.refractive) {
    /* Compute raw geometric normal from surface derivatives */
    const float3 geometric_normal_raw = safe_normalize(cross(geometry.dPdu, geometry.dPdv));

    /* Align geometric normal to BSDF frame (Mitsuba: masked(gn, dot(n, gn) < 0) *= -1) */
    float3 geometric_normal = geometric_normal_raw;
    if (dot(eval.normal, geometric_normal_raw) < 0.0f) {
      geometric_normal = -geometric_normal_raw;
    }

    const float dot_wi_gn = dot(wi, geometric_normal);
    const float base_eta = params.base_eta;

    if (dot_wi_gn < 0.0f) {
      /* Exiting (coming from inside): eta = 1/base_eta */
      h_eta = 1.0f / fmaxf(base_eta, 1e-6f);
    }
    else {
      /* Entering (arriving from outside): eta = base_eta */
      h_eta = base_eta;
    }
  }

  /* Generalized half-vector: h = normalize(wi + eta * wo), negated for refraction */
  float3 h = wi + h_eta * wo;
  if (eval.refractive) {
    h = -h;
  }
  const float h_len = len(h);
  if (h_len > 1e-8f) {
    h /= h_len;
  }

  /* Mitsuba half-vector constraint: C = [dot(h, tangent_u), dot(h, tangent_v)]
   * The residual is the half-vector itself, which gets projected onto surface tangents
   * in evaluate_double_bounce() to form the 2D constraint per vertex.
   * This matches the Jacobian formulation from Mitsuba's MPG implementation. */
  eval.residual = h;

if (MPG_DEBUG::NEWTON_DETAIL()) {
  const float dot_h_n = dot(h, eval.normal);
  printf("    Half-vector: h=(%.4f,%.4f,%.4f), dot(h,n)=%.4f, refract=%d, eta=%.4f\n",
         h.x, h.y, h.z, dot_h_n, eval.refractive, h_eta);
  printf("      wi=(%.4f,%.4f,%.4f), wo=(%.4f,%.4f,%.4f), n=(%.4f,%.4f,%.4f)\n",
         wi.x, wi.y, wi.z, wo.x, wo.y, wo.z, eval.normal.x, eval.normal.y, eval.normal.z);
}
}

static bool smooth_normals_at_hit(KernelGlobals kg,
                                  const Ray &ray,
                                  const Intersection &isect,
                                  bool &has_smooth_normals)
{
  ShaderData hit_sd = {};
  shader_setup_from_ray(kg, &hit_sd, &ray, const_cast<Intersection *>(&isect));

  const ConstIntegratorState integrator_state = nullptr;
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
      kg, integrator_state, &hit_sd, nullptr, PATH_RAY_CAMERA, true);

  has_smooth_normals = (hit_sd.shader & SHADER_SMOOTH_NORMAL) != 0;
  return true;
}

bool specular_parameters_from_surface(KernelGlobals kg,
                                      const ShaderData &sd,
                                      const SpecularSurfaceGeometry &geometry,
                                      const MpgSeedRay &seed,
                                      const int bounce_index,
                                      const float u,
                                      const float v,
                                      SpecularParameters &params)
{
  params = SpecularParameters();

  const float w = 1.0f - u - v;
  const float3 spec_point = geometry.verts[0] * w + geometry.verts[1] * u + geometry.verts[2] * v;
  const float3 geometric_normal = safe_normalize(cross(geometry.dPdu, geometry.dPdv));
  const bool has_geometric_normal = !is_zero(geometric_normal);
  float3 ray_dir = spec_point - sd.P;
  const float distance = len(ray_dir);
  /* MPG reference does NOT check minimum distance here. Removed check to match.
   * For thin geometry (solidified planes), vertices can be very close together.
   * Only prevent exact zero for safe normalization. */
  if (!(distance > 1e-12f)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG DEBUG specular_parameters: distance too small (%.12f)\n", distance);
}
    return false;
  }
  ray_dir /= distance;

  Ray ray;
  float3 n_off = (dot(sd.Ng, ray_dir) >= 0.0f) ? sd.Ng : -sd.Ng;
  ray.P = ray_offset(sd.P, n_off);
  ray.D = ray_dir;
  ray.tmin = 1.0e-4f;
  ray.tmax = distance;
  ray.time = sd.time;
  ray.self.prim = seed.prim;
  ray.self.object = seed.object;
  ray.self.light_prim = PRIM_NONE;
  ray.self.light_object = OBJECT_NONE;

  Intersection isect;
  isect.t = distance;
  isect.u = u;
  isect.v = v;
  isect.prim = seed.prim;
  isect.object = seed.object;
  const int object_flag = kernel_data_fetch(object_flag, seed.object);
  isect.type = (object_flag & SD_OBJECT_MOTION) ? PRIMITIVE_MOTION_TRIANGLE : PRIMITIVE_TRIANGLE;

  ShaderData spec_sd = {};
  shader_setup_from_ray(kg, &spec_sd, &ray, &isect);

  const ConstIntegratorState integrator_state = nullptr;
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
      kg, integrator_state, &spec_sd, nullptr, PATH_RAY_CAMERA, true);

  /* CRITICAL: Do NOT use spec_sd.N directly as it may be faceforward-ed!
   * Per Codex finding: Mitsuba builds ManifoldVertex from bsdf()->frame (not faceforward).
   * We'll use microfacet->N (BSDF frame) when available, or raw geometric_normal.
   * Never use spec_sd.N which is flipped to face the ray - this breaks entering/exiting. */
  const float3 raw_shading_normal = safe_normalize(spec_sd.N);
  const bool has_raw_shading_normal = !is_zero(raw_shading_normal);

if (MPG_DEBUG::PARAMS()) {
  printf("MPG DEBUG specular_parameters_from_surface: Normal sources\n");
  printf("  geometric_normal (from cross product): (%.6f, %.6f, %.6f)\n",
         geometric_normal.x, geometric_normal.y, geometric_normal.z);
  printf("  spec_sd.N (potentially faceforward-ed): (%.6f, %.6f, %.6f)\n",
         raw_shading_normal.x, raw_shading_normal.y, raw_shading_normal.z);
  printf("  Will use microfacet->N if available, else geometric_normal\n");
}

  const MicrofacetBsdf *reflection_microfacet = nullptr;
  const MicrofacetBsdf *refraction_microfacet = nullptr;
  const bool shader_reports_transmission = (spec_sd.flag & SD_BSDF_HAS_TRANSMISSION) != 0;

if (MPG_DEBUG::PARAMS()) {
  /* Closure details will be shown at function end */
}

  enum class SeedHemisphere {
    Unknown,
    Reflection,
    Transmission,
  };

  /* Use geometric_normal for hemisphere determination, not faceforward-ed spec_sd.N */
  float3 hemisphere_normal = geometric_normal;
  bool hemisphere_normal_valid = has_geometric_normal;
  if (!hemisphere_normal_valid && has_raw_shading_normal) {
    hemisphere_normal = raw_shading_normal;
    hemisphere_normal_valid = true;
  }
  if (hemisphere_normal_valid) {
    hemisphere_normal = safe_normalize(hemisphere_normal);
    hemisphere_normal_valid = !is_zero(hemisphere_normal);
  }

  float3 seed_direction_hint = ray_dir;
  if (!is_zero(seed.direction)) {
    const float3 normalized_seed = safe_normalize(seed.direction);
    if (!is_zero(normalized_seed)) {
      seed_direction_hint = normalized_seed;
    }
  }

  SeedHemisphere seed_hemisphere = SeedHemisphere::Unknown;
  if (hemisphere_normal_valid && !is_zero(seed_direction_hint)) {
    const float dot_seed = dot(hemisphere_normal, seed_direction_hint);
    if (fabsf(dot_seed) > 1.0e-6f) {
      seed_hemisphere = (dot_seed > 0.0f) ? SeedHemisphere::Transmission : SeedHemisphere::Reflection;
    }
  }

  const bool seed_prefers_transmission = (seed_hemisphere == SeedHemisphere::Transmission);
  const bool seed_prefers_reflection = (seed_hemisphere == SeedHemisphere::Reflection);

  float3 light_dir = zero_float3();
  float light_distance = 0.0f;
  compute_light_sample_direction(seed.light_sample, spec_point, light_dir, light_distance);

  const bool light_dir_degenerate = (seed.light_sample.t == FLT_MAX) ? is_zero(seed.light_sample.D) :
                                                                            !(light_distance > 0.0f);

  const bool tau_hint_valid = (seed.tau_count > bounce_index);
  bool force_transmission = false;
  bool force_reflection = false;
if (MPG_DEBUG::PARAMS()) {
  printf("MPG DEBUG specular_parameters: Selection criteria\n");
  printf("  bounce_index: %d\n", bounce_index);
  printf("  seed.scatter: %d (0=none, 1=refraction, 2=reflection)\n", (int)seed.scatter);
  printf("  tau_hint_valid: %s\n", tau_hint_valid ? "TRUE" : "FALSE");
  if (tau_hint_valid) {
    printf("  tau_bits: 0x%x, tau_count: %d\n", seed.tau_bits, seed.tau_count);
    printf("  tau bit at position %d: %d (1=transmission, 0=reflection)\n", bounce_index, (int)get_chaintype_bit(seed.tau_bits, bounce_index));
  }
}
  if (tau_hint_valid) {
    force_transmission = get_chaintype_bit(seed.tau_bits, bounce_index);
    force_reflection = !force_transmission;
if (MPG_DEBUG::PARAMS()) {
    printf("  -> TAU HINT: %s\n", force_transmission ? "FORCE TRANSMISSION" : "FORCE REFLECTION");
}
  }
  else if (seed.scatter == MPG_SEED_SCATTER_REFRACTION) {
    force_transmission = true;
if (MPG_DEBUG::PARAMS()) {
    printf("  -> SEED SCATTER: FORCE TRANSMISSION\n");
}
  }
  else if (seed.scatter == MPG_SEED_SCATTER_REFLECTION) {
    force_reflection = true;
if (MPG_DEBUG::PARAMS()) {
    printf("  -> SEED SCATTER: FORCE REFLECTION\n");
}
  }

  bool prefer_reflection_from_geometry = false;
  /* For directional lights, light_dir points TO the light source (opposite of actual light ray
   * travel direction), making the "same side" test invalid. Only apply geometric checks for
   * finite lights where light_dir represents the actual light ray direction. */
  const bool is_directional_light = (seed.light_sample.t == FLT_MAX);

  /* CRITICAL: Tau hints must NEVER be overridden. They encode the successful path found during
   * seed generation. The hemisphere test should only influence preference when NO tau hint exists. */
  const bool have_tau_hint = (force_transmission || force_reflection);

  if (!light_dir_degenerate && hemisphere_normal_valid && !is_directional_light && !have_tau_hint) {
    /* Compare directions in the shading-normal frame (hemisphere_normal) to detect interface crossings. */
    const float dot_in = dot(hemisphere_normal, -ray_dir);
    const float dot_light = dot(hemisphere_normal, light_dir);
if (MPG_DEBUG::PARAMS()) {
    printf("MPG DEBUG specular_parameters: hemisphere test\n");
    printf("  hemisphere_normal: (%.6f, %.6f, %.6f)\n",
           hemisphere_normal.x, hemisphere_normal.y, hemisphere_normal.z);
    printf("  ray_dir: (%.6f, %.6f, %.6f)\n", ray_dir.x, ray_dir.y, ray_dir.z);
    printf("  light_dir: (%.6f, %.6f, %.6f)\n", light_dir.x, light_dir.y, light_dir.z);
    printf("  dot_in = dot(normal, -ray_dir): %.6f\n", dot_in);
    printf("  dot_light = dot(normal, light_dir): %.6f\n", dot_light);
}
    if ((dot_in < 0.0f && dot_light > 0.0f) || (dot_in > 0.0f && dot_light < 0.0f)) {
if (MPG_DEBUG::PARAMS()) {
      printf("  -> FORCING TRANSMISSION (opposite hemispheres)\n");
}
      force_transmission = true;
      force_reflection = false;
    }
    else if ((dot_in > 0.0f && dot_light > 0.0f) || (dot_in < 0.0f && dot_light < 0.0f)) {
      /* When both rays occupy the same half-space the Mitsuba reference prefers
       * the reflective branch even if the seed requested refraction. */
if (MPG_DEBUG::PARAMS()) {
      printf("  -> PREFERRING REFLECTION (same hemisphere)\n");
}
      prefer_reflection_from_geometry = true;
    }
  }
else if (MPG_DEBUG::PARAMS() && have_tau_hint) {
    printf("  -> SKIPPING hemisphere test (tau hint active)\n");
  }

  const bool has_forced_scatter = force_transmission || force_reflection;

  bool prefer_transmission = force_transmission;
  bool prefer_reflection = force_reflection;
  if (!prefer_transmission && !prefer_reflection) {
    prefer_transmission = seed_prefers_transmission;
    prefer_reflection = seed_prefers_reflection;
  }

  bool prefer_reflection_from_seed = false;
  if (seed.scatter == MPG_SEED_SCATTER_REFRACTION || seed.scatter == MPG_SEED_SCATTER_REFLECTION) {
    const bool selected_refraction = (seed.scatter == MPG_SEED_SCATTER_REFRACTION);
    if (isfinite_safe(seed.seed_scatter_pdf)) {
      const float scatter_pdf = clamp(seed.seed_scatter_pdf, 0.0f, 1.0f);
      const bool has_dual_branches = (scatter_pdf > 1.0e-6f) && (scatter_pdf < 1.0f - 1.0e-6f);
      if (selected_refraction && has_dual_branches) {
        const float reflection_probability = 1.0f - scatter_pdf;
        const float transmission_probability = scatter_pdf;
        if (reflection_probability > transmission_probability + 1.0e-4f) {
          prefer_reflection_from_seed = true;
        }
      }
      else if (!selected_refraction && !has_dual_branches) {
        prefer_reflection_from_seed = true;
      }
    }
  }

  /* Mirror Mitsuba's branch selection: geometric same-side tests and the
   * Fresnel/guide probabilities can override a refraction request so we still
   * explore the reflective branch when it is the only valid transport mode. */
  if (prefer_reflection_from_geometry || prefer_reflection_from_seed) {
    prefer_reflection = true;
    if (!force_transmission && !force_reflection) {
      prefer_transmission = false;
    }
  }

  auto log_forced_scatter_mismatch = [&](const char *requested, const char *fallback) {
    if (MPG_DEBUG::PARAMS()) {
      if (!has_forced_scatter) {
        return;
      }
      if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
        LOG_DEBUG << "MPG seed forced " << requested << " scatter but only " << fallback
                  << " available; falling back.";
      }
    }
    else {
      (void)requested;
      (void)fallback;
    }
  };

  bool have_singular_reflection = false;
  float eta_singular = 1.5f; // overwritten below with microfacet IOR when available
  SpecularParameters singular_refraction_params;
  bool have_singular_refraction_params = false;

  /* Track if we found any BSDF closures that were non-specular (incompatible with MPG).
   * Type 3 (CLOSURE_BSDF_BURLEY_ID) is diffuse, which cannot be handled by manifold walk. */
  bool found_non_specular_bsdf = false;
  int non_specular_closure_type = -1;

  for (int i = 0; i < spec_sd.num_closure; ++i) {
    const ShaderClosure *closure = &spec_sd.closure[i];

    if (!CLOSURE_IS_BSDF(closure->type)) {
      continue;
    }

    const bool is_trans = CLOSURE_IS_BSDF_TRANSMISSION(closure->type);
    const bool is_glass = CLOSURE_IS_GLASS(closure->type);
    const bool is_micro = CLOSURE_IS_BSDF_MICROFACET(closure->type);
    const bool is_singular = CLOSURE_IS_BSDF_SINGULAR(closure->type);

if (MPG_DEBUG::PARAMS()) {
    printf("MPG DEBUG specular_parameters: Closure %d\n", i);
    printf("  Type: %d, is_bsdf=1, is_glass=%d, is_micro=%d, is_singular=%d, is_trans=%d\n",
           (int)closure->type, is_glass, is_micro, is_singular, is_trans);
}
    /* Glass BSDFs are specular and compatible with MPG.
     * CLOSURE_IS_BSDF_SINGULAR only includes transparent/portal, not glass. */
    if (!(is_micro || is_singular || is_glass)) {
if (MPG_DEBUG::PARAMS()) {
      printf("  -> REJECTED: Not specular (not micro, singular, or glass)\n");
}
      found_non_specular_bsdf = true;
      non_specular_closure_type = (int)closure->type;
      continue;
    }

    if (is_micro) {
      const MicrofacetBsdf *mf = reinterpret_cast<const MicrofacetBsdf *>(closure);
      const float ax = fmaxf(mf->alpha_x, 0.0f);
      const float ay = fmaxf(mf->alpha_y, 0.0f);
      const bool delta_like = (ax <= 1.0e-6f) && (ay <= 1.0e-6f);
if (MPG_DEBUG::PARAMS()) {
      printf("  Microfacet: alpha_x=%.9f, alpha_y=%.9f, ior=%.6f, delta_like=%d\n",
             ax, ay, mf->ior, delta_like);
}
      if (delta_like) {
        if (!(is_trans || is_glass)) {
          have_singular_reflection = true;
        }
        if (mf->ior > 0.0f) {
          eta_singular = mf->ior;
        }
      }
      const bool reports_reflection = !is_trans || is_glass;
      bool reports_transmission = is_trans || is_glass;
      if (!reports_transmission && shader_reports_transmission) {
        reports_transmission = fabsf(mf->ior) > 1.0f + 1.0e-6f;
      }

if (MPG_DEBUG::PARAMS()) {
      printf("  reports_reflection=%d, reports_transmission=%d\n",
             reports_reflection, reports_transmission);
}
      if (reports_transmission) {
        refraction_microfacet = mf;
if (MPG_DEBUG::PARAMS()) {
        printf("  -> Set refraction_microfacet\n");
}
      }
      if (reports_reflection) {
        reflection_microfacet = mf;
if (MPG_DEBUG::PARAMS()) {
        printf("  -> Set reflection_microfacet\n");
}
      }
    }
    if (is_singular) {
      if (is_trans) {
        SpecularParameters refraction_params;
        refraction_params.has_microfacet = false;
        refraction_params.is_refraction = true;

        float eta = 1.45f;
        if (refraction_microfacet) {
          eta = fmaxf(1.0e-6f, refraction_microfacet->ior);
        }
        refraction_params.base_eta = fabsf(eta);
        /* Per Codex: use BSDF frame (microfacet->N) or geometric_normal, NOT faceforward-ed shading */
        if (refraction_microfacet && !is_zero(refraction_microfacet->N)) {
          refraction_params.normal = safe_normalize(refraction_microfacet->N);
          refraction_params.has_normal = true;
        }
        else if (has_geometric_normal) {
          refraction_params.normal = geometric_normal;
          refraction_params.has_normal = true;
        }
        singular_refraction_params = refraction_params;
        have_singular_refraction_params = true;
        continue;
      }
      have_singular_reflection = true;
      continue;
    }
  }
  auto fill_singular_reflection = [&]() {
    params = SpecularParameters();
    params.has_microfacet = false;
    params.is_refraction = false;
    params.base_eta = fabsf(eta_singular);
    if (reflection_microfacet != nullptr) {
      params.normal = safe_normalize(reflection_microfacet->N);
      params.has_normal = !is_zero(params.normal);
      const MicrofacetFresnel fresnel_type =
          static_cast<MicrofacetFresnel>(reflection_microfacet->fresnel_type);
      if (fresnel_type == MicrofacetFresnel::NONE) {
        params.singular_reflection_weight = reflection_microfacet->weight;
        params.use_singular_reflection_weight = true;
      }
      if (reflection_microfacet->fresnel != nullptr) {
        if (fresnel_type == MicrofacetFresnel::CONDUCTOR) {
          params.has_conductor_fresnel = true;
          const FresnelConductor *fresnel = reinterpret_cast<const FresnelConductor *>(
              reflection_microfacet->fresnel);
          params.conductor_ior = fresnel->ior;
        }
        else if (fresnel_type == MicrofacetFresnel::DIELECTRIC ||
                 fresnel_type == MicrofacetFresnel::DIELECTRIC_TINT)
        {
          if (reflection_microfacet->ior > 0.0f) {
            params.base_eta = fabsf(reflection_microfacet->ior);
          }
        }
      }
    }
    /* Per Codex: use geometric_normal, NOT faceforward-ed shading_normal */
    if (!params.has_normal && has_geometric_normal) {
      params.normal = geometric_normal;
      params.has_normal = true;
    }
  };

  if (have_singular_reflection || have_singular_refraction_params) {
    if (prefer_transmission) {
      if (!have_singular_refraction_params) {
        log_forced_scatter_mismatch("refraction", "singular reflection");
      }
      else {
        params = singular_refraction_params;
        return true;
      }
    }
    if (prefer_reflection) {
      if (!have_singular_reflection) {
        log_forced_scatter_mismatch("reflection", "singular refraction");
      }
      else {
        fill_singular_reflection();
        return true;
      }
    }
    if (!have_singular_reflection) {
      params = singular_refraction_params;
      return true;
    }
    if (!have_singular_refraction_params) {
      fill_singular_reflection();
      return true;
    }
    /* Ambiguous preference: mirror previous behavior and fall back to transmission. */
    params = singular_refraction_params;
    return true;
  }

  const MicrofacetBsdf *microfacet = nullptr;
  bool selected_refraction = false;
  if (prefer_transmission && refraction_microfacet != nullptr) {
    microfacet = refraction_microfacet;
    selected_refraction = true;
  }
  else if (prefer_transmission && has_forced_scatter) {
    const char *available = (reflection_microfacet != nullptr) ? "microfacet reflection"
                                                              : "no microfacet lobe";
    log_forced_scatter_mismatch("refraction", available);
  }

  if (microfacet == nullptr && prefer_reflection && reflection_microfacet != nullptr) {
    microfacet = reflection_microfacet;
    selected_refraction = false;
  }
  else if (microfacet == nullptr && prefer_reflection && has_forced_scatter) {
    const char *available = (refraction_microfacet != nullptr) ? "microfacet refraction"
                                                              : "no microfacet lobe";
    log_forced_scatter_mismatch("reflection", available);
  }

  if (microfacet == nullptr && refraction_microfacet != nullptr && reflection_microfacet == nullptr) {
    microfacet = refraction_microfacet;
    selected_refraction = true;
  }
  else if (microfacet == nullptr && reflection_microfacet != nullptr &&
           refraction_microfacet == nullptr)
  {
    microfacet = reflection_microfacet;
    selected_refraction = false;
  }
  else if (microfacet == nullptr && refraction_microfacet != nullptr && reflection_microfacet != nullptr) {
    /* If both microfacets point to the same closure, it's a glass BSDF that supports both
     * reflection and transmission via Fresnel. Glass should ALWAYS use refraction mode with
     * proper IOR in the half-vector constraint, regardless of which Fresnel lobe we sample. */
    const bool is_glass_closure = (refraction_microfacet == reflection_microfacet);

    if (is_glass_closure) {
      /* Glass: Always use refraction mode with IOR. The prefer_reflection flag only affects
       * which Fresnel lobe is sampled (reflected vs transmitted ray), not the constraint mode. */
      microfacet = refraction_microfacet;
      selected_refraction = true;
if (MPG_DEBUG::PARAMS()) {
      printf("  -> Glass closure detected: forcing refraction mode with IOR=%.6f\n", refraction_microfacet->ior);
}
    }
    else if (prefer_reflection) {
      microfacet = reflection_microfacet;
      selected_refraction = false;
    }
    else {
      microfacet = refraction_microfacet;
      selected_refraction = true;
    }
  }
  if (microfacet == nullptr) {
if (MPG_DEBUG::PARAMS()) {
    printf("----------------------------------------\n");
    printf("ERROR: No suitable BSDF for MPG!\n");
    printf("  Total closures evaluated: %d\n", spec_sd.num_closure);
    printf("  Has reflection: %s\n", reflection_microfacet ? "YES" : "NO");
    printf("  Has refraction: %s\n", refraction_microfacet ? "YES" : "NO");
    if (found_non_specular_bsdf) {
      printf("  Found incompatible BSDF: type=%d (diffuse/glossy)\n", non_specular_closure_type);
      printf("  ** MPG requires glass/mirror surfaces only **\n");
    }
    printf("========================================\n\n");
}
    return false;
  }
  params = SpecularParameters();
  copy_microfacet_to_parameters(microfacet, params);

  params.is_refraction = selected_refraction;
if (MPG_DEBUG::PARAMS()) {
  printf("  -> Selected mode: %s\n", selected_refraction ? "REFRACTION" : "REFLECTION");
  printf("  -> Microfacet IOR from closure: %.6f\n", microfacet->ior);
  printf("  -> params.base_eta (after copy): %.6f\n", params.base_eta);
}
  if (params.is_refraction) {
    const float eta = microfacet->ior;
    if (fabsf(eta) <= 1e-6f) {
      if (reflection_microfacet != nullptr) {
        /* Refraction lobe turned out invalid (e.g., eta ~ 0). Try the reflective
         * branch instead to mirror Mitsuba's retry logic. */
        microfacet = reflection_microfacet;
        selected_refraction = false;
        prefer_reflection = true;
        prefer_transmission = false;
        params = SpecularParameters();
        copy_microfacet_to_parameters(microfacet, params);
        params.is_refraction = false;
        params.base_eta = 1.0f;
      }
      else {
if (MPG_DEBUG::PARAMS()) {
        printf("MPG DEBUG specular_parameters: refraction eta invalid (%.6f), no reflection fallback\n", eta);
}
        return false;
      }
    }
    else {
      /* Cycles may return the relative IOR in either direction depending on ray orientation.
       * For MPG, we always need eta > 1.0 representing the absolute IOR of the refractive medium.
       * If eta < 1.0, it's the reciprocal (e.g., 0.667 = 1/1.5), so invert it. */
      float corrected_eta = fabsf(eta);
      if (corrected_eta < 1.0f - 1e-6f) {
        corrected_eta = 1.0f / corrected_eta;
      }
      params.base_eta = corrected_eta;
    }
  }
  if (!params.is_refraction) {
    params.base_eta = 1.0f;
  }
  params.microfacet.N = normalize(params.microfacet.N);

  /* Per Codex: Align geometric_normal to BSDF frame, matching Mitsuba's approach.
   * Mitsuba does: masked(gn, dot(n, gn) < 0) *= -1
   * This ensures geometric normal points in same hemisphere as BSDF frame.
   * Unlike faceforward which aligns to viewing direction, this aligns to BSDF frame. */
  float3 aligned_geometric_normal = geometric_normal;
  if (has_geometric_normal && !is_zero(params.microfacet.N)) {
    if (dot(params.microfacet.N, geometric_normal) < 0.0f) {
      aligned_geometric_normal = -geometric_normal;
if (MPG_DEBUG::PARAMS()) {
      printf("  -> Aligned geometric_normal to BSDF frame: flipped from (%.6f, %.6f, %.6f) to (%.6f, %.6f, %.6f)\n",
             geometric_normal.x, geometric_normal.y, geometric_normal.z,
             aligned_geometric_normal.x, aligned_geometric_normal.y, aligned_geometric_normal.z);
}
    }
  }

  /* Per Codex: use microfacet->N (BSDF frame) or aligned_geometric_normal, NOT faceforward-ed shading */
  if (!is_zero(params.microfacet.N)) {
    params.normal = params.microfacet.N;
    params.has_normal = true;
  }
  else if (has_geometric_normal) {
    params.normal = aligned_geometric_normal;
    params.has_normal = true;
  }

  /* Per Codex: Do NOT flip microfacet.N based on incident/outgoing directions.
   * Mitsuba only flips geometric normal to match BSDF frame (done above).
   * The extra bad_refraction/bad_reflection flip diverges from Mitsuba and causes
   * conflicting entering/exiting classification when the BSDF frame becomes
   * inverted relative to the geometric normal. REMOVED. */

  /* Fallback to aligned_geometric_normal if still no normal (should not happen with microfacet) */
  if (!params.has_normal && has_geometric_normal) {
    params.normal = aligned_geometric_normal;
    params.has_normal = true;
  }
  /* Store backfacing flag for robust entering/exiting determination.
   * SD_BACKFACING is set by shader_setup_from_ray based on dot(Ng, wi) < 0. */
  params.backfacing = (spec_sd.flag & SD_BACKFACING) != 0;
if (MPG_DEBUG::PARAMS()) {
  printf("  -> SD_BACKFACING flag: %s\n", params.backfacing ? "TRUE (exiting)" : "FALSE (entering)");
  printf("  -> Final params.base_eta: %.6f\n", params.base_eta);
}
  return true;
}

bool load_surface_geometry(KernelGlobals kg,
                           const ShaderData &sd,
                           const MpgSeedRay &seed,
                           SpecularSurfaceGeometry &geometry)
{
  if (seed.object == OBJECT_NONE || seed.prim < 0) {
    return false;
  }

  const int object = seed.object;
  const int prim = seed.prim;
  const int object_flag = kernel_data_fetch(object_flag, object);

  if (object_flag & SD_OBJECT_MOTION) {
    motion_triangle_vertices_and_normals(kg, object, prim, sd.time, geometry.verts, geometry.normals);
  }
  else {
    triangle_vertices_and_normals(kg, prim, geometry.verts, geometry.normals);
  }

  ShaderData object_sd = {};
  object_sd.object = object;
  object_sd.object_flag = object_flag;
  shader_setup_object_transforms(kg, &object_sd, sd.time);

  if (!(object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    object_position_transform_auto(kg, &object_sd, &geometry.verts[0]);
    object_position_transform_auto(kg, &object_sd, &geometry.verts[1]);
    object_position_transform_auto(kg, &object_sd, &geometry.verts[2]);
    object_normal_transform_auto(kg, &object_sd, &geometry.normals[0]);
    object_normal_transform_auto(kg, &object_sd, &geometry.normals[1]);
    object_normal_transform_auto(kg, &object_sd, &geometry.normals[2]);
  }

  geometry.dPdu = geometry.verts[1] - geometry.verts[0];
  geometry.dPdv = geometry.verts[2] - geometry.verts[0];

  if (!seed.use_smooth_normals) {
    /* For flat shading, ALWAYS use the geometric face normal computed from triangle vertices.
     * Mesh vertex normals can be incorrect (e.g., smoothed, beveled, or from modifiers),
     * causing refraction to use wrong angles. The geometric normal is always correct for
     * flat faces and ensures proper entering/exiting determination for glass. */
    const float3 face_normal = safe_normalize(cross(geometry.dPdu, geometry.dPdv));
    if (is_zero(face_normal)) {
      return false;
    }
if (MPG_DEBUG::GEOMETRY()) {
    const float3 mesh_normal = safe_normalize(geometry.normals[0]);
    printf("MPG DEBUG load_surface_geometry: Flat shading for object %d, prim %d\n", object, prim);
    printf("  Mesh normals (IGNORED):\n");
    printf("    n0: (%.6f, %.6f, %.6f)\n", geometry.normals[0].x, geometry.normals[0].y, geometry.normals[0].z);
    printf("    n1: (%.6f, %.6f, %.6f)\n", geometry.normals[1].x, geometry.normals[1].y, geometry.normals[1].z);
    printf("    n2: (%.6f, %.6f, %.6f)\n", geometry.normals[2].x, geometry.normals[2].y, geometry.normals[2].z);
    printf("  Using geometric face normal: (%.6f, %.6f, %.6f)\n",
           face_normal.x, face_normal.y, face_normal.z);
    if (!is_zero(mesh_normal)) {
      const float consistency = dot(mesh_normal, face_normal);
      if (fabsf(consistency) < 0.9f) {
        printf("  NOTE: Mesh normal differs from geometric normal (dot=%.6f)\n", consistency);
      }
    }
}
    geometry.normals[0] = face_normal;
    geometry.normals[1] = face_normal;
    geometry.normals[2] = face_normal;
  }
  return true;
}

float compute_visibility(KernelGlobals kg,
                         const ShaderData &sd,
                         const MpgSeedRay &seed,
                         const SpecularEval &eval)
{
  if (eval.distance_sl <= 0.0f) {
    return 0.0f;
  }

  /* Skip visibility test for directional lights (sun, etc.) following the
   * Mitsuba reference implementation. Directional lights are infinitely distant
   * and illuminate from a direction rather than a point, so shadow ray tests
   * don't apply. */
  if (seed.light_sample.t == FLT_MAX) {
    return 1.0f;
  }

  Ray shadow_ray;
  float3 offset_normal = eval.normal;
  if (dot(offset_normal, eval.dir_sl) < 0.0f) {
    /* For transmission events the light direction is on the opposite side of the
     * surface normal. Flip the offset normal so that the ray offset moves the
     * origin along the outgoing direction instead of back into the surface. */
    offset_normal = -offset_normal;
  }
  shadow_ray.P = ray_offset(eval.point, offset_normal);
  shadow_ray.D = eval.dir_sl;
  shadow_ray.tmin = 1.0e-4f;
  shadow_ray.tmax = fmaxf(eval.distance_sl - 1e-4f, 0.0f);
  shadow_ray.time = sd.time;
  shadow_ray.self.prim = seed.prim;
  shadow_ray.self.object = seed.object;
  shadow_ray.self.light_prim = seed.light_sample.prim;
  shadow_ray.self.light_object = seed.light_sample.object;

  const bool occluded = scene_intersect_shadow(kg, &shadow_ray, PATH_RAY_SHADOW);
  return occluded ? 0.0f : 1.0f;
}

float3 derivative_specular_reflection(const float3 &dir_ds,
                                      const float3 &d_dir_ds,
                                      const float3 &normal,
                                      const float3 &d_normal)
{
  const float3 dir_in = -dir_ds;
  const float3 d_dir_in = -d_dir_ds;
  const float dot_in_n = dot(dir_in, normal);
  const float d_dot = dot(d_dir_in, normal) + dot(dir_in, d_normal);
  const float3 d_reflect = d_dir_in - 2.0f * (d_dot * normal + dot_in_n * d_normal);
  return d_reflect;
}

float3 derivative_specular_refraction(const float3 &dir_ds,
                                      const float3 &d_dir_ds,
                                      const float3 &normal,
                                      const float3 &d_normal,
                                      const float eta,
                                      const float cos_theta_i,
                                      const float cos_theta_t,
                                      const float sin_theta_i)
{
  const float3 dir_in = -dir_ds;
  const float3 d_dir_in = -d_dir_ds;
  const float d_cos_theta_i = dot(d_dir_in, normal) + dot(dir_in, d_normal);
  const float abs_cos_theta_t = fmaxf(1e-8f, fabsf(cos_theta_t));
  const float denom = copysignf(abs_cos_theta_t, cos_theta_t);
  const float d_cos_theta_t = (eta * eta * cos_theta_i / denom) * d_cos_theta_i;
  const float3 term_dir = eta * d_dir_in;
  const float3 term_normal = (eta * d_cos_theta_i - d_cos_theta_t) * normal +
                             (eta * cos_theta_i - cos_theta_t) * d_normal;
  return term_dir + term_normal;
}

void compute_jacobian(const ShadingPoint &D,
                      const MpgSeedRay &seed,
                      const SpecularSurfaceGeometry &geometry,
                      const SpecularEval &eval,
                      float3 J[2])
{
  const float3 d_dir_ds_du = derivative_normalized(eval.point - D.position, geometry.dPdu);
  const float3 d_dir_ds_dv = derivative_normalized(eval.point - D.position, geometry.dPdv);

  const float3 d_dir_sl_du = compute_light_sample_direction_derivative(
      seed.light_sample, eval.point, geometry.dPdu);
  const float3 d_dir_sl_dv = compute_light_sample_direction_derivative(
      seed.light_sample, eval.point, geometry.dPdv);

  float3 d_spec_du, d_spec_dv;
  if (!eval.refractive) {
    d_spec_du = derivative_specular_reflection(eval.dir_ds, d_dir_ds_du, eval.normal, eval.dNdu);
    d_spec_dv = derivative_specular_reflection(eval.dir_ds, d_dir_ds_dv, eval.normal, eval.dNdv);
  }
  else {
    const float sin_theta_i = sqrtf(fmaxf(0.0f, 1.0f - eval.cos_theta_i * eval.cos_theta_i));
    d_spec_du = derivative_specular_refraction(eval.dir_ds,
                                               d_dir_ds_du,
                                               eval.normal,
                                               eval.dNdu,
                                               eval.eta,
                                               eval.cos_theta_i,
                                               eval.cos_theta_t,
                                               sin_theta_i);
    d_spec_dv = derivative_specular_refraction(eval.dir_ds,
                                               d_dir_ds_dv,
                                               eval.normal,
                                               eval.dNdv,
                                               eval.eta,
                                               eval.cos_theta_i,
                                               eval.cos_theta_t,
                                               sin_theta_i);
  }

  J[0] = d_dir_sl_du - d_spec_du;
  J[1] = d_dir_sl_dv - d_spec_dv;
}

/* Compute derivatives of tangent frame when BSDF frame normal varies across surface.
 * Per Codex Pass 2 #2, Pass 3 #1, Pass 4 #1: When BSDF frame varies (smooth normals),
 * tangent basis also varies and contributes to Jacobian via product rule.
 *
 * For orthonormal frame {s, t, N}, when N rotates the tangents rotate with it.
 * Angular velocity: ω = N × (dN/dx)
 * Tangent derivatives: ds/dx = ω × s, dt/dx = ω × t
 *
 * Reference: Differential geometry of curves and surfaces (do Carmo) */
ccl_device_inline void compute_tangent_frame_derivatives(
    const float3 &normal,
    const float3 &tangent_s,
    const float3 &tangent_t,
    const float3 &dN_du,
    const float3 &dN_dv,
    float3 &ds_du,
    float3 &ds_dv,
    float3 &dt_du,
    float3 &dt_dv)
{
  /* Angular velocity vectors for frame rotation */
  const float3 omega_u = cross(normal, dN_du);
  const float3 omega_v = cross(normal, dN_dv);

  /* Tangent derivatives from rotating frame: d(tangent)/dx = ω_x × tangent */
  ds_du = cross(omega_u, tangent_s);
  ds_dv = cross(omega_v, tangent_s);
  dt_du = cross(omega_u, tangent_t);
  dt_dv = cross(omega_v, tangent_t);
}

/* Compute Jacobian for Mitsuba's half-vector constraint formulation.
 * Constraint: C = [dot(s, h), dot(t, h)] where h = normalize(wi + eta * wo)
 *
 * Per Codex: Full Jacobian includes both half-vector and tangent-frame derivatives:
 *   dC/du = [dot(ds/du, h) + dot(s, dh/du), dot(dt/du, h) + dot(t, dh/du)]
 *
 * Returns 2D Jacobian embedded in 3D vectors (third component is zero). */
void compute_halfvector_jacobian(const ShadingPoint &D,
                                  const MpgSeedRay &seed,
                                  const SpecularSurfaceGeometry &geometry,
                                  const SpecularEval &eval,
                                  const float3 &h,
                                  const float h_eta,
                                  const float3 &tangent_u,
                                  const float3 &tangent_v,
                                  float3 J[2])
{
  /* Compute direction derivatives */
  const float3 d_dir_ds_du = derivative_normalized(eval.point - D.position, geometry.dPdu);
  const float3 d_dir_ds_dv = derivative_normalized(eval.point - D.position, geometry.dPdv);

  const float3 d_dir_sl_du = compute_light_sample_direction_derivative(
      seed.light_sample, eval.point, geometry.dPdu);
  const float3 d_dir_sl_dv = compute_light_sample_direction_derivative(
      seed.light_sample, eval.point, geometry.dPdv);

  /* Derivatives of wi and wo */
  const float3 d_wi_du = -d_dir_ds_du;
  const float3 d_wi_dv = -d_dir_ds_dv;
  const float3 d_wo_du = d_dir_sl_du;
  const float3 d_wo_dv = d_dir_sl_dv;

  /* Compute unnormalized half-vector g = wi + eta * wo (before normalization and negation) */
  const float3 wi = -eval.dir_ds;
  const float3 wo = eval.dir_sl;
  float3 g = wi + h_eta * wo;
  float3 dg_du = d_wi_du + h_eta * d_wo_du;
  float3 dg_dv = d_wi_dv + h_eta * d_wo_dv;

  /* For refraction, g is negated before normalization */
  if (eval.refractive) {
    g = -g;
    dg_du = -dg_du;
    dg_dv = -dg_dv;
  }

  const float g_len = len(g);
  /* Per Codex Pass 1 #2: Mitsuba always normalizes h via reciprocal length and proceeds.
   * Previous check (g_len > 1e-8f) was too strict and could reject solvable configurations.
   * Use much smaller threshold matching typical safe_normalize checks. */
  if (!(g_len > 1e-30f) || !isfinite_safe(g_len)) {
    /* Truly degenerate case - set Jacobian to zero */
    J[0] = make_float3(0.0f, 0.0f, 0.0f);
    J[1] = make_float3(0.0f, 0.0f, 0.0f);
    return;
  }

  /* Derivative of normalized vector: dh/dx = (1/||g||) * (dg/dx - h * dot(h, dg/dx))
   * Per Codex: The 1/||g|| factor applies to BOTH terms! */
  const float inv_g_len = 1.0f / g_len;
  const float3 dh_du = inv_g_len * (dg_du - h * dot(h, dg_du));
  const float3 dh_dv = inv_g_len * (dg_dv - h * dot(h, dg_dv));

  /* Per Codex Pass 2 #2, Pass 3 #1, Pass 4 #1: Include tangent-frame derivatives.
   * Constraint: C = [dot(s, h), dot(t, h)]
   * Full derivative: dC/du = [dot(ds/du, h) + dot(s, dh/du), dot(dt/du, h) + dot(t, dh/du)]
   *
   * Mitsuba's compute_step_halfvector includes dot(ds_du, h) and dot(dt_du, h) terms,
   * which are essential when BSDF frame varies across surface (smooth normals, bump maps). */

  float3 ds_du = zero_float3(), ds_dv = zero_float3();
  float3 dt_du = zero_float3(), dt_dv = zero_float3();

  const bool frame_varies = !is_zero(eval.dNdu) || !is_zero(eval.dNdv);
  if (frame_varies) {
    /* BSDF frame varies across surface - compute tangent frame derivatives */
    compute_tangent_frame_derivatives(eval.normal, tangent_u, tangent_v,
                                     eval.dNdu, eval.dNdv,
                                     ds_du, ds_dv, dt_du, dt_dv);
  }
  /* else: Frame constant (microfacet with constant normal) - derivatives are zero */

  /* Compute full Jacobian including both half-vector and frame derivatives.
   * J[0] = dC/du, J[1] = dC/dv
   * Each row is one constraint: C0 = dot(s, h), C1 = dot(t, h) */
  const float dC0_du = dot(ds_du, h) + dot(tangent_u, dh_du);
  const float dC1_du = dot(dt_du, h) + dot(tangent_v, dh_du);
  const float dC0_dv = dot(ds_dv, h) + dot(tangent_u, dh_dv);
  const float dC1_dv = dot(dt_dv, h) + dot(tangent_v, dh_dv);

  /* Embed in 3D vectors with third component = 0 */
  J[0] = make_float3(dC0_du, dC1_du, 0.0f);
  J[1] = make_float3(dC0_dv, dC1_dv, 0.0f);
}

bool compute_residual_matrix(const ShadingPoint &D,
                             const MpgSeedRay &seed,
                             const SpecularSurfaceGeometry &geometry,
                             const SpecularEval &eval,
                             float matrix[2][2])
{
  float3 J[2];
  compute_jacobian(D, seed, geometry, eval, J);

  if (!isfinite_safe(eval.dir_sl.x) || !isfinite_safe(eval.dir_sl.y) || !isfinite_safe(eval.dir_sl.z)) {
    return false;
  }

  const float dir_len_sq = len_squared(eval.dir_sl);
  if (!isfinite_safe(dir_len_sq) || !(dir_len_sq > 0.0f)) {
    return false;
  }

  float3 tangent_u, tangent_v;
  make_orthonormals(eval.dir_sl, &tangent_u, &tangent_v);

  matrix[0][0] = dot(tangent_u, J[0]);
  matrix[0][1] = dot(tangent_u, J[1]);
  matrix[1][0] = dot(tangent_v, J[0]);
  matrix[1][1] = dot(tangent_v, J[1]);

  return isfinite_safe(matrix[0][0]) && isfinite_safe(matrix[0][1]) &&
         isfinite_safe(matrix[1][0]) && isfinite_safe(matrix[1][1]);
}

/* DEPRECATED: Normal equations solver (Gauss-Newton style).
 * Per Codex Pass 1 #1: This amplifies conditioning problems by computing JᵀJ.
 * Use solve_direct_2x2() instead for single-bounce paths. */
bool solve_step(const float3 &J0,
                const float3 &J1,
                const float3 &residual,
                float2 &delta)
{
  const float a00 = dot(J0, J0);
  const float a01 = dot(J0, J1);
  const float a11 = dot(J1, J1);
  const float b0 = dot(J0, residual);
  const float b1 = dot(J1, residual);

  float det = a00 * a11 - a01 * a01;
  if (!(isfinite_safe(det)) || fabsf(det) < 1e-10f) {
    // Diagonal damping proportional to trace for scale invariance
    const float trace = a00 + a11 + 1e-20f;
    const float lambda = 1e-4f * trace;

    const float a00d = a00 + lambda;
    const float a11d = a11 + lambda;
    det = a00d * a11d - a01 * a01;

    if (!(isfinite_safe(det)) || fabsf(det) < 1e-20f) {
      return false;
    }

    delta.x = (a11d * b0 - a01 * b1) / det;
    delta.y = (a00d * b1 - a01 * b0) / det;
  }
  else {
    delta.x = (a11 * b0 - a01 * b1) / det;
    delta.y = (a00 * b1 - a01 * b0) / det;
  }
  return isfinite_safe(delta.x) && isfinite_safe(delta.y);
}

/* Direct 2×2 linear system solver per Mitsuba reference.
 * Solves: J * delta = C where J is 2×2 Jacobian, C is 2D constraint.
 *
 * Per Codex Pass 1 #1: Mitsuba solves J * dx = C directly without forming normal equations.
 * Normal equations (JᵀJ * dx = JᵀC) square the condition number and can amplify ill-conditioning.
 *
 * Input:
 *   J_cols[0] = [dC0/du, dC1/du, 0]  (first column of Jacobian)
 *   J_cols[1] = [dC0/dv, dC1/dv, 0]  (second column of Jacobian)
 *   residual = [C0, C1, 0]  (constraint violations)
 *
 * Solves:
 *   [dC0/du  dC0/dv] [delta.x]   [C0]
 *   [dC1/du  dC1/dv] [delta.y] = [C1]
 */
bool solve_direct_2x2(const float3 J_cols[2],
                      const float3 &residual,
                      float2 &delta)
{
  /* Extract 2×2 Jacobian matrix elements from column vectors */
  const float j00 = J_cols[0].x;  /* dC0/du */
  const float j10 = J_cols[0].y;  /* dC1/du */
  const float j01 = J_cols[1].x;  /* dC0/dv */
  const float j11 = J_cols[1].y;  /* dC1/dv */

  /* Extract 2D constraint vector */
  const float c0 = residual.x;
  const float c1 = residual.y;

  /* Compute determinant */
  float det = j00 * j11 - j01 * j10;

  /* Check for singular/near-singular Jacobian */
  if (!(isfinite_safe(det)) || fabsf(det) < 1e-10f) {
    /* Apply Levenberg-Marquardt damping: J_damped = J + λI */
    const float trace = fabsf(j00) + fabsf(j11) + 1e-20f;
    const float lambda = 1e-4f * trace;

    const float j00d = j00 + lambda;
    const float j11d = j11 + lambda;
    det = j00d * j11d - j01 * j10;

    if (!(isfinite_safe(det)) || fabsf(det) < 1e-20f) {
      return false;
    }

    /* Solve with damped Jacobian using Cramer's rule */
    delta.x = (j11d * c0 - j01 * c1) / det;
    delta.y = (j00d * c1 - j10 * c0) / det;
  }
  else {
    /* Solve directly using Cramer's rule */
    delta.x = (j11 * c0 - j01 * c1) / det;
    delta.y = (j00 * c1 - j10 * c0) / det;
  }

  return isfinite_safe(delta.x) && isfinite_safe(delta.y);
}

void project_barycentrics(float &u, float &v)
{
  const float w = 1.0f - u - v;
  if (u >= 0.0f && v >= 0.0f && w >= 0.0f) {
    return;
  }
  float3 bary = make_float3(u, v, w);
  bary = max(bary, make_float3(0.0f, 0.0f, 0.0f));
  const float sum = bary.x + bary.y + bary.z;
  if (sum < 1e-10f) {
    bary = make_float3(1.0f, 0.0f, 0.0f);
  }
  else {
    bary /= sum;
  }
  u = bary.x;
  v = bary.y;
}

ShadingPoint shading_point_from_shader_data(const ShaderData &sd)
{
  ShadingPoint shading_point;
  shading_point.position = sd.P;
  shading_point.geometric_normal = sd.Ng;
  shading_point.shading_normal = sd.N;
  shading_point.wo = -sd.wi;
  shading_point.time = sd.time;
  return shading_point;
}

float3 surface_point_from_barycentric(const SpecularSurfaceGeometry &geometry, const float u, const float v)
{
  const float w = 1.0f - u - v;
  return geometry.verts[0] * w + geometry.verts[1] * u + geometry.verts[2] * v;
}

float3 surface_normal_from_barycentric(const SpecularSurfaceGeometry &geometry, const float u, const float v)
{
  return combine_vertex_normals(geometry, u, v);
}

bool build_tangent_basis(const float3 &dXdu, const float3 &dXdv, float3 &tangent_u, float3 &tangent_v)
{
  tangent_u = dXdu;
  const float len_u = len(tangent_u);
  if (!(len_u > 0.0f)) {
if (MPG_DEBUG::GEOMETRY()) {
    printf("MPG DEBUG build_tangent_basis: dXdu is zero-length\n");
}
    return false;
  }
  tangent_u /= len_u;

  tangent_v = dXdv - tangent_u * dot(tangent_u, dXdv);
  const float len_v = len(tangent_v);
  if (!(len_v > 0.0f)) {
if (MPG_DEBUG::GEOMETRY()) {
    printf("MPG DEBUG build_tangent_basis: tangent_v is zero after Gram-Schmidt\n");
    printf("  dXdu=[%.9f, %.9f, %.9f] len=%.12f\n", dXdu.x, dXdu.y, dXdu.z, len_u);
    printf("  dXdv=[%.9f, %.9f, %.9f] len=%.12f\n", dXdv.x, dXdv.y, dXdv.z, len(dXdv));
    printf("  dot(tangent_u, dXdv)=%.12f\n", dot(tangent_u, dXdv));
    printf("  dXdv - projection=[%.9f, %.9f, %.9f] len=%.12f\n",
           tangent_v.x, tangent_v.y, tangent_v.z, len_v);
}
    return false;
  }
  tangent_v /= len_v;
  return true;
}

bool trace_secondary_seed(KernelGlobals kg,
                          const ShaderData &sd,
                          const SpecularSurfaceGeometry &primary_geometry,
                          const float primary_u,
                          const float primary_v,
                          const MpgSeedRay &seed,
                          const SpecularParameters &primary_params,
                          MpgSeedRay &secondary_seed)
{
  const float3 primary_point = surface_point_from_barycentric(primary_geometry, primary_u, primary_v);

  /* Per Codex finding: Use BSDF frame normal for refraction, NOT geometric/vertex normals.
   * Mitsuba and specular_parameters_from_surface use the BSDF frame normal (microfacet->N)
   * for Snell's law. Using geometric normals can send the refracted ray in the wrong
   * direction, hitting diffuse surfaces instead of the expected exit surface. */
  float3 specular_normal;
  if (primary_params.has_normal) {
    /* Use BSDF frame normal from primary_params - highest priority */
    specular_normal = primary_params.normal;
  }
  else {
    /* Fallback to geometric normal if BSDF frame not available */
    float3 primary_normal;
    if (seed.use_smooth_normals) {
      primary_normal = combine_vertex_normals(primary_geometry, primary_u, primary_v);
    }
    else {
      primary_normal = primary_geometry.normals[0];
    }

    if (!is_zero(primary_normal)) {
      specular_normal = safe_normalize(primary_normal);
    }
    else {
      /* Last resort: geometric normal from cross product */
      specular_normal = safe_normalize(cross(primary_geometry.dPdu, primary_geometry.dPdv));
    }
  }

  if (is_zero(specular_normal)) {
    return false;
  }

  /* Compute geometric normal for offset (used later for ray offsetting, not for refraction) */
  const float3 offset_ng_raw = cross(primary_geometry.dPdu, primary_geometry.dPdv);
  bool use_offset_ng = !is_zero(offset_ng_raw);
  float3 offset_ng = use_offset_ng ? safe_normalize(offset_ng_raw) : zero_float3();
  if (use_offset_ng && is_zero(offset_ng)) {
    use_offset_ng = false;
  }

  /* Direction light arrives at primary vertex (FROM receiver TO primary).
   * For Mitsuba path x₀(receiver)→v[0](primary)→v[1](secondary), light travels:
   * receiver→primary, so incoming direction is (primary - receiver). */
  float3 dir_ds = primary_point - sd.P;  // Direction: receiver → primary (light arrival)
  float distance_ds = len(dir_ds);
  if (!(distance_ds > 1e-4f)) {
    return false;
  }
  dir_ds /= distance_ds;

  /* specular_normal already computed from BSDF frame above - use it as-is.
   * The entering/exiting determination happens in compute_specular() using dot(normal, ray). */

if (MPG_DEBUG::PARAMS()) {
  printf("----------------------------------------\n");
  printf("PRIMARY VERTEX COMPUTED:\n");
  printf("  Position: (%.6f, %.6f, %.6f)\n", primary_point.x, primary_point.y, primary_point.z);
  printf("  Normal: (%.6f, %.6f, %.6f)\n", specular_normal.x, specular_normal.y, specular_normal.z);
  printf("  Z relative to receiver: %.6f %s\n",
         primary_point.z - sd.P.z,
         (primary_point.z - sd.P.z) > 0 ? "(ABOVE)" : "(BELOW)");
  printf("\n");
  printf("RAY: Primary → Receiver (light flow direction):\n");
  printf("  Direction: (%.6f, %.6f, %.6f) %s\n",
         dir_ds.x, dir_ds.y, dir_ds.z,
         dir_ds.z > 0 ? "[UPWARD]" : "[DOWNWARD]");
  printf("  Distance: %.6f\n", distance_ds);
  printf("\n");
}

  /* Per Codex: Mitsuba expects direction AWAY from surface (wi = x_prev - x_cur).
   * dir_ds is receiver → primary (toward surface), so negate it. */
  bool tir = false;
  float cos_theta_i = 0.0f;
  float cos_theta_t = 0.0f;
  float eta_used = 1.0f;
  const float3 dir_sl = compute_specular(
      -dir_ds, specular_normal, primary_params, tir, cos_theta_i, cos_theta_t, eta_used);

if (MPG_DEBUG::PARAMS()) {
  printf("REFRACTION at primary vertex:\n");
  printf("  Incoming ray (from diffuse): (%.6f, %.6f, %.6f) %s\n",
         dir_ds.x, dir_ds.y, dir_ds.z,
         dir_ds.z > 0 ? "[UPWARD]" : "[DOWNWARD]");
  printf("  Surface normal: (%.6f, %.6f, %.6f) %s\n",
         specular_normal.x, specular_normal.y, specular_normal.z,
         specular_normal.z > 0 ? "[UP]" : "[DOWN]");
  printf("  IOR used: %.6f (%s)\n", eta_used,
         eta_used > 1.0 ? "exiting glass→air" : "entering air→glass");
  printf("  Outgoing ray (toward receiver): (%.6f, %.6f, %.6f) %s\n",
         dir_sl.x, dir_sl.y, dir_sl.z,
         dir_sl.z > 0 ? "[UPWARD]" : "[DOWNWARD]");
  if (tir) {
    printf("  RESULT: TOTAL INTERNAL REFLECTION (angle exceeds critical angle)\n");
  }
  else {
    printf("  RESULT: Refraction successful\n");
  }
  printf("\n");
}

  (void)cos_theta_i;
  (void)cos_theta_t;
  (void)eta_used;
  if (tir || is_zero(dir_sl)) {
    return false;
  }

  if (seed.object == OBJECT_NONE || seed.prim < 0) {
    return false;
  }

  ShaderData offset_sd = {};
  offset_sd.object = seed.object;
  offset_sd.prim = seed.prim;
  offset_sd.time = sd.time;
  offset_sd.object_flag = kernel_data_fetch(object_flag, offset_sd.object);
  offset_sd.type = (offset_sd.object_flag & SD_OBJECT_MOTION) ? PRIMITIVE_MOTION_TRIANGLE :
                                                                PRIMITIVE_TRIANGLE;
  shader_setup_object_transforms(kg, &offset_sd, offset_sd.time);

  float3 offset_n = specular_normal;
  if (dot(offset_n, dir_sl) < 0.0f) {
    offset_n = -offset_n;
  }
  offset_n = safe_normalize(offset_n);
  if (is_zero(offset_n)) {
    return false;
  }
  if (use_offset_ng) {
    offset_ng = offset_n;
    offset_sd.Ng = offset_ng;
  }
  else {
    offset_sd.Ng = offset_n;
  }

  /* For refraction through thin geometry, we need to handle the case where entry and exit
   * surfaces are very close (solidified planes). Use a tiny offset in the refracted direction
   * to move slightly away from the entry surface before tracing. This matches how Mitsuba
   * handles thin dielectrics by ensuring we don't immediately re-hit the entry surface. */
  const float tiny_offset = 1e-5f;
  const float3 offset_point = primary_point + dir_sl * tiny_offset;

  Ray ray;
  ray.P = offset_point;
  ray.D = dir_sl;
  ray.tmin = 0.0f;  /* No additional tmin needed since we already offset the starting point */
  ray.tmax = FLT_MAX;
  ray.time = sd.time;
  /* Leave self references clear so refraction chains can re-hit the same triangle from
   * the opposite side when tracing the exit interface (regression: thin pane double bounce). */
  ray.self.prim = PRIM_NONE;
  ray.self.object = OBJECT_NONE;
  ray.self.light_prim = PRIM_NONE;
  ray.self.light_object = OBJECT_NONE;

  Intersection isect;
  if (!scene_intersect(kg, &ray, PATH_RAY_ALL_VISIBILITY, &isect)) {
if (MPG_DEBUG::PARAMS()) {
    printf("  trace_secondary_seed: NO intersection found\n");
}
    return false;
  }

  if (!(isect.type & PRIMITIVE_TRIANGLE)) {
    return false;
  }

if (MPG_DEBUG::PARAMS()) {
  const float3 hit_point = ray.P + ray.D * isect.t;
  printf("SECONDARY VERTEX FOUND:\n");
  printf("  Position: (%.6f, %.6f, %.6f)\n", hit_point.x, hit_point.y, hit_point.z);
  printf("  Object: %d, Prim: %d\n", isect.object, isect.prim);
  printf("  Distance from primary: %.6f\n", isect.t);
  printf("  Z relative to primary: %.6f %s\n",
         hit_point.z - primary_point.z,
         (hit_point.z - primary_point.z) > 0 ? "(ABOVE)" : "(BELOW)");
  printf("\n");

  const char *hit_surface = "UNKNOWN";
  if (isect.object != seed.object) {
    hit_surface = "DIFFERENT OBJECT (likely diffuse plane)";
  } else if (hit_point.z > primary_point.z) {
    hit_surface = "GLASS TOP (exit surface)";
  } else {
    hit_surface = "GLASS BOTTOM or OTHER";
  }
  printf("  Identified as: %s\n", hit_surface);
  printf("\n");
}

  secondary_seed = seed;
  secondary_seed.object = isect.object;
  secondary_seed.prim = isect.prim;
  secondary_seed.bary_u = isect.u;
  secondary_seed.bary_v = isect.v;

  /* With full-path encoding, tau_bits encodes all bounces positionally:
   * - Bit 0 = primary bounce type
   * - Bit 1 = secondary bounce type
   * No shifting or modification needed - bounce_index parameter selects which bit to query. */

  /* Force flat shading for ALL MPG vertices to ensure tangent basis orthogonality.
   * Smooth normals (interpolated from vertices) are not guaranteed to be orthogonal
   * to the geometric tangent basis, causing Newton solver divergence. */
  secondary_seed.use_smooth_normals = false;

  /* Per Codex finding: Validate that the hit surface is actually specular before accepting.
   * Without this check, we accept non-specular (diffuse) hits, which later fail in
   * specular_parameters_from_surface with MPG_FAILURE_NO_SPECULAR, blocking the intended
   * fallback to single-bounce. Check if the surface has specular BSDFs. */
  {
    ShaderData check_sd = {};
    shader_setup_from_ray(kg, &check_sd, &ray, const_cast<Intersection *>(&isect));
    const ConstIntegratorState integrator_state = nullptr;
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
        kg, integrator_state, &check_sd, nullptr, PATH_RAY_CAMERA, true);

    bool has_specular = false;
    for (int i = 0; i < check_sd.num_closure; ++i) {
      const ShaderClosure *c = &check_sd.closure[i];
      if (!CLOSURE_IS_BSDF(c->type)) {
        continue;
      }
      const bool is_micro = CLOSURE_IS_BSDF_MICROFACET(c->type);
      const bool is_singular = CLOSURE_IS_BSDF_SINGULAR(c->type);
      const bool is_glass = CLOSURE_IS_GLASS(c->type);

      /* Glass BSDFs are specular surfaces that MPG can use as intermediate vertices */
      if (is_singular || is_glass) {
        has_specular = true;
        break;
      }
      if (is_micro) {
        const MicrofacetBsdf *mf = reinterpret_cast<const MicrofacetBsdf *>(c);
        const float a = fmaxf(mf->alpha_x, mf->alpha_y);
        /* Match solver threshold (1e-6) - only near-delta microfacets are specular */
        if (a <= 1e-6f) {
          has_specular = true;
          break;
        }
      }
    }

    if (!has_specular) {
if (MPG_DEBUG::PARAMS()) {
      printf("  trace_secondary_seed: Hit surface is NOT specular (diffuse/non-specular)\n");
      printf("  Rejecting to allow single-bounce fallback\n");
}
      return false;
    }
  }

  /* Mitsuba approach: Accept ray-traced secondary vertices that have specular BSDFs.
   * Let the Newton solver naturally reject infeasible seeds through convergence failure.
   * This matches the reference implementation which traces rays iteratively for multi-bounce
   * paths and relies on Newton's constraint evaluation to filter invalid configurations. */
  return true;
}

struct DoubleBounceEval {
  SpecularEval primary;
  SpecularEval secondary;
  float residual[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

bool evaluate_double_bounce(const ShadingPoint &receiver,
                            const SpecularSurfaceGeometry &primary_geometry,
                            const SpecularParameters &primary_params,
                            const SpecularSurfaceGeometry &secondary_geometry,
                            const SpecularParameters &secondary_params,
                            const LightSample &light_sample,
                            const float u1,
                            const float v1,
                            const float u2,
                            const float v2,
                            const bool primary_use_smooth_normals,
                            const bool secondary_use_smooth_normals,
                            DoubleBounceEval &eval)
{
  const float3 secondary_point = surface_point_from_barycentric(secondary_geometry, u2, v2);

  /* For double-bounce, the primary vertex treats the secondary vertex as a finite light.
   * Initialize light_sample with the secondary point and ensure t != FLT_MAX so derivatives
   * are computed correctly (not treated as directional). */
  MpgSeedRay primary_seed = {};
  primary_seed.light_sample.P = secondary_point;
  primary_seed.light_sample.t = 0.0f;  /* Finite light, distance will be computed from positions */
  primary_seed.use_smooth_normals = primary_use_smooth_normals;

  evaluate_specular(receiver, primary_seed, primary_geometry, primary_params, u1, v1, eval.primary);
  if (!isfinite_safe(eval.primary.distance_ds) || !(eval.primary.distance_ds > 1e-4f)) {
if (MPG_DEBUG::EVAL_FAIL()) {
    printf("  evaluate_double_bounce FAIL: Primary distance_ds invalid (%.6e)\n", eval.primary.distance_ds);
}
    return false;
  }
  if (!isfinite_safe(eval.primary.distance_sl) || !(eval.primary.distance_sl > 1e-4f)) {
if (MPG_DEBUG::EVAL_FAIL()) {
    printf("  evaluate_double_bounce FAIL: Primary distance_sl invalid (%.6e)\n", eval.primary.distance_sl);
}
    return false;
  }
  if (eval.primary.tir) {
if (MPG_DEBUG::EVAL_FAIL()) {
    printf("  evaluate_double_bounce FAIL: Primary TIR\n");
}
    return false;
  }

  ShadingPoint intermediate_point = receiver;
  intermediate_point.position = eval.primary.point;
  intermediate_point.geometric_normal = eval.primary.normal;
  intermediate_point.shading_normal = eval.primary.normal;

  MpgSeedRay secondary_seed = {};
  secondary_seed.light_sample = light_sample;
  secondary_seed.use_smooth_normals = secondary_use_smooth_normals;

  evaluate_specular(intermediate_point, secondary_seed, secondary_geometry, secondary_params, u2, v2, eval.secondary);
  if (!isfinite_safe(eval.secondary.distance_ds) || !(eval.secondary.distance_ds > 1e-4f)) {
if (MPG_DEBUG::EVAL_FAIL()) {
    printf("  evaluate_double_bounce FAIL: Secondary distance_ds invalid (%.6e)\n", eval.secondary.distance_ds);
}
    return false;
  }
  if (!isfinite_safe(eval.secondary.distance_sl) || !(eval.secondary.distance_sl > 1e-4f)) {
if (MPG_DEBUG::EVAL_FAIL()) {
    printf("  evaluate_double_bounce FAIL: Secondary distance_sl invalid (%.6e)\n", eval.secondary.distance_sl);
}
    return false;
  }
  if (eval.secondary.tir) {
if (MPG_DEBUG::EVAL_FAIL()) {
    printf("  evaluate_double_bounce FAIL: Secondary TIR\n");
}
    return false;
  }

  /* Per Codex: Use BSDF frame tangents (from eval.tangent_u/v), not geometric tangents.
   * Mitsuba projects onto frame.s and frame.t (orthogonal to BSDF frame normal). */
  if (is_zero(eval.primary.tangent_u) || is_zero(eval.primary.tangent_v)) {
if (MPG_DEBUG::EVAL_FAIL()) {
    printf("  evaluate_double_bounce FAIL: Primary BSDF frame tangents not initialized\n");
}
    return false;
  }
  eval.residual[0] = dot(eval.primary.residual, eval.primary.tangent_u);
  eval.residual[1] = dot(eval.primary.residual, eval.primary.tangent_v);

  if (is_zero(eval.secondary.tangent_u) || is_zero(eval.secondary.tangent_v)) {
if (MPG_DEBUG::EVAL_FAIL()) {
    printf("  evaluate_double_bounce FAIL: Secondary BSDF frame tangents not initialized\n");
}
    return false;
  }
  eval.residual[2] = dot(eval.secondary.residual, eval.secondary.tangent_u);
  eval.residual[3] = dot(eval.secondary.residual, eval.secondary.tangent_v);

if (MPG_DEBUG::NEWTON_DETAIL()) {
  const float norm = sqrtf(eval.residual[0]*eval.residual[0] + eval.residual[1]*eval.residual[1] +
                           eval.residual[2]*eval.residual[2] + eval.residual[3]*eval.residual[3]);
  printf("    Residual after projection: C=(%.4f, %.4f, %.4f, %.4f), norm=%.4f\n",
         eval.residual[0], eval.residual[1], eval.residual[2], eval.residual[3], norm);
}

  return true;
}

bool compute_double_bounce_jacobian_analytical(const ShadingPoint &receiver,
                                               const SpecularSurfaceGeometry &primary_geometry,
                                               const SpecularSurfaceGeometry &secondary_geometry,
                                               const DoubleBounceEval &eval,
                                               const float u1,
                                               const float v1,
                                               const float u2,
                                               const float v2,
                                               float J[4][4])
{
  /* Analytical Jacobian for double-bounce manifold constraints.
   *
   * The constraint residual is C = [C1, C2, C3, C4]^T where:
   * - C1, C2: Primary vertex half-vector constraint projected onto tangent frame
   * - C3, C4: Secondary vertex half-vector constraint projected onto tangent frame
   *
   * The Jacobian is a 4x4 matrix:
   * J = [dC1/du1  dC1/dv1  dC1/du2  dC1/dv2]
   *     [dC2/du1  dC2/dv1  dC2/du2  dC2/dv2]
   *     [dC3/du1  dC3/dv1  dC3/du2  dC3/dv2]
   *     [dC4/du1  dC4/dv1  dC4/du2  dC4/dv2]
   *
   * Block structure:
   * - J[0:2, 0:2]: Primary constraint w.r.t. primary params (diagonal block)
   * - J[0:2, 2:4]: Primary constraint w.r.t. secondary params (coupling block)
   * - J[2:4, 0:2]: Secondary constraint w.r.t. primary params (coupling block)
   * - J[2:4, 2:4]: Secondary constraint w.r.t. secondary params (diagonal block)
   */

  /* === Use BSDF frame tangent frames (not geometric tangents) === */
  /* Per Codex: Mitsuba uses frame.s and frame.t (orthogonal to BSDF frame normal).
   * These are already computed in evaluate_specular(). */
  const float3 primary_tangent_u = eval.primary.tangent_u;
  const float3 primary_tangent_v = eval.primary.tangent_v;
  if (is_zero(primary_tangent_u) || is_zero(primary_tangent_v)) {
    return false;
  }

  const float3 secondary_tangent_u = eval.secondary.tangent_u;
  const float3 secondary_tangent_v = eval.secondary.tangent_v;
  if (is_zero(secondary_tangent_u) || is_zero(secondary_tangent_v)) {
    return false;
  }

  /* === Compute half-vectors for both vertices === */
  /* CRITICAL: Must compute eta dynamically using dot(wi, gn) like in evaluate_specular().
   * Using eval.primary.eta (from params.backfacing) causes Jacobian/constraint mismatch! */
  const float3 primary_wi = -eval.primary.dir_ds;
  const float3 primary_wo = eval.primary.dir_sl;

  /* Compute primary half-vector eta dynamically matching evaluate_specular() */
  float primary_h_eta = 1.0f;
  if (eval.primary.refractive) {
    const float3 primary_gn_raw = safe_normalize(cross(primary_geometry.dPdu, primary_geometry.dPdv));

    /* Align geometric normal to BSDF frame (Mitsuba: masked(gn, dot(n, gn) < 0) *= -1) */
    float3 primary_gn = primary_gn_raw;
    if (dot(eval.primary.normal, primary_gn_raw) < 0.0f) {
      primary_gn = -primary_gn_raw;
    }

    const float primary_dot_wi_gn = dot(primary_wi, primary_gn);
    /* Reconstruct base_eta from Snell's law eta */
    const float primary_base_eta = (eval.primary.eta >= 1.0f) ? eval.primary.eta : (1.0f / eval.primary.eta);

    if (primary_dot_wi_gn < 0.0f) {
      primary_h_eta = 1.0f / fmaxf(primary_base_eta, 1e-6f);  /* Exiting */
    }
    else {
      primary_h_eta = primary_base_eta;  /* Entering */
    }
  }

  float3 primary_g = primary_wi + primary_h_eta * primary_wo;
  if (eval.primary.refractive) {
    primary_g = -primary_g;
  }
  const float primary_g_len = len(primary_g);
  /* Per Codex Pass 1 #2: Use relaxed threshold like Mitsuba */
  if (!(primary_g_len > 1e-30f) || !isfinite_safe(primary_g_len)) {
    return false;
  }
  const float3 primary_h = primary_g / primary_g_len;

  const float3 secondary_wi = -eval.secondary.dir_ds;
  const float3 secondary_wo = eval.secondary.dir_sl;

  /* Compute secondary half-vector eta dynamically matching evaluate_specular() */
  float secondary_h_eta = 1.0f;
  if (eval.secondary.refractive) {
    const float3 secondary_gn_raw = safe_normalize(cross(secondary_geometry.dPdu, secondary_geometry.dPdv));

    /* Align geometric normal to BSDF frame (Mitsuba: masked(gn, dot(n, gn) < 0) *= -1) */
    float3 secondary_gn = secondary_gn_raw;
    if (dot(eval.secondary.normal, secondary_gn_raw) < 0.0f) {
      secondary_gn = -secondary_gn_raw;
    }

    const float secondary_dot_wi_gn = dot(secondary_wi, secondary_gn);
    /* Reconstruct base_eta from Snell's law eta */
    const float secondary_base_eta = (eval.secondary.eta >= 1.0f) ? eval.secondary.eta : (1.0f / eval.secondary.eta);

    if (secondary_dot_wi_gn < 0.0f) {
      secondary_h_eta = 1.0f / fmaxf(secondary_base_eta, 1e-6f);  /* Exiting */
    }
    else {
      secondary_h_eta = secondary_base_eta;  /* Entering */
    }
  }

  float3 secondary_g = secondary_wi + secondary_h_eta * secondary_wo;
  if (eval.secondary.refractive) {
    secondary_g = -secondary_g;
  }
  const float secondary_g_len = len(secondary_g);
  /* Per Codex Pass 1 #2: Use relaxed threshold like Mitsuba */
  if (!(secondary_g_len > 1e-30f) || !isfinite_safe(secondary_g_len)) {
    return false;
  }
  const float3 secondary_h = secondary_g / secondary_g_len;

  /* === DIAGONAL BLOCK 1: J[0:2, 0:2] - Primary constraint w.r.t. primary params === */
  /*
   * For the primary vertex:
   * - wi direction: from receiver to primary vertex (fixed receiver, moving primary)
   * - wo direction: from primary vertex to secondary vertex (moving primary, moving secondary)
   *
   * When u1, v1 change:
   * - Primary vertex position changes
   * - wi changes (derivative_normalized)
   * - wo changes (depends on how primary moves relative to secondary)
   */
  const float3 d_primary_wi_du1 = derivative_normalized(eval.primary.point - receiver.position, primary_geometry.dPdu);
  const float3 d_primary_wi_dv1 = derivative_normalized(eval.primary.point - receiver.position, primary_geometry.dPdv);

  /* For wo: direction from primary to secondary vertex. When primary moves, this changes. */
  const float3 d_primary_wo_du1 = derivative_normalized(eval.secondary.point - eval.primary.point, -primary_geometry.dPdu);
  const float3 d_primary_wo_dv1 = derivative_normalized(eval.secondary.point - eval.primary.point, -primary_geometry.dPdv);

  /* Compute dg/du1 and dg/dv1 for primary vertex */
  float3 d_primary_g_du1 = d_primary_wi_du1 + primary_h_eta * d_primary_wo_du1;
  float3 d_primary_g_dv1 = d_primary_wi_dv1 + primary_h_eta * d_primary_wo_dv1;

  if (eval.primary.refractive) {
    d_primary_g_du1 = -d_primary_g_du1;
    d_primary_g_dv1 = -d_primary_g_dv1;
  }

  /* Derivative of normalized half-vector: dh/dx = (1/||g||) * (dg/dx - h * dot(h, dg/dx))
   * Per Codex: The 1/||g|| factor applies to BOTH terms! */
  const float inv_primary_g_len = 1.0f / primary_g_len;
  const float3 d_primary_h_du1 = inv_primary_g_len * (d_primary_g_du1 - primary_h * dot(primary_h, d_primary_g_du1));
  const float3 d_primary_h_dv1 = inv_primary_g_len * (d_primary_g_dv1 - primary_h * dot(primary_h, d_primary_g_dv1));

  /* Per Codex Pass 2 #2, Pass 3 #1, Pass 4 #1: Include tangent-frame derivatives */
  float3 d_primary_s_du1 = zero_float3(), d_primary_s_dv1 = zero_float3();
  float3 d_primary_t_du1 = zero_float3(), d_primary_t_dv1 = zero_float3();

  const bool primary_frame_varies = !is_zero(eval.primary.dNdu) || !is_zero(eval.primary.dNdv);
  if (primary_frame_varies) {
    compute_tangent_frame_derivatives(eval.primary.normal, primary_tangent_u, primary_tangent_v,
                                     eval.primary.dNdu, eval.primary.dNdv,
                                     d_primary_s_du1, d_primary_s_dv1,
                                     d_primary_t_du1, d_primary_t_dv1);
  }

  /* Project onto tangent frame with full derivative including frame variation */
  J[0][0] = dot(d_primary_s_du1, primary_h) + dot(primary_tangent_u, d_primary_h_du1);  // dC1/du1
  J[0][1] = dot(d_primary_s_dv1, primary_h) + dot(primary_tangent_u, d_primary_h_dv1);  // dC1/dv1
  J[1][0] = dot(d_primary_t_du1, primary_h) + dot(primary_tangent_v, d_primary_h_du1);  // dC2/du1
  J[1][1] = dot(d_primary_t_dv1, primary_h) + dot(primary_tangent_v, d_primary_h_dv1);  // dC2/dv1

  /* === COUPLING BLOCK 1: J[0:2, 2:4] - Primary constraint w.r.t. secondary params === */
  /*
   * When u2, v2 change:
   * - Secondary vertex position changes
   * - Primary wi stays the same (receiver to primary is independent)
   * - Primary wo changes (primary to secondary direction changes)
   */
  const float3 d_primary_wo_du2 = derivative_normalized(eval.secondary.point - eval.primary.point, secondary_geometry.dPdu);
  const float3 d_primary_wo_dv2 = derivative_normalized(eval.secondary.point - eval.primary.point, secondary_geometry.dPdv);

  float3 d_primary_g_du2 = primary_h_eta * d_primary_wo_du2;  // wi doesn't change
  float3 d_primary_g_dv2 = primary_h_eta * d_primary_wo_dv2;

  if (eval.primary.refractive) {
    d_primary_g_du2 = -d_primary_g_du2;
    d_primary_g_dv2 = -d_primary_g_dv2;
  }

  const float3 d_primary_h_du2 = inv_primary_g_len * (d_primary_g_du2 - primary_h * dot(primary_h, d_primary_g_du2));
  const float3 d_primary_h_dv2 = inv_primary_g_len * (d_primary_g_dv2 - primary_h * dot(primary_h, d_primary_g_dv2));

  J[0][2] = dot(primary_tangent_u, d_primary_h_du2);  // dC1/du2
  J[0][3] = dot(primary_tangent_u, d_primary_h_dv2);  // dC1/dv2
  J[1][2] = dot(primary_tangent_v, d_primary_h_du2);  // dC2/du2
  J[1][3] = dot(primary_tangent_v, d_primary_h_dv2);  // dC2/dv2

  /* === COUPLING BLOCK 2: J[2:4, 0:2] - Secondary constraint w.r.t. primary params === */
  /*
   * When u1, v1 change:
   * - Primary vertex position changes
   * - Secondary wi = normalize(primary - secondary) changes
   * - Secondary wo stays the same (secondary to light is independent of primary)
   *
   * Per Mitsuba reference: wi = x_prev - x_cur, so for secondary vertex:
   * wi_secondary = normalize(primary.point - secondary.point)
   * When primary moves by +dPdu, derivative is computed with respect to the first argument.
   */
  const float3 d_secondary_wi_du1 = derivative_normalized(eval.primary.point - eval.secondary.point, primary_geometry.dPdu);
  const float3 d_secondary_wi_dv1 = derivative_normalized(eval.primary.point - eval.secondary.point, primary_geometry.dPdv);

  float3 d_secondary_g_du1 = d_secondary_wi_du1;  // wo doesn't change
  float3 d_secondary_g_dv1 = d_secondary_wi_dv1;

  if (eval.secondary.refractive) {
    d_secondary_g_du1 = -d_secondary_g_du1;
    d_secondary_g_dv1 = -d_secondary_g_dv1;
  }

  const float inv_secondary_g_len = 1.0f / secondary_g_len;
  const float3 d_secondary_h_du1 = inv_secondary_g_len * (d_secondary_g_du1 - secondary_h * dot(secondary_h, d_secondary_g_du1));
  const float3 d_secondary_h_dv1 = inv_secondary_g_len * (d_secondary_g_dv1 - secondary_h * dot(secondary_h, d_secondary_g_dv1));

  J[2][0] = dot(secondary_tangent_u, d_secondary_h_du1);  // dC3/du1
  J[2][1] = dot(secondary_tangent_u, d_secondary_h_dv1);  // dC3/dv1
  J[3][0] = dot(secondary_tangent_v, d_secondary_h_du1);  // dC4/du1
  J[3][1] = dot(secondary_tangent_v, d_secondary_h_dv1);  // dC4/dv1

  /* === DIAGONAL BLOCK 2: J[2:4, 2:4] - Secondary constraint w.r.t. secondary params === */
  /*
   * When u2, v2 change:
   * - Secondary vertex position changes
   * - Secondary wi = normalize(primary - secondary) changes
   * - Secondary wo changes (secondary to light direction changes)
   *
   * Per Mitsuba: wi_secondary = normalize(primary - secondary)
   * When secondary moves by +dPdu, derivative is -dPdu (moving away from primary).
   */
  const float3 d_secondary_wi_du2 = derivative_normalized(eval.primary.point - eval.secondary.point, -secondary_geometry.dPdu);
  const float3 d_secondary_wi_dv2 = derivative_normalized(eval.primary.point - eval.secondary.point, -secondary_geometry.dPdv);

  /* For wo: compute light direction derivative (handles both finite and directional lights) */
  MpgSeedRay temp_seed;
  temp_seed.light_sample = eval.secondary.light_sample;
  const float3 d_secondary_wo_du2 = compute_light_sample_direction_derivative(
      temp_seed.light_sample, eval.secondary.point, secondary_geometry.dPdu);
  const float3 d_secondary_wo_dv2 = compute_light_sample_direction_derivative(
      temp_seed.light_sample, eval.secondary.point, secondary_geometry.dPdv);

  float3 d_secondary_g_du2 = d_secondary_wi_du2 + secondary_h_eta * d_secondary_wo_du2;
  float3 d_secondary_g_dv2 = d_secondary_wi_dv2 + secondary_h_eta * d_secondary_wo_dv2;

  if (eval.secondary.refractive) {
    d_secondary_g_du2 = -d_secondary_g_du2;
    d_secondary_g_dv2 = -d_secondary_g_dv2;
  }

  const float3 d_secondary_h_du2 = inv_secondary_g_len * (d_secondary_g_du2 - secondary_h * dot(secondary_h, d_secondary_g_du2));
  const float3 d_secondary_h_dv2 = inv_secondary_g_len * (d_secondary_g_dv2 - secondary_h * dot(secondary_h, d_secondary_g_dv2));

  /* Per Codex Pass 2 #2, Pass 3 #1, Pass 4 #1: Include tangent-frame derivatives */
  float3 d_secondary_s_du2 = zero_float3(), d_secondary_s_dv2 = zero_float3();
  float3 d_secondary_t_du2 = zero_float3(), d_secondary_t_dv2 = zero_float3();

  const bool secondary_frame_varies = !is_zero(eval.secondary.dNdu) || !is_zero(eval.secondary.dNdv);
  if (secondary_frame_varies) {
    compute_tangent_frame_derivatives(eval.secondary.normal, secondary_tangent_u, secondary_tangent_v,
                                     eval.secondary.dNdu, eval.secondary.dNdv,
                                     d_secondary_s_du2, d_secondary_s_dv2,
                                     d_secondary_t_du2, d_secondary_t_dv2);
  }

  J[2][2] = dot(d_secondary_s_du2, secondary_h) + dot(secondary_tangent_u, d_secondary_h_du2);  // dC3/du2
  J[2][3] = dot(d_secondary_s_dv2, secondary_h) + dot(secondary_tangent_u, d_secondary_h_dv2);  // dC3/dv2
  J[3][2] = dot(d_secondary_t_du2, secondary_h) + dot(secondary_tangent_v, d_secondary_h_du2);  // dC4/du2
  J[3][3] = dot(d_secondary_t_dv2, secondary_h) + dot(secondary_tangent_v, d_secondary_h_dv2);  // dC4/dv2

  return true;
}

bool compute_double_bounce_jacobian(KernelGlobals kg,
                                    const ShaderData &sd,
                                    const MpgSeedRay &primary_seed,
                                    const MpgSeedRay &secondary_seed,
                                    const ShadingPoint &receiver,
                                    const SpecularSurfaceGeometry &primary_geometry,
                                    const SpecularSurfaceGeometry &secondary_geometry,
                                    const LightSample &light_sample,
                                    const float u1,
                                    const float v1,
                                    const float u2,
                                    const float v2,
                                    const bool primary_use_smooth_normals,
                                    const bool secondary_use_smooth_normals,
                                    const float base_residual[4],
                                    float J[4][4])
{
  (void)kg;
  (void)sd;
  (void)primary_seed;
  (void)secondary_seed;
  (void)light_sample;
  (void)primary_use_smooth_normals;
  (void)secondary_use_smooth_normals;
  (void)base_residual;

  /* Use analytical Jacobian instead of finite differences.
   * The eval structure is recomputed here, but this is necessary to have
   * all the geometric information needed for the analytical derivatives. */
  DoubleBounceEval eval;
  SpecularParameters primary_params;
  if (!specular_parameters_from_surface(
          kg, sd, primary_geometry, primary_seed, 0, u1, v1, primary_params))
  {
    return false;
  }

  ShaderData primary_sd;
  if (!build_primary_shading_data(sd, primary_geometry, primary_seed, u1, v1, primary_sd)) {
    return false;
  }

  SpecularParameters secondary_params;
  if (!specular_parameters_from_surface(kg,
                                        primary_sd,
                                        secondary_geometry,
                                        secondary_seed,
                                        1,
                                        u2,
                                        v2,
                                        secondary_params))
  {
    return false;
  }

  if (!evaluate_double_bounce(receiver,
                              primary_geometry,
                              primary_params,
                              secondary_geometry,
                              secondary_params,
                              light_sample,
                              u1,
                              v1,
                              u2,
                              v2,
                              primary_use_smooth_normals,
                              secondary_use_smooth_normals,
                              eval))
  {
    return false;
  }

  return compute_double_bounce_jacobian_analytical(receiver,
                                                   primary_geometry,
                                                   secondary_geometry,
                                                   eval,
                                                   u1,
                                                   v1,
                                                   u2,
                                                   v2,
                                                   J);
}

bool solve_linear_system_4x4(const float J[4][4], const float rhs[4], float delta[4])
{
  double mat[4][5];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      mat[i][j] = double(J[i][j]);
    }
    mat[i][4] = double(rhs[i]);
  }

  for (int i = 0; i < 4; ++i) {
    int pivot = i;
    double max_abs = std::fabs(mat[i][i]);
    for (int row = i + 1; row < 4; ++row) {
      const double value = std::fabs(mat[row][i]);
      if (value > max_abs) {
        max_abs = value;
        pivot = row;
      }
    }

    if (max_abs < 1.0e-12) {
      return false;
    }

    if (pivot != i) {
      for (int col = i; col <= 4; ++col) {
        std::swap(mat[i][col], mat[pivot][col]);
      }
    }

    const double inv = 1.0 / mat[i][i];
    for (int col = i; col <= 4; ++col) {
      mat[i][col] *= inv;
    }

    for (int row = 0; row < 4; ++row) {
      if (row == i) {
        continue;
      }
      const double factor = mat[row][i];
      for (int col = i; col <= 4; ++col) {
        mat[row][col] -= factor * mat[i][col];
      }
    }
  }

  for (int i = 0; i < 4; ++i) {
    delta[i] = float(mat[i][4]);
  }
  return true;
}

float compute_segment_visibility(KernelGlobals kg,
                                 const float3 &start_point,
                                 const float3 &start_normal,
                                 const float3 &end_point,
                                 const float time,
                                 const int skip_object,
                                 const int skip_prim,
                                 const int skip_light_object,
                                 const int skip_light_prim)
{
  float3 dir = end_point - start_point;
  const float distance = len(dir);
  if (!(distance > 1e-4f)) {
    return 0.0f;
  }
  dir /= distance;

  float3 offset_normal = start_normal;
  if (is_zero(offset_normal)) {
    offset_normal = dir;
  }
  if (dot(offset_normal, dir) < 0.0f) {
    offset_normal = -offset_normal;
  }

  Ray ray;
  ray.P = ray_offset(start_point, offset_normal);
  ray.D = dir;
  ray.tmin = 1.0e-4f;
  ray.tmax = fmaxf(distance - 1.0e-4f, 0.0f);
  ray.time = time;
  ray.self.prim = skip_prim;
  ray.self.object = skip_object;
  ray.self.light_prim = skip_light_prim;
  ray.self.light_object = skip_light_object;

  const bool occluded = scene_intersect_shadow(kg, &ray, PATH_RAY_SHADOW);
  return occluded ? 0.0f : 1.0f;
}

/* Compute 2D offset normal from path guiding data.
 * Projects the guide's mean direction onto the surface tangent frame.
 * Returns zero offset if guide has no valid direction (rbar too small). */
ccl_device_inline float2 compute_guide_offset_normal(const GuideSummary &guide,
                                                      const float3 &tangent_u,
                                                      const float3 &tangent_v,
                                                      const float3 &normal)
{
  /* Only use guide offset when path guiding has a reliable direction.
   * This threshold matches the one used in seeding (mpg_seed.cpp:616). */
  const bool has_direction = (guide.rbar > 1.0e-4f);

  if (!has_direction || is_zero(guide.mean_dir)) {
    return make_float2(0.0f, 0.0f);
  }

  /* Normalize the guide direction */
  const float3 guide_dir = safe_normalize(guide.mean_dir);

  if (is_zero(guide_dir)) {
    return make_float2(0.0f, 0.0f);
  }

  /* Project the guide direction onto the surface tangent frame.
   * This gives us the 2D offset normal N in Mitsuba's constraint C = H - N.
   * We project the *reflected* guide direction (in the upper hemisphere)
   * to match Mitsuba's half-vector formulation. */
  const float guide_dot_normal = dot(guide_dir, normal);

  /* Reflect guide direction if it's in the lower hemisphere */
  const float3 guide_dir_reflected = (guide_dot_normal < 0.0f) ?
                                     guide_dir - 2.0f * guide_dot_normal * normal :
                                     guide_dir;

  /* Project onto tangent frame */
  const float offset_u = dot(tangent_u, guide_dir_reflected);
  const float offset_v = dot(tangent_v, guide_dir_reflected);

  return make_float2(offset_u, offset_v);
}

/* Ray-traced reproject for single-bounce manifold vertex.
 *
 * Validates a proposed (u,v) by ray-tracing from receiver through the
 * proposed vertex position and verifying we hit the expected surface.
 *
 * Per Mitsuba reference (manifold_path_guiding.h:reproject):
 * - Compute 3D position from potentially OUT-OF-BOUNDS barycentric coords
 *   (position in tangent plane extension, might be off-triangle)
 * - Ray-trace from start point toward this 3D position
 * - Verify intersection hits the same shape as previous iteration
 * - Return the HIT's barycentric coordinates (not the input coords)
 *
 * This differs from parameter-space clamping: Newton can propose far off-triangle,
 * and reproject finds where the ray actually hits.
 *
 * Returns: true if ray-traced path is geometrically valid, false otherwise
 * On success, updates hit_u and hit_v with the intersection's barycentric coords */
ccl_device_inline bool reproject_single_bounce(KernelGlobals kg,
                                               const ShadingPoint &receiver,
                                               int receiver_object,
                                               int receiver_prim,
                                               const SpecularSurfaceGeometry &geometry,
                                               const SpecularParameters &params,
                                               float proposed_u,
                                               float proposed_v,
                                               int expected_object,
                                               int expected_prim,
                                               float ray_time,
                                               int &hit_object,
                                               int &hit_prim,
                                               float &hit_u,
                                               float &hit_v)
{
  /* Compute proposed 3D position from UNCLAMPED barycentric coordinates.
   * This extends into the triangle's tangent plane, matching Mitsuba's approach:
   * p_prop = v.p - step_scale * beta * (v.dp_du * dx[0] + v.dp_dv * dx[1])
   * The position might be far off-triangle if Newton proposes a large step. */
  const float w = 1.0f - proposed_u - proposed_v;
  const float3 proposed_point = geometry.verts[0] * w +
                                 geometry.verts[1] * proposed_u +
                                 geometry.verts[2] * proposed_v;

if (MPG_DEBUG::NEWTON_DETAIL()) {
  printf("    reproject_single DEBUG:\n");
  printf("      proposed (u,v,w) = (%.6f, %.6f, %.6f)\n", proposed_u, proposed_v, w);
  printf("      Triangle verts: v0=(%.4f,%.4f,%.4f) v1=(%.4f,%.4f,%.4f) v2=(%.4f,%.4f,%.4f)\n",
         geometry.verts[0].x, geometry.verts[0].y, geometry.verts[0].z,
         geometry.verts[1].x, geometry.verts[1].y, geometry.verts[1].z,
         geometry.verts[2].x, geometry.verts[2].y, geometry.verts[2].z);
  printf("      Proposed 3D point: (%.6f, %.6f, %.6f)\n",
         proposed_point.x, proposed_point.y, proposed_point.z);
  printf("      Receiver position: (%.6f, %.6f, %.6f)\n",
         receiver.position.x, receiver.position.y, receiver.position.z);
}

  /* Setup ray from receiver toward proposed vertex */
  Ray ray;
  ray.P = receiver.position;
  const float3 direction = proposed_point - receiver.position;
  const float distance = len(direction);
  if (!(distance > 1e-6f)) {
    /* Proposed point too close to receiver */
    return false;
  }
  ray.D = direction / distance;
  ray.tmin = 0.0f;
  ray.tmax = FLT_MAX;  /* Don't limit - let it find whatever it hits */
  ray.time = ray_time;  /* Use shading time for motion blur consistency */
  ray.dP = differential_zero_compact();
  ray.dD = differential_zero_compact();

  /* Configure ray to skip the receiver surface - critical to avoid self-intersection */
  ray.self.object = receiver_object;
  ray.self.prim = receiver_prim;
  ray.self.light_object = OBJECT_NONE;
  ray.self.light_prim = PRIM_NONE;

  /* Ray-trace to find intersection */
  Intersection isect;
  if (!scene_intersect(kg, &ray, PATH_RAY_ALL_VISIBILITY, &isect)) {
    /* No intersection - proposed vertex not reachable */
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject FAIL: No intersection\n");
}
    return false;
  }

  /* Verify we hit the expected shape (Mitsuba's key check).
   * Per Mitsuba: compare Shape (object), NOT primitive (triangle).
   * Newton is allowed to walk across triangle boundaries on the same continuous mesh. */
  if (isect.object != expected_object) {
    /* Hit different mesh - reject this Newton step */
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject FAIL: Wrong mesh object (expected obj=%d, got obj=%d)\n",
           expected_object, isect.object);
}
    return false;
  }

  /* If we hit a different triangle on the same mesh, that's OK - Newton walked to adjacent triangle */
if (MPG_DEBUG::NEWTON_DETAIL() && isect.prim != expected_prim) {
  printf("    reproject: Walked to adjacent triangle (prim %d → %d on same mesh obj=%d)\n",
         expected_prim, isect.prim, isect.object);
}

  /* Extract hit information: object, primitive, and barycentric coordinates.
   * This is what Mitsuba does: use the ray-traced intersection point's actual coordinates.
   * If we walked to a different triangle, caller must reload geometry for the new prim. */
  hit_object = isect.object;
  hit_prim = isect.prim;
  hit_u = isect.u;
  hit_v = isect.v;

if (MPG_DEBUG::NEWTON_DETAIL()) {
  printf("    reproject SUCCESS: Hit surface at (u=%.6f, v=%.6f)\n", hit_u, hit_v);
}

  /* All checks passed - proposed vertex is geometrically valid */
  return true;
}

/* Ray-traced reproject for double-bounce manifold path.
 *
 * Per Mitsuba reference: validates proposed vertices by scatter-and-trace loop:
 * 1. Ray-trace receiver → proposed primary vertex
 * 2. Apply specular scattering (refraction/reflection) at primary vertex
 * 3. Trace scattered direction to find where it hits (determines secondary vertex)
 * 4. Verify hits are on expected objects
 *
 * This validates BOTH geometry (ray hits surface) AND physics (Snell's law, no TIR).
 * Unlike parameter-space validation, this rejects physically impossible paths during
 * reproject (before Jacobian computation).
 *
 * Returns: true if path is geometrically and physically valid
 * On success, updates hit coordinates (primary from ray-trace, secondary from scatter)
 */
ccl_device_inline bool reproject_double_bounce(KernelGlobals kg,
                                               const ShadingPoint &receiver,
                                               int receiver_object,
                                               int receiver_prim,
                                               const SpecularSurfaceGeometry &primary_geometry,
                                               SpecularParameters primary_params,  /* Pass by value so we can update on triangle walk */
                                               float proposed_primary_u,
                                               float proposed_primary_v,
                                               int expected_primary_object,
                                               int expected_primary_prim,
                                               float ray_time,
                                               int &hit_primary_object,
                                               int &hit_primary_prim,
                                               float &hit_primary_u,
                                               float &hit_primary_v,
                                               const SpecularSurfaceGeometry &secondary_geometry,
                                               SpecularParameters secondary_params,  /* Pass by value so we can update on triangle walk */
                                               float proposed_secondary_u,
                                               float proposed_secondary_v,
                                               int expected_secondary_object,
                                               int expected_secondary_prim,
                                               int &hit_secondary_object,
                                               int &hit_secondary_prim,
                                               float &hit_secondary_u,
                                               float &hit_secondary_v)
{
  /* === FIRST SEGMENT: receiver → primary vertex === */

  /* Compute proposed primary 3D position */
  const float w1 = 1.0f - proposed_primary_u - proposed_primary_v;
  const float3 proposed_primary_point = primary_geometry.verts[0] * w1 +
                                         primary_geometry.verts[1] * proposed_primary_u +
                                         primary_geometry.verts[2] * proposed_primary_v;

if (MPG_DEBUG::NEWTON_DETAIL()) {
  printf("    reproject_double DEBUG (PRIMARY):\n");
  printf("      proposed (u,v,w) = (%.6f, %.6f, %.6f)\n", proposed_primary_u, proposed_primary_v, w1);
  printf("      Triangle verts: v0=(%.4f,%.4f,%.4f) v1=(%.4f,%.4f,%.4f) v2=(%.4f,%.4f,%.4f)\n",
         primary_geometry.verts[0].x, primary_geometry.verts[0].y, primary_geometry.verts[0].z,
         primary_geometry.verts[1].x, primary_geometry.verts[1].y, primary_geometry.verts[1].z,
         primary_geometry.verts[2].x, primary_geometry.verts[2].y, primary_geometry.verts[2].z);
  printf("      Proposed 3D primary point: (%.6f, %.6f, %.6f)\n",
         proposed_primary_point.x, proposed_primary_point.y, proposed_primary_point.z);
  printf("      Receiver position: (%.6f, %.6f, %.6f)\n",
         receiver.position.x, receiver.position.y, receiver.position.z);
}

  /* Ray-trace from receiver toward proposed primary */
  Ray ray1;
  ray1.P = receiver.position;
  const float3 direction1 = proposed_primary_point - receiver.position;
  const float distance1 = len(direction1);
  if (!(distance1 > 1e-6f)) {
    return false;
  }
  ray1.D = direction1 / distance1;
  ray1.tmin = 0.0f;
  ray1.tmax = FLT_MAX;  /* Don't limit - let it find whatever it hits */
  ray1.time = ray_time;  /* Use shading time for motion blur consistency */
  ray1.dP = differential_zero_compact();
  ray1.dD = differential_zero_compact();
  /* Configure ray to skip the receiver surface - critical to avoid self-intersection */
  ray1.self.object = receiver_object;
  ray1.self.prim = receiver_prim;
  ray1.self.light_object = OBJECT_NONE;
  ray1.self.light_prim = PRIM_NONE;

  Intersection isect1;
  if (!scene_intersect(kg, &ray1, PATH_RAY_ALL_VISIBILITY, &isect1)) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double FAIL: Primary vertex not reachable\n");
}
    return false;
  }

  /* Verify we hit the expected primary shape.
   * Per Mitsuba: compare Shape (object), NOT primitive (triangle).
   * Newton is allowed to walk across triangle boundaries on the same continuous mesh. */
  if (isect1.object != expected_primary_object) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double FAIL: Wrong primary mesh (expected obj=%d, got obj=%d)\n",
           expected_primary_object, isect1.object);
}
    return false;
  }

  /* If we hit a different triangle on the same mesh, that's OK - Newton walked to adjacent triangle */
if (MPG_DEBUG::NEWTON_DETAIL() && isect1.prim != expected_primary_prim) {
  printf("    reproject_double: Primary walked to adjacent triangle (prim %d → %d on obj=%d)\n",
         expected_primary_prim, isect1.prim, isect1.object);
}

  /* Extract primary hit information: object, primitive, and barycentric coordinates.
   * If we walked to a different triangle, we must reload geometry for the new prim! */
  hit_primary_object = isect1.object;
  hit_primary_prim = isect1.prim;
  hit_primary_u = isect1.u;
  hit_primary_v = isect1.v;

  /* === SECOND SEGMENT: Apply specular scattering at primary, trace scattered ray === */

  /* Per Codex: If Newton walked to adjacent triangle, reload geometry from actual hit.
   * Using stale geometry corrupts position/normal/scattering direction! */
  SpecularSurfaceGeometry actual_primary_geometry = primary_geometry;
  if (hit_primary_prim != expected_primary_prim) {
    /* Load geometry for the NEW triangle we actually hit */
    const int object = hit_primary_object;
    const int prim = hit_primary_prim;
    const int object_flag = kernel_data_fetch(object_flag, object);

    /* Load vertices and normals (with motion blur support) */
    if (object_flag & SD_OBJECT_MOTION) {
      motion_triangle_vertices_and_normals(kg, object, prim, ray_time,
                                          actual_primary_geometry.verts,
                                          actual_primary_geometry.normals);
    }
    else {
      triangle_vertices_and_normals(kg, prim,
                                    actual_primary_geometry.verts,
                                    actual_primary_geometry.normals);
    }

    /* Apply object transforms if needed */
    if (!(object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
      ShaderData temp_sd = {};
      temp_sd.object = object;
      temp_sd.object_flag = object_flag;
      shader_setup_object_transforms(kg, &temp_sd, ray_time);

      object_position_transform_auto(kg, &temp_sd, &actual_primary_geometry.verts[0]);
      object_position_transform_auto(kg, &temp_sd, &actual_primary_geometry.verts[1]);
      object_position_transform_auto(kg, &temp_sd, &actual_primary_geometry.verts[2]);
      object_normal_transform_auto(kg, &temp_sd, &actual_primary_geometry.normals[0]);
      object_normal_transform_auto(kg, &temp_sd, &actual_primary_geometry.normals[1]);
      object_normal_transform_auto(kg, &temp_sd, &actual_primary_geometry.normals[2]);
    }

    /* Compute surface derivatives */
    actual_primary_geometry.dPdu = actual_primary_geometry.verts[1] - actual_primary_geometry.verts[0];
    actual_primary_geometry.dPdv = actual_primary_geometry.verts[2] - actual_primary_geometry.verts[0];
  }

  /* Compute actual hit primary 3D position from ACTUAL geometry */
  const float w1_hit = 1.0f - hit_primary_u - hit_primary_v;
  const float3 hit_primary_point = actual_primary_geometry.verts[0] * w1_hit +
                                    actual_primary_geometry.verts[1] * hit_primary_u +
                                    actual_primary_geometry.verts[2] * hit_primary_v;

  /* Per Codex Pass 1 #2 and Pass 4 #2: When Newton walks to adjacent triangle, update backfacing.
   * The backfacing flag is critical for entering/exiting determination and must match NEW geometry. */
  if (hit_primary_prim != expected_primary_prim) {
    /* Recompute backfacing using the NEW geometry's normal and the ray direction.
     * Per shader_setup_from_ray: backfacing = dot(Ng, ray.D) < 0 */
    const float3 actual_gn = safe_normalize(cross(actual_primary_geometry.dPdu, actual_primary_geometry.dPdv));
    const float3 ray_to_primary = safe_normalize(hit_primary_point - receiver.position);
    primary_params.backfacing = (dot(actual_gn, ray_to_primary) < 0.0f);

    /* Per Codex Pass 4 #2: Also invalidate stale BSDF normal.
     * If has_normal was true, it refers to the OLD triangle's BSDF frame.
     * Mitsuba reconstructs vertex from current intersection; we must rebuild normal. */
    if (primary_params.has_normal) {
      primary_params.has_normal = false;  /* Force recompute from actual geometry */
    }

if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double: Updated primary backfacing=%s for new triangle (prim %d → %d)\n",
           primary_params.backfacing ? "TRUE" : "FALSE", expected_primary_prim, hit_primary_prim);
}
  }

  /* Get primary vertex normal for scattering from ACTUAL geometry.
   * Use smooth shading if available, otherwise face normal. */
  float3 primary_normal;
  if (primary_params.has_normal) {
    /* Use explicit normal from params (e.g., bump mapping) */
    primary_normal = primary_params.normal;
  } else {
    /* Interpolate vertex normals (smooth shading) or use face normal from ACTUAL geometry */
    primary_normal = combine_vertex_normals(actual_primary_geometry, hit_primary_u, hit_primary_v);
    if (is_zero(primary_normal)) {
      /* Fallback to face normal from ACTUAL geometry */
      primary_normal = actual_primary_geometry.normals[0];
    }
  }
  primary_normal = safe_normalize(primary_normal);

  /* CRITICAL: Apply specular scattering (refraction/reflection) at primary vertex.
   * This is the key difference from geometric-only reproject - we validate physics!
   * Per Mitsuba: compute_specular() determines the scattered direction using Snell's law.
   * Per Codex: Mitsuba expects direction AWAY from surface, so negate ray1.D. */
  bool tir_at_primary = false;
  float cos_theta_i, cos_theta_t, eta_used;
  const float3 scattered_direction = compute_specular(
      -ray1.D,             // Direction away from primary (Mitsuba convention: wi = x_prev - x_cur)
      primary_normal,      // Primary vertex normal
      primary_params,      // Primary material (IOR, is_refraction, backfacing, etc.)
      tir_at_primary,      // Output: did total internal reflection occur?
      cos_theta_i,         // Output: incident angle cosine
      cos_theta_t,         // Output: transmitted angle cosine (0 if TIR)
      eta_used);           // Output: eta ratio used

  if (tir_at_primary) {
    /* Total internal reflection at primary vertex - path is physically invalid.
     * This is exactly what Mitsuba checks: if scattering fails (TIR), reject immediately.
     * No need to check further - this path cannot exist. */
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double FAIL: TIR at primary vertex (physics violation)\n");
}
    return false;
  }

  /* Trace the SCATTERED direction to find where it hits.
   * Per Mitsuba: the secondary vertex is determined by WHERE THE SCATTERED RAY HITS,
   * not by the Newton proposal! The proposal just suggests a direction to try. */
  Ray ray2;
  ray2.P = hit_primary_point;
  ray2.D = scattered_direction;  // Use physics-correct scattered direction!
  ray2.tmin = 1e-4f;  // Small offset to avoid self-intersection
  ray2.tmax = FLT_MAX;
  ray2.time = ray_time;  /* Use shading time for motion blur consistency */
  ray2.dP = differential_zero_compact();
  ray2.dD = differential_zero_compact();
  /* Skip primary surface to avoid self-intersection */
  ray2.self.object = hit_primary_object;  // Use actual hit, not expected
  ray2.self.prim = hit_primary_prim;
  ray2.self.light_object = OBJECT_NONE;
  ray2.self.light_prim = PRIM_NONE;

  Intersection isect2;
  if (!scene_intersect(kg, &ray2, PATH_RAY_ALL_VISIBILITY, &isect2)) {
    /* Scattered ray didn't hit anything - path is physically invalid.
     * This can happen if the scattered direction points away from the scene. */
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double FAIL: Scattered ray from primary didn't hit anything\n");
}
    return false;
  }

  /* Verify we hit the expected secondary shape.
   * Per Mitsuba: compare Shape (object), NOT primitive (triangle).
   * Newton is allowed to walk across triangle boundaries on the same continuous mesh. */
  if (isect2.object != expected_secondary_object) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double FAIL: Wrong secondary mesh (expected obj=%d, got obj=%d)\n",
           expected_secondary_object, isect2.object);
}
    return false;
  }

  /* If we hit a different triangle on the same mesh, that's OK - Newton walked to adjacent triangle */
if (MPG_DEBUG::NEWTON_DETAIL() && isect2.prim != expected_secondary_prim) {
  printf("    reproject_double: Secondary walked to adjacent triangle (prim %d → %d on obj=%d)\n",
         expected_secondary_prim, isect2.prim, isect2.object);
}

  /* Extract secondary hit information from WHERE THE SCATTERED RAY HIT.
   * CRITICAL: These are NOT the proposed secondary coordinates!
   * The secondary vertex position is determined by physics (Snell's law),
   * not by the Newton proposal. The proposal just suggests a search direction. */
  hit_secondary_object = isect2.object;
  hit_secondary_prim = isect2.prim;
  hit_secondary_u = isect2.u;  // From scattered ray hit
  hit_secondary_v = isect2.v;  // From scattered ray hit

if (MPG_DEBUG::NEWTON_DETAIL()) {
  printf("    reproject_double SUCCESS:\n");
  printf("      Primary: hit (%.6f, %.6f) on obj=%d prim=%d\n",
         hit_primary_u, hit_primary_v, hit_primary_object, hit_primary_prim);
  printf("      Secondary: scattered ray hit (%.6f, %.6f) on obj=%d prim=%d\n",
         hit_secondary_u, hit_secondary_v, hit_secondary_object, hit_secondary_prim);
  printf("      (Note: secondary coords from physics, not from proposal)\n");
}

  /* All checks passed - path is geometrically and physically valid!
   * Both ray-tracing (geometry) and scattering (physics) succeeded. */
  return true;
}

}  // namespace

float mpg_compute_segment_visibility(KernelGlobals kg,
                                     const float3 &start_point,
                                     const float3 &start_normal,
                                     const float3 &end_point,
                                     const float time,
                                     const int skip_object,
                                     const int skip_prim,
                                     const int skip_light_object,
                                     const int skip_light_prim)
{
  return compute_segment_visibility(kg,
                                    start_point,
                                    start_normal,
                                    end_point,
                                    time,
                                    skip_object,
                                    skip_prim,
                                    skip_light_object,
                                    skip_light_prim);
}

bool mpg_solve_single_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
                             const GuideSummary &guide,
                             const MpgOptions &options,
                             RNGState &rng_state,
                             MpgSolverOutput &result,
                             MpgFailureCode &failure_code)
{
  (void)rng_state;
  (void)bsdf;

  result = MpgSolverOutput();
  failure_code = MPG_FAILURE_NONE;

  if (seed.prim < 0 || seed.object < 0) {
    failure_code = MPG_FAILURE_GEOMETRY;
    return false;
  }

  SpecularSurfaceGeometry geometry;
  if (!load_surface_geometry(kg, sd, seed, geometry)) {
    failure_code = MPG_FAILURE_GEOMETRY;
    return false;
  }

  /* Per Codex Pass 1 #1: Do NOT clamp initial barycentric coordinates.
   * Mitsuba starts from actual intersection barycentrics and relies on reproject
   * to handle out-of-bounds proposals. Hard clamping shifts the initial point off
   * the true solution when valid specular is near triangle edge/vertex. */
  float u = seed.bary_u;
  float v = seed.bary_v;
  /* Ensure valid barycentric (w = 1-u-v >= 0) using project_barycentrics */
  project_barycentrics(u, v);

  SpecularParameters params;
  if (!specular_parameters_from_surface(kg, sd, geometry, seed, 0, u, v, params)) {
    failure_code = MPG_FAILURE_NO_SPECULAR;
    return false;
  }

  /* Check for glossy (rough) speculars which are not yet supported.
   * Current implementation only supports perfect speculars (alpha_x=alpha_y=0).
   * For glossy surfaces, offset normal sampling from microfacet distribution is needed. */
  if (params.has_microfacet) {
    const float alpha_x = fmaxf(params.microfacet.alpha_x, 0.0f);
    const float alpha_y = fmaxf(params.microfacet.alpha_y, 0.0f);
    const bool is_glossy = (alpha_x > 1e-6f || alpha_y > 1e-6f);
    if (is_glossy) {
if (MPG_DEBUG::PARAMS()) {
      printf("MPG DEBUG: Rejecting glossy specular (alpha_x=%.6f, alpha_y=%.6f)\n", alpha_x, alpha_y);
      printf("  Current implementation only supports perfect speculars (roughness=0)\n");
}
      failure_code = MPG_FAILURE_NO_SPECULAR;
      return false;
    }
  }

  const ShadingPoint shading_point = shading_point_from_shader_data(sd);

  SpecularEval eval;
  evaluate_specular(shading_point, seed, geometry, params, u, v, eval);
  if (eval.tir) {
    failure_code = MPG_FAILURE_TOTAL_INTERNAL_REFLECTION;
    return false;
  }

  /* Per Codex: Use BSDF frame tangents (frame.s, frame.t orthogonal to BSDF frame normal).
   * Mitsuba's half-vector constraint: C = [dot(s, h), dot(t, h)] where s,t are BSDF tangents.
   * This formulation is proven to converge to 1e-4 in Mitsuba. */
  float3 tangent_u = eval.tangent_u;
  float3 tangent_v = eval.tangent_v;
  if (is_zero(tangent_u) || is_zero(tangent_v)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: BSDF frame tangents not initialized\n");
    printf("  tangent_u=[%.6f, %.6f, %.6f] len=%.9f\n", tangent_u.x, tangent_u.y, tangent_u.z, len(tangent_u));
    printf("  tangent_v=[%.6f, %.6f, %.6f] len=%.9f\n", tangent_v.x, tangent_v.y, tangent_v.z, len(tangent_v));
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  /* Compute generalized half-vector (matching Mitsuba) */
  const float3 wi = -eval.dir_ds;
  const float3 wo = eval.dir_sl;

  /* Half-vector eta determination per Mitsuba reference.
   * CRITICAL: Must test dot(wi, geometric_normal) dynamically, not use precomputed flag.
   * CRITICAL: Align geometric normal to BSDF frame before testing (Mitsuba does this).
   * Mitsuba tests dot(wi, v[i].gn) to determine entering/exiting at constraint evaluation time. */
  float h_eta = 1.0f;
  if (params.is_refraction) {
    const float3 geometric_normal_raw = safe_normalize(cross(geometry.dPdu, geometry.dPdv));

    /* Align to BSDF frame (Mitsuba: masked(gn, dot(n, gn) < 0) *= -1) */
    float3 geometric_normal = geometric_normal_raw;
    if (dot(eval.normal, geometric_normal_raw) < 0.0f) {
      geometric_normal = -geometric_normal_raw;
    }

    const float dot_wi_gn = dot(wi, geometric_normal);
    const float base_eta = params.base_eta;

    if (dot_wi_gn < 0.0f) {
      /* Exiting (coming from inside): eta = 1/base_eta */
      h_eta = 1.0f / fmaxf(base_eta, 1e-6f);
    }
    else {
      /* Entering (arriving from outside): eta = base_eta */
      h_eta = base_eta;
    }
  }

  /* Generalized half-vector: h = normalize(wi + eta * wo), negated for refraction.
   * Mitsuba normalizes without checking length threshold, trusting that
   * geometrically invalid configurations will fail naturally in the Newton solver.
   * We check for zero-length to avoid NaN, but use a minimal tolerance that only
   * catches truly degenerate cases (matching Mitsuba's approach). */
  float3 h = wi + h_eta * wo;
  if (params.is_refraction) {
    h = -h;
  }
  const float h_len = len(h);
  if (h_len > 0.0f) {
    h /= h_len;
  }
  else {
    /* Half-vector is exactly zero - this is truly degenerate.
     * Set to normal as fallback (constraint will fail to converge). */
    h = eval.normal;
  }

  /* Verify normalization produced valid result */
  if (!isfinite_safe(h.x) || !isfinite_safe(h.y) || !isfinite_safe(h.z)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: Degenerate normals at half-vector normalization (NaN/Inf)\n");
    printf("  h=[%.6f, %.6f, %.6f] h_len=%.9f\n", h.x, h.y, h.z, h_len);
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  /* Mitsuba's 2D constraint: C = H - N
   * where H is the projected half-vector and N is the offset normal.
   *
   * Per Codex Pass 2 #1: Apply guided offset normal when path guiding provides direction.
   * Without this, Newton pursues perfect specular solution even when guide suggests
   * a different target, causing convergence failure on guided paths.
   *
   * For glossy (roughness>0): N sampled from microfacet distribution (TODO) */
  const float2 offset_2d = compute_guide_offset_normal(guide, tangent_u, tangent_v, eval.normal);

  /* Project half-vector onto tangent plane to get 2D constraint.
   * This removes any normal component that might arise from numerical error. */
  const float h_normal_component = dot(eval.normal, h);
  const float3 h_tangent = h - eval.normal * h_normal_component;

  float residual_2d_u = dot(tangent_u, h_tangent) - offset_2d.x;
  float residual_2d_v = dot(tangent_v, h_tangent) - offset_2d.y;
  float residual_norm = sqrtf(residual_2d_u * residual_2d_u + residual_2d_v * residual_2d_v);

  if (!isfinite_safe(residual_norm)) {
    failure_code = MPG_FAILURE_NEWTON_DIVERGED;
    return false;
  }

  /* Mitsuba-style damped Newton: reuse Jacobian when step is rejected */
  float beta = 1.0f;  /* Step size damping factor */
  bool needs_step_update = true;
  float2 delta = make_float2(0.0f, 0.0f);

  /* Track current object/prim/geometry - may change as Newton walks across triangles.
   * Per Mitsuba: allow walking across triangle boundaries on continuous surfaces. */
  int current_object = seed.object;
  int current_prim = seed.prim;
  SpecularSurfaceGeometry current_geometry = geometry;

  for (int iter = 0; iter < options.max_iters; ++iter) {
    /* Now using Mitsuba's half-vector constraint formulation directly.
     * Should converge to 1e-4 like Mitsuba (threshold from their reference). */
    if (residual_norm < 1e-4f) {
      break;
    }

    /* Only recompute Jacobian and solve when needed (avoid redundant computation) */
    if (needs_step_update) {
      float3 J_cols[2];
      compute_halfvector_jacobian(shading_point, seed, current_geometry, eval, h, h_eta, tangent_u, tangent_v, J_cols);

      /* Embed 2D constraint in 3D vector */
      const float3 residual_3d = make_float3(residual_2d_u, residual_2d_v, 0.0f);

      /* Per Codex Pass 1 #1: Use direct 2×2 solve instead of normal equations.
       * Mitsuba solves J * dx = C directly, avoiding JᵀJ which squares condition number. */
      if (!solve_direct_2x2(J_cols, residual_3d, delta) ||
          !isfinite_safe(delta.x) ||
          !isfinite_safe(delta.y))
      {
        if (failure_code == MPG_FAILURE_NONE) {
          failure_code = MPG_FAILURE_JACOBIAN_ZERO;
        }
        break;
      }
    }

    /* Apply step with step_scale * beta scaling (Mitsuba approach).
     * Matches Mitsuba: p_prop = p - step_scale * beta * (dp_du * dx[0] + dp_dv * dx[1])
     * DO NOT clamp to [0,1] - let reproject handle out-of-bounds proposals. */
    float proposed_u = u - options.step_scale * beta * delta.x;
    float proposed_v = v - options.step_scale * beta * delta.y;

    /* Per Mitsuba: ray-trace to verify proposed vertex is geometrically reachable.
     * Reproject computes 3D position from potentially out-of-bounds barycentric coords,
     * ray-traces to find what surface is hit, and returns the hit's barycentric coords.
     * This is the key difference from parameter-space clamping. */
    int hit_object, hit_prim;
    float new_u, new_v;
    if (!reproject_single_bounce(kg, shading_point, sd.object, sd.prim,
                                  current_geometry, params, proposed_u, proposed_v,
                                  current_object, current_prim, sd.time,
                                  hit_object, hit_prim, new_u, new_v)) {
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    /* If Newton walked to a different triangle, reload geometry for the new prim.
     * Per Mitsuba: allow walking across triangle boundaries on continuous surfaces. */
    SpecularSurfaceGeometry new_geometry = current_geometry;
    MpgSeedRay current_seed = seed;  /* Track current seed (may update if triangle walk) */
    if (hit_prim != current_prim) {
      /* Per Codex: Update seed to reflect actual hit triangle before evaluating parameters.
       * specular_parameters_from_surface() uses seed.prim/object to build intersection,
       * so using stale seed causes BSDF evaluation on wrong triangle. */
      current_seed.object = hit_object;
      current_seed.prim = hit_prim;

      if (!load_surface_geometry(kg, sd, current_seed, new_geometry)) {
        beta *= 0.5f;
        needs_step_update = false;
        continue;
      }
    }

    /* Use the reprojected barycentric coordinates for subsequent evaluation.
     * Per Codex: Use current_seed (updated if triangle walk) instead of original seed. */
    SpecularParameters new_params;
    if (!specular_parameters_from_surface(kg, sd, new_geometry, current_seed, 0, new_u, new_v, new_params)) {
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    SpecularEval new_eval;
    evaluate_specular(shading_point, current_seed, new_geometry, new_params, new_u, new_v, new_eval);
    if (new_eval.tir) {
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    /* Per Codex: Use BSDF frame tangents from new_eval */
    const float3 new_tangent_u = new_eval.tangent_u;
    const float3 new_tangent_v = new_eval.tangent_v;
    if (is_zero(new_tangent_u) || is_zero(new_tangent_v)) {
      beta *= 0.5f;
      needs_step_update = false;
      continue;
    }

    const float3 new_wi = -new_eval.dir_ds;
    const float3 new_wo = new_eval.dir_sl;
    /* Compute half-vector eta dynamically using dot(wi, aligned_gn) per Mitsuba reference.
     * CRITICAL: Align geometric normal to BSDF frame before testing! */
    float new_h_eta = 1.0f;
    if (new_params.is_refraction) {
      const float3 new_geometric_normal_raw = safe_normalize(cross(new_geometry.dPdu, new_geometry.dPdv));

      /* Align to BSDF frame (Mitsuba: masked(gn, dot(n, gn) < 0) *= -1) */
      float3 new_geometric_normal = new_geometric_normal_raw;
      if (dot(new_eval.normal, new_geometric_normal_raw) < 0.0f) {
        new_geometric_normal = -new_geometric_normal_raw;
      }

      const float new_dot_wi_gn = dot(new_wi, new_geometric_normal);
      const float new_base_eta = new_params.base_eta;

      if (new_dot_wi_gn < 0.0f) {
        /* Exiting (coming from inside): eta = 1/base_eta */
        new_h_eta = 1.0f / fmaxf(new_base_eta, 1e-6f);
      }
      else {
        /* Entering (arriving from outside): eta = base_eta */
        new_h_eta = new_base_eta;
      }
    }

    /* Generalized half-vector: h = normalize(wi + eta * wo), negated for refraction */
    float3 new_h = new_wi + new_h_eta * new_wo;
    if (new_params.is_refraction) {
      new_h = -new_h;
    }
    const float new_h_len = len(new_h);
    if (new_h_len > 0.0f) {
      new_h /= new_h_len;
    }
    else {
      /* Degenerate configuration - use normal as fallback and reduce step size */
      new_h = new_eval.normal;
    }

    /* Check for NaN/Inf after normalization */
    if (!isfinite_safe(new_h.x) || !isfinite_safe(new_h.y) || !isfinite_safe(new_h.z)) {
      beta *= 0.5f;
      needs_step_update = false;
      continue;
    }

    /* Per Codex Pass 2 #1: Apply guided offset normal */
    const float2 new_offset_2d = compute_guide_offset_normal(guide, new_tangent_u, new_tangent_v, new_eval.normal);

    /* Project half-vector onto tangent plane before computing constraint */
    const float new_h_normal_component = dot(new_eval.normal, new_h);
    const float3 new_h_tangent = new_h - new_eval.normal * new_h_normal_component;

    /* Apply Mitsuba's constraint: C = H - N */
    const float new_residual_2d_u = dot(new_tangent_u, new_h_tangent) - new_offset_2d.x;
    const float new_residual_2d_v = dot(new_tangent_v, new_h_tangent) - new_offset_2d.y;
    const float new_residual_norm = sqrtf(new_residual_2d_u * new_residual_2d_u +
                                          new_residual_2d_v * new_residual_2d_v);

    if (!isfinite_safe(new_residual_norm)) {
      failure_code = MPG_FAILURE_NEWTON_DIVERGED;
      return false;
    }

    /* Step acceptance: Mitsuba approach - accept if reprojection succeeded.
     * Per Mitsuba reference, acceptance is based on geometric validity (successful
     * reprojection), not residual improvement. The solver trusts Newton's method to
     * converge through potentially non-monotonic residuals.
     *
     * beta *= 0.5 on geometric failure (already handled above via continue)
     * beta = min(1.0, 2.0 * beta) on success (here) */
    u = new_u;
    v = new_v;
    eval = new_eval;
    params = new_params;
    tangent_u = new_tangent_u;
    tangent_v = new_tangent_v;
    h = new_h;
    h_eta = new_h_eta;
    residual_2d_u = new_residual_2d_u;
    residual_2d_v = new_residual_2d_v;
    residual_norm = new_residual_norm;
    /* If Newton walked to a different triangle, update tracking variables permanently */
    current_object = hit_object;
    current_prim = hit_prim;
    current_geometry = new_geometry;
    beta = fminf(beta * 2.0f, 1.0f);
    needs_step_update = true;
  }

  /* Now using Mitsuba's half-vector constraint formulation directly.
   * Accept threshold of 1e-4 matching Mitsuba reference implementation. */
  if (!isfinite_safe(residual_norm) || residual_norm > 1e-4f) {
    if (failure_code == MPG_FAILURE_NONE) {
      failure_code = MPG_FAILURE_NEWTON_DIVERGED;
    }
    return false;
  }

  result.success = true;
  result.specular_vertex_count = 1;

  /* Per Codex: Build final seed reflecting converged triangle (may differ from initial seed) */
  MpgSeedRay final_seed = seed;
  final_seed.object = current_object;
  final_seed.prim = current_prim;

  result.visibility = compute_visibility(kg, sd, final_seed, eval);
  result.wi = eval.dir_ds;

  MpgSpecularVertex &vertex = result.specular_vertices[0];
  vertex.position = eval.point;
  vertex.normal = eval.normal;
  /* Store the incoming direction pointing toward the previous vertex. The
   * solver keeps `eval.dir_ds` as the ray direction leaving the receiver, so
   * flip it here to match the convention used elsewhere (see the double bounce
   * solver below). */
  vertex.dir_in = -eval.dir_ds;
  vertex.dir_out = eval.dir_sl;
  vertex.distance_in = eval.distance_ds;
  vertex.distance_out = eval.distance_sl;
  vertex.eta = eval.eta;
  vertex.cos_theta_in = eval.cos_theta_i;
  vertex.cos_theta_out = eval.cos_theta_t;
  vertex.dXdu = eval.dXdu;
  vertex.dXdv = eval.dXdv;
  vertex.dNdu = eval.dNdu;
  vertex.dNdv = eval.dNdv;
  vertex.u = u;
  vertex.v = v;
  vertex.is_refraction = params.is_refraction;
  vertex.total_internal_reflection = eval.tir;
  /* Per Codex: Store actual converged object/prim, not original seed */
  vertex.object = current_object;
  vertex.prim = current_prim;

  result.dir_ds = -vertex.dir_in;
  result.dir_sl = vertex.dir_out;
  result.distance_ds = vertex.distance_in;
  result.distance_sl = vertex.distance_out;
  result.specular_point = vertex.position;
  result.specular_normal = vertex.normal;
  result.dXdu = vertex.dXdu;
  result.dXdv = vertex.dXdv;
  result.dNdu = vertex.dNdu;
  result.dNdv = vertex.dNdv;
  result.u = vertex.u;
  result.v = vertex.v;
  result.object = vertex.object;
  result.prim = vertex.prim;
  result.is_refraction = vertex.is_refraction;

  /* Evaluate the specular weight using the receiver -> specular direction so the acceptance
   * checks align with the Mitsuba reference while `result.dir_ds`/`result.wi` remain the forward
   * ray from the receiver.
   */
  result.spec_weight = evaluate_specular_weight(
      kg, params, result.dir_ds, result.dir_sl, vertex.cos_theta_in, vertex.cos_theta_out);
  if (is_zero(result.spec_weight)) {
    failure_code = MPG_FAILURE_ZERO_THROUGHPUT;
    return false;
  }
  vertex.throughput = result.spec_weight;
  result.specular_throughput = result.spec_weight;

  /* Per Codex: Use converged geometry/seed for residual matrix, not original */
  float residual_matrix[2][2];
  if (!compute_residual_matrix(shading_point, final_seed, current_geometry, eval, residual_matrix)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: Degenerate normals at compute_residual_matrix\n");
    printf("  dir_sl=[%.6f, %.6f, %.6f]\n", eval.dir_sl.x, eval.dir_sl.y, eval.dir_sl.z);
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  const float determinant =
      residual_matrix[0][0] * residual_matrix[1][1] - residual_matrix[0][1] * residual_matrix[1][0];
  if (!isfinite_safe(determinant) || fabsf(determinant) <= 1.0e-12f) {
    failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    return false;
  }

  const float area_element = len(cross(eval.dXdu, eval.dXdv));
  const float cos_theta = fabsf(dot(eval.normal, result.wi));
  /* Use relaxed thresholds matching reference implementation solver_threshold (1e-4).
   * Original 1e-10 was 10000x stricter than reference's 1e-4. */
  if (area_element < 1e-4f || cos_theta < 1e-4f) {
if (MPG_DEBUG::NEWTON()) {
    printf("MPG DEBUG single-bounce: FAILED geometry check, area=%.9f cos_theta=%.9f\n",
           area_element, cos_theta);
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  /* Compute the complete Jacobian including the geometric term.
   * The Jacobian transforms from the specular surface area measure to the receiver's
   * solid angle measure. Following the Mitsuba reference, this includes:
   * - The manifold constraint Jacobian (determinant of residual matrix)
   * - The geometric factor: cos(θ) / r²
   * where θ is the angle at the specular surface and r is the distance from receiver.
   *
   * Balance numerical stability with geometric accuracy. Very small distances produce
   * extremely large Jacobians that can cause overflow. Use conservative threshold. */
  const float distance_sq = eval.distance_ds * eval.distance_ds;
  const float min_distance_sq = 1e-8f;  /* Max Jacobian contribution: ~1e8 */
  if (distance_sq < min_distance_sq) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  const float geometric_factor = cos_theta / distance_sq;
  const float jacobian = fabsf(determinant) * geometric_factor;

  /* Check for numerical issues in Jacobian computation */
  if (!isfinite_safe(jacobian) || jacobian <= 1.0e-12f) {
    failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    return false;
  }

  result.jacobian_total = jacobian;
  result.jacobian = jacobian;
  vertex.jacobian = jacobian;

  failure_code = MPG_FAILURE_NONE;

  return true;
}

bool mpg_solve_double_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
                             const GuideSummary &guide,
                             const MpgOptions &options,
                             RNGState &rng_state,
                             MpgSolverOutput &result,
                             MpgFailureCode &failure_code)
{
  (void)rng_state;
  (void)bsdf;

  result = MpgSolverOutput();
  failure_code = MPG_FAILURE_NONE;

if (MPG_DEBUG::PARAMS()) {
  printf("\n");
  printf("========================================\n");
  printf("MPG DOUBLE-BOUNCE PATH TRACE\n");
  printf("========================================\n");
  printf("RECEIVER (where camera hit):\n");
  printf("  Position: (%.6f, %.6f, %.6f)\n", sd.P.x, sd.P.y, sd.P.z);
  printf("  Normal: (%.6f, %.6f, %.6f)\n", sd.N.x, sd.N.y, sd.N.z);
  printf("  Object: %d\n", sd.object);
  printf("\n");
  printf("PRIMARY VERTEX (first specular bounce):\n");
  printf("  Seed object: %d, prim: %d\n", seed.object, seed.prim);
  printf("  Seed bary: (%.6f, %.6f)\n", seed.bary_u, seed.bary_v);
  printf("\n");
}

  if (seed.prim < 0 || seed.object < 0) {
    failure_code = MPG_FAILURE_GEOMETRY;
    return false;
  }

  SpecularSurfaceGeometry primary_geometry;
  if (!load_surface_geometry(kg, sd, seed, primary_geometry)) {
    failure_code = MPG_FAILURE_GEOMETRY;
    return false;
  }

  /* Per Codex Pass 1 #1: Do NOT clamp initial barycentric coordinates */
  float primary_u = seed.bary_u;
  float primary_v = seed.bary_v;
  project_barycentrics(primary_u, primary_v);

  SpecularParameters primary_params;
  if (!specular_parameters_from_surface(kg, sd, primary_geometry, seed, 0, primary_u, primary_v, primary_params)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: specular_parameters_from_surface failed for primary (line 2268)\n");
}
    failure_code = MPG_FAILURE_NO_SPECULAR;
    return false;
  }

  /* Check for glossy speculars - not supported yet (same as single-bounce) */
  if (primary_params.has_microfacet) {
    const float alpha_x = fmaxf(primary_params.microfacet.alpha_x, 0.0f);
    const float alpha_y = fmaxf(primary_params.microfacet.alpha_y, 0.0f);
    if (alpha_x > 1e-6f || alpha_y > 1e-6f) {
      failure_code = MPG_FAILURE_NO_SPECULAR;
      return false;
    }
  }

  MpgSeedRay secondary_seed;
  if (!trace_secondary_seed(
          kg, sd, primary_geometry, primary_u, primary_v, seed, primary_params, secondary_seed))
  {
    const bool refractive_seed = (seed.scatter == MPG_SEED_SCATTER_REFRACTION);
    const bool expects_double_bounce = (seed.bounce_count == 2);
    if (refractive_seed && expects_double_bounce) {
      MpgSeedRay single_bounce_seed = seed;
      single_bounce_seed.bounce_count = 1;
      if (single_bounce_seed.tau_count > 0) {
        single_bounce_seed.tau_bits &= 1u;
        single_bounce_seed.tau_count = 1;
      }

if (MPG_DEBUG::PARAMS()) {
      if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
        LOG_DEBUG
            << "MPG double-bounce refraction seed missing exit surface, retrying as single bounce";
      }
}

      MpgSolverOutput single_result;
      MpgFailureCode single_failure = MPG_FAILURE_NONE;
      if (mpg_solve_single_bounce(
              kg, sd, bsdf, single_bounce_seed, guide, options, rng_state, single_result, single_failure))
      {
        result = single_result;
        failure_code = MPG_FAILURE_NONE;
        return true;
      }

      if (single_failure != MPG_FAILURE_NONE) {
        failure_code = single_failure;
        return false;
      }
    }

    failure_code = MPG_FAILURE_SEED;
    return false;
  }

  SpecularSurfaceGeometry secondary_geometry;
  if (!load_surface_geometry(kg, sd, secondary_seed, secondary_geometry)) {
    failure_code = MPG_FAILURE_GEOMETRY;
    return false;
  }

  /* Per Codex Pass 1 #1: Do NOT clamp initial barycentric coordinates */
  float secondary_u = secondary_seed.bary_u;
  float secondary_v = secondary_seed.bary_v;
  project_barycentrics(secondary_u, secondary_v);

  ShaderData primary_sd;
  if (!build_primary_shading_data(sd, primary_geometry, seed, primary_u, primary_v, primary_sd)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: build_primary_shading_data failed (line 2325)\n");
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  SpecularParameters secondary_params;
  if (!specular_parameters_from_surface(
          kg, primary_sd, secondary_geometry, secondary_seed, 1, secondary_u, secondary_v, secondary_params))
  {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: specular_parameters_from_surface failed for secondary (line 2333)\n");
}
    failure_code = MPG_FAILURE_NO_SPECULAR;
    return false;
  }

  /* Check for glossy speculars in secondary vertex */
  if (secondary_params.has_microfacet) {
    const float alpha_x = fmaxf(secondary_params.microfacet.alpha_x, 0.0f);
    const float alpha_y = fmaxf(secondary_params.microfacet.alpha_y, 0.0f);
    if (alpha_x > 1e-6f || alpha_y > 1e-6f) {
      failure_code = MPG_FAILURE_NO_SPECULAR;
      return false;
    }
  }

  ShadingPoint receiver = shading_point_from_shader_data(sd);

  DoubleBounceEval eval;
  if (!evaluate_double_bounce(receiver,
                              primary_geometry,
                              primary_params,
                              secondary_geometry,
                              secondary_params,
                              seed.light_sample,
                              primary_u,
                              primary_v,
                              secondary_u,
                              secondary_v,
                              seed.use_smooth_normals,
                              secondary_seed.use_smooth_normals,
                              eval))
  {
    failure_code = (eval.primary.tir || eval.secondary.tir) ? MPG_FAILURE_TOTAL_INTERNAL_REFLECTION :
                                                                 MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  /* Per Mitsuba reference: check convergence per-vertex, not aggregated.
   * Primary vertex: residual[0], residual[1]
   * Secondary vertex: residual[2], residual[3] */
  float primary_residual_norm = sqrtf(eval.residual[0] * eval.residual[0] +
                                       eval.residual[1] * eval.residual[1]);
  float secondary_residual_norm = sqrtf(eval.residual[2] * eval.residual[2] +
                                         eval.residual[3] * eval.residual[3]);

  if (!isfinite_safe(primary_residual_norm) || !isfinite_safe(secondary_residual_norm)) {
    failure_code = MPG_FAILURE_NEWTON_DIVERGED;
    return false;
  }

if (MPG_DEBUG::NEWTON()) {
  printf("----------------------------------------\n");
  printf("NEWTON DOUBLE-BOUNCE: Starting iterations (sample %d)\n", g_current_sample);
  printf("  Initial primary residual: %.9e\n", primary_residual_norm);
  printf("  Initial secondary residual: %.9e\n", secondary_residual_norm);
  printf("  Convergence threshold: 1e-4 (per vertex)\n");
  printf("  Max iterations: %d\n", options.max_iters);
  printf("  Primary u,v: (%.6f, %.6f)\n", primary_u, primary_v);
  printf("  Secondary u,v: (%.6f, %.6f)\n", secondary_u, secondary_v);
  printf("----------------------------------------\n");
}

  /* Mitsuba-style damped Newton: reuse Jacobian when step is rejected */
  float beta = 1.0f;  /* Step size damping factor */
  bool needs_step_update = true;
  float delta[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  /* Track current object/prim/geometry/seed - may change as Newton walks across triangles.
   * Per Mitsuba: allow walking across triangle boundaries on continuous surfaces.
   * Per Codex Pass 4 #1: Track seeds so Jacobian uses correct prim/object. */
  int current_primary_object = seed.object;
  int current_primary_prim = seed.prim;
  SpecularSurfaceGeometry current_primary_geometry = primary_geometry;
  MpgSeedRay current_primary_seed = seed;
  int current_secondary_object = secondary_seed.object;
  int current_secondary_prim = secondary_seed.prim;
  SpecularSurfaceGeometry current_secondary_geometry = secondary_geometry;
  MpgSeedRay current_secondary_seed = secondary_seed;

  for (int iter = 0; iter < options.max_iters; ++iter) {
    /* Per Mitsuba: check convergence per-vertex. Both vertices must satisfy threshold. */
    const bool primary_converged = (primary_residual_norm < 1e-4f);
    const bool secondary_converged = (secondary_residual_norm < 1e-4f);
    if (primary_converged && secondary_converged) {
      break;
    }

    /* Only recompute Jacobian and solve when needed (avoid redundant computation) */
    if (needs_step_update) {
      float J[4][4];
      /* Per Codex Pass 4 #1: Use current seeds (updated if triangle walk) for Jacobian */
      if (!compute_double_bounce_jacobian(kg,
                                          sd,
                                          current_primary_seed,
                                          current_secondary_seed,
                                          receiver,
                                          current_primary_geometry,
                                          current_secondary_geometry,
                                          seed.light_sample,
                                          primary_u,
                                          primary_v,
                                          secondary_u,
                                          secondary_v,
                                          seed.use_smooth_normals,
                                          secondary_seed.use_smooth_normals,
                                          eval.residual,
                                          J))
      {
        /* Jacobian computation failed */
        break;
      }

      if (!solve_linear_system_4x4(J, eval.residual, delta)) {
        failure_code = MPG_FAILURE_JACOBIAN_ZERO;
        return false;
      }

if (MPG_DEBUG::NEWTON_DETAIL()) {
      if (iter == 0) {
        printf("    Jacobian matrix J:\n");
        printf("      [%8.4f %8.4f %8.4f %8.4f]\n", J[0][0], J[0][1], J[0][2], J[0][3]);
        printf("      [%8.4f %8.4f %8.4f %8.4f]\n", J[1][0], J[1][1], J[1][2], J[1][3]);
        printf("      [%8.4f %8.4f %8.4f %8.4f]\n", J[2][0], J[2][1], J[2][2], J[2][3]);
        printf("      [%8.4f %8.4f %8.4f %8.4f]\n", J[3][0], J[3][1], J[3][2], J[3][3]);
        printf("    Residual: [%8.4f %8.4f %8.4f %8.4f]\n",
               eval.residual[0], eval.residual[1], eval.residual[2], eval.residual[3]);
      }
}
    }

    /* Apply step with step_scale * beta scaling (Mitsuba approach).
     * Matches Mitsuba: p_prop = p - step_scale * beta * (dp_du * dx[0] + dp_dv * dx[1])
     * DO NOT clamp to [0,1] - let reproject handle out-of-bounds proposals. */
    float proposed_primary_u = primary_u - options.step_scale * beta * delta[0];
    float proposed_primary_v = primary_v - options.step_scale * beta * delta[1];
    float proposed_secondary_u = secondary_u - options.step_scale * beta * delta[2];
    float proposed_secondary_v = secondary_v - options.step_scale * beta * delta[3];

if (MPG_DEBUG::NEWTON_DETAIL()) {
    if ((iter + 1) % 5 == 0 || iter == 0) {
      printf("    Step %2d: delta=(%.4f,%.4f,%.4f,%.4f) → proposed prim(%.4f,%.4f) sec(%.4f,%.4f)\n",
             iter + 1, delta[0], delta[1], delta[2], delta[3],
             proposed_primary_u, proposed_primary_v, proposed_secondary_u, proposed_secondary_v);
    }
}

    /* Per Mitsuba: ray-trace to verify proposed path is geometrically reachable.
     * Reproject computes 3D positions from potentially out-of-bounds barycentric coords,
     * ray-traces the full path, and returns the hits' barycentric coords.
     * This validates the entire chain: receiver → primary → secondary */
    int hit_primary_object, hit_primary_prim;
    int hit_secondary_object, hit_secondary_prim;
    float new_primary_u, new_primary_v, new_secondary_u, new_secondary_v;
    if (!reproject_double_bounce(kg, receiver, sd.object, sd.prim,
                                  current_primary_geometry, primary_params, proposed_primary_u, proposed_primary_v,
                                  current_primary_object, current_primary_prim, sd.time,
                                  hit_primary_object, hit_primary_prim, new_primary_u, new_primary_v,
                                  current_secondary_geometry, secondary_params, proposed_secondary_u, proposed_secondary_v,
                                  current_secondary_object, current_secondary_prim,
                                  hit_secondary_object, hit_secondary_prim, new_secondary_u, new_secondary_v)) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
      printf("  Iter %2d: REJECT #0 - reproject failed, beta %.6f→%.6f\n", iter + 1, beta, beta * 0.5f);
}
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    /* If Newton walked to different triangles, reload geometry for the new prims.
     * Per Mitsuba: allow walking across triangle boundaries on continuous surfaces.
     * Per Codex Pass 4 #1: Update tracking seeds to reflect actual hit triangles. */
    SpecularSurfaceGeometry new_primary_geometry = current_primary_geometry;
    MpgSeedRay new_primary_seed = current_primary_seed;
    if (hit_primary_prim != current_primary_prim) {
      new_primary_seed.object = hit_primary_object;
      new_primary_seed.prim = hit_primary_prim;
      if (!load_surface_geometry(kg, sd, new_primary_seed, new_primary_geometry)) {
        beta *= 0.5f;
        needs_step_update = false;
        continue;
      }
    }

    SpecularSurfaceGeometry new_secondary_geometry = current_secondary_geometry;
    MpgSeedRay new_secondary_seed = current_secondary_seed;
    if (hit_secondary_prim != current_secondary_prim) {
      new_secondary_seed.object = hit_secondary_object;
      new_secondary_seed.prim = hit_secondary_prim;
      if (!load_surface_geometry(kg, sd, new_secondary_seed, new_secondary_geometry)) {
        beta *= 0.5f;
        needs_step_update = false;
        continue;
      }
    }

    /* Use the reprojected barycentric coordinates for subsequent evaluation.
     * Per Codex Pass 4 #1: Use new seeds (updated if triangle walk) instead of original. */
    SpecularParameters new_primary_params;
    if (!specular_parameters_from_surface(
            kg, sd, new_primary_geometry, new_primary_seed, 0, new_primary_u, new_primary_v, new_primary_params))
    {
if (MPG_DEBUG::NEWTON_DETAIL()) {
      printf("  Iter %2d: REJECT #1 - primary params extraction failed, beta %.6f→%.6f\n", iter + 1, beta, beta * 0.5f);
}
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    ShaderData new_primary_sd;
    if (!build_primary_shading_data(sd, new_primary_geometry, new_primary_seed, new_primary_u, new_primary_v, new_primary_sd)) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
      printf("  Iter %2d: REJECT #2 - primary shading data build failed, beta %.6f→%.6f\n", iter + 1, beta, beta * 0.5f);
}
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    SpecularParameters new_secondary_params;
    if (!specular_parameters_from_surface(kg,
                                          new_primary_sd,
                                          new_secondary_geometry,
                                          new_secondary_seed,
                                          1,
                                          new_secondary_u,
                                          new_secondary_v,
                                          new_secondary_params))
    {
if (MPG_DEBUG::NEWTON_DETAIL()) {
      printf("  Iter %2d: REJECT #3 - secondary params extraction failed, beta %.6f→%.6f\n", iter + 1, beta, beta * 0.5f);
}
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    DoubleBounceEval new_eval;
    if (!evaluate_double_bounce(receiver,
                                new_primary_geometry,
                                new_primary_params,
                                new_secondary_geometry,
                                new_secondary_params,
                                seed.light_sample,
                                new_primary_u,
                                new_primary_v,
                                new_secondary_u,
                                new_secondary_v,
                                seed.use_smooth_normals,
                                secondary_seed.use_smooth_normals,
                                new_eval))
    {
if (MPG_DEBUG::NEWTON_DETAIL()) {
      printf("  Iter %2d: REJECT #4 - evaluate_double_bounce failed, beta %.6f→%.6f\n", iter + 1, beta, beta * 0.5f);
}
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    /* Compute per-vertex residual norms (Mitsuba approach) */
    const float new_primary_residual_norm = sqrtf(new_eval.residual[0] * new_eval.residual[0] +
                                                    new_eval.residual[1] * new_eval.residual[1]);
    const float new_secondary_residual_norm = sqrtf(new_eval.residual[2] * new_eval.residual[2] +
                                                      new_eval.residual[3] * new_eval.residual[3]);
    if (!isfinite_safe(new_primary_residual_norm) || !isfinite_safe(new_secondary_residual_norm)) {
      failure_code = MPG_FAILURE_NEWTON_DIVERGED;
      return false;
    }

    /* Step acceptance: Mitsuba approach - accept if reprojection succeeded.
     * Per Mitsuba reference, acceptance is based on geometric validity (successful
     * reprojection of both primary and secondary vertices), not residual improvement.
     * The solver trusts Newton's method to converge through potentially non-monotonic
     * residuals.
     *
     * beta *= 0.5 on geometric failure (already handled above via continue)
     * beta = min(1.0, 2.0 * beta) on success (here) */
    primary_u = new_primary_u;
    primary_v = new_primary_v;
    secondary_u = new_secondary_u;
    secondary_v = new_secondary_v;
    eval = new_eval;
    primary_params = new_primary_params;
    secondary_params = new_secondary_params;
    primary_sd = new_primary_sd;
    primary_residual_norm = new_primary_residual_norm;
    secondary_residual_norm = new_secondary_residual_norm;
    /* If Newton walked to different triangles, update tracking variables permanently.
     * Per Codex Pass 4 #1: Also update tracking seeds for Jacobian. */
    current_primary_object = hit_primary_object;
    current_primary_prim = hit_primary_prim;
    current_primary_geometry = new_primary_geometry;
    current_primary_seed = new_primary_seed;
    current_secondary_object = hit_secondary_object;
    current_secondary_prim = hit_secondary_prim;
    current_secondary_geometry = new_secondary_geometry;
    current_secondary_seed = new_secondary_seed;
    beta = fminf(beta * 2.0f, 1.0f);
    needs_step_update = true;

if (MPG_DEBUG::PARAMS()) {
    const float max_residual = fmaxf(primary_residual_norm, secondary_residual_norm);
    if ((iter + 1) % 5 == 0 || iter == 0 || max_residual < 1e-4f) {
      printf("  Iter %2d: prim_res=%.6e, sec_res=%.6e, prim(%.4f,%.4f) sec(%.4f,%.4f)\n",
             iter + 1, primary_residual_norm, secondary_residual_norm,
             primary_u, primary_v, secondary_u, secondary_v);
    }
}
  }

  /* Per Mitsuba: accept only if BOTH vertices converged to threshold.
   * Each vertex's 2D constraint must satisfy norm(C) <= 1e-4. */
  const bool primary_converged = (primary_residual_norm <= 1e-4f);
  const bool secondary_converged = (secondary_residual_norm <= 1e-4f);
  if (!primary_converged || !secondary_converged) {
if (MPG_DEBUG::NEWTON()) {
    printf("========================================\n");
    printf("MPG DOUBLE-BOUNCE: NEWTON FAILED TO CONVERGE\n");
    printf("========================================\n");
    printf("  Primary residual: %.9e (threshold: 1e-4) %s\n",
           primary_residual_norm, primary_converged ? "[CONVERGED]" : "[FAILED]");
    printf("  Secondary residual: %.9e (threshold: 1e-4) %s\n",
           secondary_residual_norm, secondary_converged ? "[CONVERGED]" : "[FAILED]");
    printf("  Max iterations reached: 30\n");
    printf("  Primary u,v: (%.6f, %.6f)\n", primary_u, primary_v);
    printf("  Secondary u,v: (%.6f, %.6f)\n", secondary_u, secondary_v);
    printf("========================================\n\n");
}
    failure_code = MPG_FAILURE_NEWTON_DIVERGED;
    return false;
  }

  /* Per Codex Pass 3 #2: Use current geometry/seeds (may have walked) for post-convergence evaluation.
   * If Newton converged on adjacent triangle, using original geometry would be inconsistent. */
  if (!specular_parameters_from_surface(kg, sd, current_primary_geometry, current_primary_seed, 0, primary_u, primary_v, primary_params)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: specular_parameters_from_surface (post-Newton, primary) (line 2518)\n");
}
    failure_code = MPG_FAILURE_NO_SPECULAR;
    return false;
  }

  if (!build_primary_shading_data(sd, current_primary_geometry, current_primary_seed, primary_u, primary_v, primary_sd)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: build_primary_shading_data (post-Newton) (line 2523)\n");
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  if (!specular_parameters_from_surface(kg,
                                        primary_sd,
                                        current_secondary_geometry,
                                        current_secondary_seed,
                                        1,
                                        secondary_u,
                                        secondary_v,
                                        secondary_params))
  {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: specular_parameters_from_surface (post-Newton, secondary) (line 2528)\n");
}
    failure_code = MPG_FAILURE_NO_SPECULAR;
    return false;
  }

  if (!evaluate_double_bounce(receiver,
                              current_primary_geometry,
                              primary_params,
                              current_secondary_geometry,
                              secondary_params,
                              seed.light_sample,
                              primary_u,
                              primary_v,
                              secondary_u,
                              secondary_v,
                              seed.use_smooth_normals,
                              secondary_seed.use_smooth_normals,
                              eval))
  {
    failure_code = (eval.primary.tir || eval.secondary.tir) ? MPG_FAILURE_TOTAL_INTERNAL_REFLECTION :
                                                                 MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  /* For double-bounce with directional lights, convert to point light for Jacobian.
   * The primary vertex "sees" the secondary vertex as a finite light source.
   * Distance is the actual geometric distance, not the fake infinite distance. */
  MpgSeedRay primary_seed_for_jacobian = seed;
  const bool seed_light_was_distant = (seed.light_sample.t == FLT_MAX) ||
                                      (seed.light_sample.type == LIGHT_DISTANT) ||
                                      (seed.light_sample.type == LIGHT_BACKGROUND);
  primary_seed_for_jacobian.light_sample.P = eval.secondary.point;
  /* Use actual distance between primary and secondary vertices, not fake infinite distance.
   * This matches Mitsuba's approach of placing a fake vertex at distance 1 for directional lights. */
  const float actual_distance_primary_to_secondary = len(eval.secondary.point - eval.primary.point);
  primary_seed_for_jacobian.light_sample.t = actual_distance_primary_to_secondary;
  primary_seed_for_jacobian.light_sample.D = eval.primary.dir_sl;
  if (seed_light_was_distant) {
    primary_seed_for_jacobian.light_sample.type = LIGHT_POINT;
  }

  float matrix_primary[2][2];
  if (!compute_residual_matrix(receiver, primary_seed_for_jacobian, primary_geometry, eval.primary, matrix_primary)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: compute_residual_matrix (primary double-bounce) (line 2586)\n");
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  float determinant_primary = matrix_primary[0][0] * matrix_primary[1][1] -
                              matrix_primary[0][1] * matrix_primary[1][0];
  if (!isfinite_safe(determinant_primary) || fabsf(determinant_primary) <= 1.0e-12f) {
    failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    return false;
  }

  const float area_primary = len(cross(eval.primary.dXdu, eval.primary.dXdv));
  const float cos_primary = fabsf(dot(eval.primary.normal, eval.primary.dir_ds));
  if (area_primary < 1e-4f || cos_primary < 1e-4f) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: geometry check (primary double-bounce), area=%.9f cos=%.9f (line 2600)\n",
           area_primary, cos_primary);
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  /* Compute Jacobian for primary bounce including geometric term.
   * Use same threshold as single-bounce for consistency. */
  const float distance_primary_sq = eval.primary.distance_ds * eval.primary.distance_ds;
  const float min_distance_sq = 1e-8f;  /* Max Jacobian contribution: ~1e8 */
  if (distance_primary_sq < min_distance_sq) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  const float geometric_factor_primary = cos_primary / distance_primary_sq;
  const float jacobian_primary = fabsf(determinant_primary) * geometric_factor_primary;

  /* Check primary Jacobian for numerical issues */
  if (!isfinite_safe(jacobian_primary) || jacobian_primary <= 1.0e-12f) {
    failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    return false;
  }

  ShadingPoint intermediate_point = receiver;
  intermediate_point.position = eval.primary.point;
  intermediate_point.geometric_normal = eval.primary.normal;
  intermediate_point.shading_normal = eval.primary.normal;

  float matrix_secondary[2][2];
  if (!compute_residual_matrix(intermediate_point, secondary_seed, secondary_geometry, eval.secondary, matrix_secondary)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: compute_residual_matrix (secondary double-bounce) (line 2632)\n");
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  float determinant_secondary = matrix_secondary[0][0] * matrix_secondary[1][1] -
                                matrix_secondary[0][1] * matrix_secondary[1][0];
  if (!isfinite_safe(determinant_secondary) || fabsf(determinant_secondary) <= 1.0e-12f) {
    failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    return false;
  }

  const float area_secondary = len(cross(eval.secondary.dXdu, eval.secondary.dXdv));
  const float cos_secondary = fabsf(dot(eval.secondary.normal, eval.secondary.dir_sl));
  if (area_secondary < 1e-4f || cos_secondary < 1e-4f) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: geometry check (secondary double-bounce), area=%.9f cos=%.9f (line 2649)\n",
           area_secondary, cos_secondary);
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  /* Compute Jacobian for secondary bounce including geometric term.
   * Use same threshold as primary and single-bounce for consistency. */
  const float distance_secondary_sq = eval.secondary.distance_ds * eval.secondary.distance_ds;
  if (distance_secondary_sq < min_distance_sq) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  const float geometric_factor_secondary = cos_secondary / distance_secondary_sq;
  const float jacobian_secondary = fabsf(determinant_secondary) * geometric_factor_secondary;

  /* Check secondary Jacobian for numerical issues before multiplying */
  if (!isfinite_safe(jacobian_secondary) || jacobian_secondary <= 1.0e-12f) {
    failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    return false;
  }

  const float jacobian_total = jacobian_primary * jacobian_secondary;
  if (!isfinite_safe(jacobian_total) || fabsf(jacobian_total) <= 1.0e-12f) {
    failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    return false;
  }

  const float3 dir_primary_in = eval.primary.dir_ds;
  const float3 dir_primary_out = eval.primary.dir_sl;
  const Spectrum primary_weight = evaluate_specular_weight(
      kg, primary_params, dir_primary_in, dir_primary_out, eval.primary.cos_theta_i, eval.primary.cos_theta_t);
  if (is_zero(primary_weight)) {
    failure_code = MPG_FAILURE_ZERO_THROUGHPUT;
    return false;
  }

  const float3 dir_secondary_in = eval.secondary.dir_ds;
  const float3 dir_secondary_out = eval.secondary.dir_sl;
  const Spectrum secondary_weight = evaluate_specular_weight(kg,
                                                            secondary_params,
                                                            dir_secondary_in,
                                                            dir_secondary_out,
                                                            eval.secondary.cos_theta_i,
                                                            eval.secondary.cos_theta_t);
  if (is_zero(secondary_weight)) {
    failure_code = MPG_FAILURE_ZERO_THROUGHPUT;
    return false;
  }

  const Spectrum spec_throughput = primary_weight * secondary_weight;
  if (is_zero(spec_throughput)) {
    failure_code = MPG_FAILURE_ZERO_THROUGHPUT;
    return false;
  }

  const float visibility_primary = compute_segment_visibility(kg,
                                                              sd.P,
                                                              sd.Ng,
                                                              eval.primary.point,
                                                              sd.time,
                                                              sd.object,
                                                              sd.prim,
                                                              OBJECT_NONE,
                                                              PRIM_NONE);
  /* For glass refraction, the intermediate ray travels through the glass interior.
   * For solidified geometry, entry and exit surfaces are different primitives of the
   * same object. Skip BOTH primitives so the ray can travel through the interior
   * without detecting either face as an occlusion. */
  const float visibility_intermediate = compute_segment_visibility(kg,
                                                                   eval.primary.point,
                                                                   eval.primary.normal,
                                                                   eval.secondary.point,
                                                                   sd.time,
                                                                   seed.object,
                                                                   seed.prim,
                                                                   secondary_seed.object,
                                                                   secondary_seed.prim);
  /* Skip visibility test for directional lights following the Mitsuba reference.
   * Directional lights are infinitely distant so shadow ray tests don't apply. */
  float visibility_secondary = 1.0f;
  if (seed.light_sample.t != FLT_MAX) {
    const float3 secondary_light_point =
        compute_distant_visibility_endpoint(seed.light_sample, eval.secondary.point);
    visibility_secondary = compute_segment_visibility(kg,
                                                      eval.secondary.point,
                                                      eval.secondary.normal,
                                                      secondary_light_point,
                                                      sd.time,
                                                      secondary_seed.object,
                                                      secondary_seed.prim,
                                                      seed.light_sample.object,
                                                      seed.light_sample.prim);
  }
  const float visibility = visibility_primary * visibility_intermediate * visibility_secondary;

  result.success = true;
  result.specular_vertex_count = 2;
  result.wi = eval.primary.dir_ds;
  result.visibility = visibility;
  result.jacobian_total = jacobian_total;
  result.jacobian = jacobian_total;
  result.specular_throughput = spec_throughput;
  result.spec_weight = spec_throughput;

  MpgSpecularVertex &primary_vertex = result.specular_vertices[0];
  primary_vertex.position = eval.primary.point;
  primary_vertex.normal = eval.primary.normal;
  primary_vertex.dir_in = dir_primary_in;
  primary_vertex.dir_out = dir_primary_out;
  primary_vertex.distance_in = eval.primary.distance_ds;
  primary_vertex.distance_out = eval.primary.distance_sl;
  primary_vertex.eta = eval.primary.eta;
  primary_vertex.cos_theta_in = eval.primary.cos_theta_i;
  primary_vertex.cos_theta_out = eval.primary.cos_theta_t;
  primary_vertex.throughput = primary_weight;
  primary_vertex.dXdu = eval.primary.dXdu;
  primary_vertex.dXdv = eval.primary.dXdv;
  primary_vertex.dNdu = eval.primary.dNdu;
  primary_vertex.dNdv = eval.primary.dNdv;
  primary_vertex.u = primary_u;
  primary_vertex.v = primary_v;
  primary_vertex.jacobian = jacobian_primary;
  primary_vertex.is_refraction = primary_params.is_refraction;
  primary_vertex.total_internal_reflection = eval.primary.tir;
  /* Per Codex: Store actual converged object/prim, not original seed */
  primary_vertex.object = current_primary_object;
  primary_vertex.prim = current_primary_prim;

  MpgSpecularVertex &secondary_vertex = result.specular_vertices[1];
  secondary_vertex.position = eval.secondary.point;
  secondary_vertex.normal = eval.secondary.normal;
  secondary_vertex.dir_in = dir_secondary_in;
  secondary_vertex.dir_out = dir_secondary_out;
  secondary_vertex.distance_in = eval.secondary.distance_ds;
  secondary_vertex.distance_out = eval.secondary.distance_sl;
  secondary_vertex.eta = eval.secondary.eta;
  secondary_vertex.cos_theta_in = eval.secondary.cos_theta_i;
  secondary_vertex.cos_theta_out = eval.secondary.cos_theta_t;
  secondary_vertex.throughput = secondary_weight;
  secondary_vertex.dXdu = eval.secondary.dXdu;
  secondary_vertex.dXdv = eval.secondary.dXdv;
  secondary_vertex.dNdu = eval.secondary.dNdu;
  secondary_vertex.dNdv = eval.secondary.dNdv;
  secondary_vertex.u = secondary_u;
  secondary_vertex.v = secondary_v;
  secondary_vertex.jacobian = jacobian_secondary;
  secondary_vertex.is_refraction = secondary_params.is_refraction;
  secondary_vertex.total_internal_reflection = eval.secondary.tir;
  /* Per Codex: Store actual converged object/prim, not original seed */
  secondary_vertex.object = current_secondary_object;
  secondary_vertex.prim = current_secondary_prim;

  result.is_refraction = secondary_vertex.is_refraction;
  result.specular_point = primary_vertex.position;
  result.specular_normal = primary_vertex.normal;
  result.dir_ds = -primary_vertex.dir_in;
  result.dir_sl = secondary_vertex.dir_out;
  result.distance_ds = primary_vertex.distance_in;
  result.distance_sl = secondary_vertex.distance_out;
  result.dXdu = primary_vertex.dXdu;
  result.dXdv = primary_vertex.dXdv;
  result.dNdu = primary_vertex.dNdu;
  result.dNdv = primary_vertex.dNdv;
  result.u = primary_vertex.u;
  result.v = primary_vertex.v;
  result.object = primary_vertex.object;
  result.prim = primary_vertex.prim;

  failure_code = MPG_FAILURE_NONE;

if (MPG_DEBUG::PARAMS()) {
  printf("========================================\n");
  printf("MPG DOUBLE-BOUNCE SOLVER: SUCCESS!\n");
  printf("========================================\n");
  printf("SOLUTION SUMMARY:\n");
  printf("  Primary vertex: (%.6f, %.6f, %.6f)\n",
         primary_vertex.position.x, primary_vertex.position.y, primary_vertex.position.z);
  printf("  Secondary vertex: (%.6f, %.6f, %.6f)\n",
         secondary_vertex.position.x, secondary_vertex.position.y, secondary_vertex.position.z);
  printf("  Primary refraction: %d, Secondary refraction: %d\n",
         primary_params.is_refraction, secondary_params.is_refraction);
  printf("  Visibility: %.6f\n", result.visibility);
  printf("  Specular throughput: (%.6f, %.6f, %.6f)\n",
         result.specular_throughput.x, result.specular_throughput.y, result.specular_throughput.z);
  printf("  Jacobian total: %.9e\n", result.jacobian_total);
  printf("  Outgoing direction (wi): (%.6f, %.6f, %.6f)\n",
         result.wi.x, result.wi.y, result.wi.z);
  printf("========================================\n\n");
}

  return true;
}


/* ========================================================================
 * Main Entry Point (mpg_try_connect)
 * From: mpg.cpp
 * ======================================================================== */


/* Debug printing toggles - set categories to true to enable specific debug output */
struct MPG_DEBUG {
  /* Final success handoff to integrator */
  static constexpr bool SUCCESS = false;
};

namespace {

constexpr int MPG_BOOTSTRAP_RNG_OFFSET = 128;

GuideSummary sanitize_guide_summary(const GuideSummary &input)
{
  GuideSummary result = input;

  if (!(isfinite_safe(result.mean_dir.x) && isfinite_safe(result.mean_dir.y) &&
        isfinite_safe(result.mean_dir.z)))
  {
    result.mean_dir = zero_float3();
  }

  if (!isfinite_safe(result.peak_weight)) {
    result.peak_weight = 0.0f;
  }
  else {
    result.peak_weight = fminf(fmaxf(result.peak_weight, 0.0f), 1.0f);
  }

  if (!isfinite_safe(result.kappa) || result.kappa < 0.0f) {
    result.kappa = 0.0f;
  }

  if (!isfinite_safe(result.rbar)) {
    result.rbar = 0.0f;
  }
  else {
    result.rbar = fminf(fmaxf(result.rbar, 0.0f), 1.0f);
  }

  return result;
}

float3 compute_distant_light_endpoint(const LightSample &light_sample,
                                      const float3 &origin)
{
  float3 dir = light_sample.D;
  if (is_zero(dir)) {
    return origin;
  }
  dir = normalize(dir);
  return origin + dir * MPG_DISTANT_LIGHT_VISIBILITY_DISTANCE;
}

float compute_visibility_after_update(KernelGlobals kg,
                                      const ShaderData &sd,
                                      const LightSample &light_sample,
                                      const MpgResult &result)
{
  if (result.specular_vertex_count <= 0) {
    return 0.0f;
  }

  float visibility = 1.0f;
  float3 segment_start = sd.P;
  float3 segment_normal = sd.Ng;
  int skip_object = sd.object;
  int skip_prim = sd.prim;

  for (int i = 0; i < result.specular_vertex_count; ++i) {
    const MpgSpecularVertex &vertex = result.specular_vertices[i];
    const float3 vertex_geometric_normal = safe_normalize(cross(vertex.dXdu, vertex.dXdv));
    const float3 visibility_normal = is_zero(vertex_geometric_normal) ? vertex.normal : vertex_geometric_normal;
    visibility *= mpg_compute_segment_visibility(kg,
                                                 segment_start,
                                                 segment_normal,
                                                 vertex.position,
                                                 sd.time,
                                                 skip_object,
                                                 skip_prim,
                                                 OBJECT_NONE,
                                                 PRIM_NONE);
    if (visibility == 0.0f) {
      return 0.0f;
    }

    segment_start = vertex.position;
    segment_normal = visibility_normal;
    skip_object = vertex.object;
    skip_prim = vertex.prim;
  }

  /* Skip visibility test for directional lights following the Mitsuba reference.
   * Directional lights are infinitely distant so shadow ray tests don't apply.
   * This matches the solver's behavior in mpg_solve.cpp. */
  if (light_sample.t != FLT_MAX) {
    /* For single-bounce refraction through closed geometry (solidified planes),
     * the ray from specular vertex to light must pass through the opposite face
     * of the same glass object. Skip all primitives of the glass object to allow
     * this by setting skip_prim to PRIM_NONE. */
    const bool is_single_bounce_refraction = (result.specular_vertex_count == 1 &&
                                              result.specular_vertices[0].is_refraction);
    const int skip_self_prim = is_single_bounce_refraction ? PRIM_NONE : skip_prim;

    visibility *= mpg_compute_segment_visibility(kg,
                                                 segment_start,
                                                 segment_normal,
                                                 light_sample.P,
                                                 sd.time,
                                                 skip_object,
                                                 skip_self_prim,
                                                 light_sample.object,
                                                 light_sample.prim);
  }

  return visibility;
}

}  // namespace

MpgResult mpg_try_connect(KernelGlobals kg,
                          const ShaderData &sd,
                          const ShaderClosure &bsdf,
                          const GuideSummary &g,
                          const MpgOptions &opt,
                          const uint32_t path_flag,
                          const int bounce,
                          RNGState &rng_state)
{
  /* Set current sample number for debug filtering (uses setter to avoid thread_local extern issues) */
  mpg_set_current_sample(rng_state.sample);

  MpgResult result{};
  result.failure_code = MPG_FAILURE_NONE;
  result.attempt_count = 0;
  result.gate_mask = MPG_GATE_MASK_NONE;

  const GuideSummary guide = sanitize_guide_summary(g);

  if (opt.max_bounces <= 0) {
    result.failure_code = MPG_FAILURE_UNSUPPORTED;
    return result;
  }

  /* MPG implements the Mitsuba reference "Manifold Path Guiding" (SIGGRAPH Asia 2023).
   * The technique handles paths with structure: Non-specular → [Specular chain] → Non-specular
   *
   * MPG MUST be invoked from NON-SPECULAR surfaces only (diffuse, glossy with roughness).
   * It should NOT be called from specular surfaces like glass, mirrors, or perfect refraction.
   *
   * The "non-specular separators" in the paper are the starting and ending surfaces that
   * bound the specular chain. The specular surfaces themselves are found by the solver,
   * not provided as the invocation point.
   *
   * Reject:
   * - Transparent/portal (except ray portal with specialized handling)
   * - Glass closures (specular, handled as intermediate vertices by solver)
   * - Pure specular reflection/refraction (delta BSDFs)
   *
   * Reference: https://dl.acm.org/doi/abs/10.1145/3618360 */
  const bool is_singular = CLOSURE_IS_BSDF_SINGULAR(bsdf.type);
  const bool is_glass = CLOSURE_IS_GLASS(bsdf.type);

  if (is_singular && !CLOSURE_IS_RAY_PORTAL(bsdf.type)) {
    result.failure_code = MPG_FAILURE_UNSUPPORTED;
    return result;
  }

  if (is_glass) {
    /* Glass surfaces are specular - they should be found by the solver as intermediate
     * vertices, not used as the starting "non-specular separator" for MPG. */
    result.failure_code = MPG_FAILURE_UNSUPPORTED;
    return result;
  }

  /* Also reject delta (perfectly specular) microfacet surfaces.
   * These have alpha_x, alpha_y ≈ 0 and behave like perfect mirrors/refractors.
   * Only roughened surfaces (glossy with alpha > 0) are valid non-specular separators. */
  if (CLOSURE_IS_BSDF_MICROFACET(bsdf.type)) {
    const MicrofacetBsdf *mf = reinterpret_cast<const MicrofacetBsdf *>(&bsdf);
    const bool is_delta = (mf->alpha_x <= 1e-6f) && (mf->alpha_y <= 1e-6f);
    if (is_delta) {
      result.failure_code = MPG_FAILURE_UNSUPPORTED;
      return result;
    }
  }

  const bool gate_active = (opt.gate_w > 0.0f) || (opt.gate_kappa > 0.0f);
  const bool has_direction_relaxed = (guide.rbar > 1.0e-4f);
  const bool has_direction_strict = (guide.rbar > 1.0e-3f);
  const bool strict_gate = has_direction_strict && (guide.peak_weight >= opt.gate_w) &&
                           (guide.kappa >= opt.gate_kappa);
  const bool relaxed_gate = opt.relax_gate && has_direction_relaxed;
  const bool bootstrap_gate = opt.relax_gate && !has_direction_relaxed;

  uint32_t gate_mask = MPG_GATE_MASK_NONE;
  if (gate_active) {
    gate_mask |= MPG_GATE_MASK_ACTIVE;
  }
  if (has_direction_relaxed) {
    gate_mask |= MPG_GATE_MASK_HAS_DIRECTION_RELAXED;
  }
  if (has_direction_strict) {
    gate_mask |= MPG_GATE_MASK_HAS_DIRECTION_STRICT;
  }
  if (strict_gate) {
    gate_mask |= MPG_GATE_MASK_STRICT_PASS;
  }
  if (relaxed_gate) {
    gate_mask |= MPG_GATE_MASK_RELAX_PASS;
  }
  if (bootstrap_gate) {
    gate_mask |= MPG_GATE_MASK_BOOTSTRAP_PASS;
  }
  if (!gate_active) {
    gate_mask |= MPG_GATE_MASK_STRICT_PASS;
  }
  result.gate_mask = gate_mask;

  const bool strict_gate_pass = (gate_mask & MPG_GATE_MASK_STRICT_PASS) != 0;
  const bool relaxed_gate_pass = (gate_mask & MPG_GATE_MASK_RELAX_PASS) != 0;
  const bool bootstrap_gate_pass = (gate_mask & MPG_GATE_MASK_BOOTSTRAP_PASS) != 0;
  const bool gate_permits_solver = strict_gate_pass || (opt.relax_gate &&
                                                        (relaxed_gate_pass || bootstrap_gate_pass));

  if (!gate_permits_solver) {
    result.failure_code = MPG_FAILURE_GATE;
    result.attempt_count = 0;
    return result;
  }
  /* When the gate is disabled we still attempt a bootstrap seed even if the guide has no
   * dominant direction. This mirrors the Mitsuba reference fallback behaviour. */

  MpgSeedRay seed;
  MpgFailureCode seed_failure = MPG_FAILURE_NONE;
  bool seed_success = mpg_generate_seed(
      kg, sd, bsdf, guide, opt, path_flag, bounce, rng_state, seed, seed_failure);

  bool attempted_bootstrap_fallback = false;
  bool used_bootstrap_seed = false;

  if (!seed_success && bootstrap_gate_pass) {
    attempted_bootstrap_fallback = true;

    GuideSummary bootstrap_summary = guide;
    bootstrap_summary.mean_dir = zero_float3();
    bootstrap_summary.peak_weight = 0.0f;
    bootstrap_summary.kappa = 0.0f;
    bootstrap_summary.rbar = 0.0f;

    MpgFailureCode bootstrap_failure = MPG_FAILURE_NONE;
    seed_success = mpg_generate_seed(kg,
                                     sd,
                                     bsdf,
                                     bootstrap_summary,
                                     opt,
                                     path_flag,
                                     bounce,
                                     rng_state,
                                     seed,
                                     bootstrap_failure,
                                     MPG_BOOTSTRAP_RNG_OFFSET);

    if (!seed_success) {
      if (bootstrap_failure != MPG_FAILURE_NONE) {
        seed_failure = bootstrap_failure;
      }
    }
    else {
      seed_failure = MPG_FAILURE_NONE;
      used_bootstrap_seed = true;
#ifdef WITH_CYCLES_DEBUG
      if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
        LOG_DEBUG << "MPG bootstrap gating fallback seed succeeded";
      }
#endif
    }
  }

  if (!seed_success) {
#ifdef WITH_CYCLES_DEBUG
    if (bootstrap_gate_pass && LOG_IS_ON(LOG_LEVEL_DEBUG)) {
      LOG_DEBUG << "MPG bootstrap gating exhausted seeds (fallback="
                << (attempted_bootstrap_fallback ? "yes" : "no")
                << ", failure=" << static_cast<int>(seed_failure) << ")";
    }
#endif
    result.failure_code = (seed_failure != MPG_FAILURE_NONE) ? seed_failure : MPG_FAILURE_SEED;
    result.attempt_count = 0;
    result.seed_trial_count = seed.trial_count;
    result.seed_accepted_trial_count = seed.accepted_trial_count;
    result.seed_guided_trial_count = seed.guided_trial_count;
    result.seed_fallback_trial_count = seed.fallback_trial_count;
    result.seed_pdf_raw = seed.seed_pdf_raw;
    result.seed_pdf = seed.seed_pdf;
    result.seed_resample_factor = seed.seed_resample_factor;
    result.seed_branch_pdf = seed.seed_branch_pdf;
    result.seed_direction_pdf = seed.seed_direction_pdf;
    result.seed_scatter_pdf = seed.seed_scatter_pdf;
    result.seed_direction_normalized = seed.direction_normalized;
    result.seed_tau_bits = seed.tau_bits;
    result.seed_tau_count = seed.tau_count;
    result.seed_branch = seed.branch;
    result.seed_scatter = seed.scatter;
    result.bounce_pdf_raw = seed.bounce_pdf_raw;
    result.bounce_pdf = seed.bounce_pdf;
    result.bounce_count = seed.bounce_count;
    return result;
  }
  result.seed_pdf_raw = seed.seed_pdf_raw;
  result.seed_trial_count = seed.trial_count;
  result.seed_accepted_trial_count = seed.accepted_trial_count;
  result.seed_guided_trial_count = seed.guided_trial_count;
  result.seed_fallback_trial_count = seed.fallback_trial_count;
  result.light = seed.light_sample;
  result.seed_pdf = seed.seed_pdf;
  result.seed_resample_factor = seed.seed_resample_factor;
  result.seed_branch_pdf = seed.seed_branch_pdf;
  result.seed_direction_pdf = seed.seed_direction_pdf;
  result.seed_scatter_pdf = seed.seed_scatter_pdf;
  result.seed_direction_normalized = seed.direction_normalized;
  result.seed_tau_bits = seed.tau_bits;
  result.seed_tau_count = seed.tau_count;
  result.seed_branch = seed.branch;
  result.seed_scatter = seed.scatter;
  result.bounce_pdf_raw = seed.bounce_pdf_raw;
  result.bounce_pdf = seed.bounce_pdf;
  result.bounce_count = seed.bounce_count;
  if (!is_zero(seed.direction)) {
    result.wi = normalize(seed.direction);
  }

  if (!gate_permits_solver) {
    result.failure_code = MPG_FAILURE_GATE;
    result.attempt_count = 0;
    return result;
  }

  MpgSolverOutput solution;
  bool solved = false;
  int attempt_count = 0;
  MpgFailureCode solver_failure = MPG_FAILURE_NONE;

  /* Retry loop: try up to max_seed_repeat_trials different seeds */
  const int max_trials = (opt.max_seed_repeat_trials > 0) ? opt.max_seed_repeat_trials : 1;
  for (int trial = 0; trial < max_trials && !solved; ++trial) {
    /* For retries (trial > 0), generate a new seed */
    if (trial > 0) {
      MpgFailureCode retry_seed_failure = MPG_FAILURE_NONE;
      bool retry_seed_success = mpg_generate_seed(
          kg, sd, bsdf, guide, opt, path_flag, bounce, rng_state, seed, retry_seed_failure);

      if (!retry_seed_success) {
        /* If we can't generate a new seed, continue to next trial or give up */
        seed_failure = retry_seed_failure;
        continue;
      }
    }

    solution.seed_resample_factor = seed.seed_resample_factor;
    solution.seed_branch = seed.branch;
    solution.seed_branch_pdf = seed.seed_branch_pdf;
    solution.seed_direction_pdf = seed.seed_direction_pdf;
    solution.seed_scatter = seed.scatter;
    solution.seed_scatter_pdf = seed.seed_scatter_pdf;
    solution.seed_guided_trial_count = seed.guided_trial_count;
    solution.seed_fallback_trial_count = seed.fallback_trial_count;

    if (seed.bounce_count == 2) {
      ++attempt_count;
      solved = mpg_solve_double_bounce(kg, sd, bsdf, seed, guide, opt, rng_state, solution, solver_failure);
      if (!solved && solver_failure != MPG_FAILURE_NONE) {
        result.failure_code = solver_failure;
      }
    }
    else {
      ++attempt_count;
      solved = mpg_solve_single_bounce(kg, sd, bsdf, seed, guide, opt, rng_state, solution, solver_failure);
      if (!solved && solver_failure != MPG_FAILURE_NONE) {
        result.failure_code = solver_failure;
      }
    }
  }

  result.attempt_count = attempt_count;

  if (!solved) {
    if (result.failure_code == MPG_FAILURE_NONE) {
      result.failure_code = (solver_failure != MPG_FAILURE_NONE) ? solver_failure : MPG_FAILURE_NO_SPECULAR;
    }
    result.attempt_count = attempt_count;
#ifdef WITH_CYCLES_DEBUG
    if (solver_failure == MPG_FAILURE_NEWTON_DIVERGED && seed.scatter == MPG_SEED_SCATTER_REFRACTION &&
        seed.bounce_count == 1)
    {
      DCHECK(false);
    }
#endif
#ifdef WITH_CYCLES_DEBUG
    if (bootstrap_gate_pass && LOG_IS_ON(LOG_LEVEL_DEBUG)) {
      LOG_DEBUG << "MPG bootstrap gate solver attempts=" << attempt_count
                << " (used_bootstrap_seed=" << (used_bootstrap_seed ? "yes" : "no")
                << ", failure=" << static_cast<int>(result.failure_code) << ")";
    }
#endif
    return result;
  }

  if (solution.specular_vertex_count <= 0) {
    result.failure_code = MPG_FAILURE_NO_SPECULAR;
    result.attempt_count = attempt_count;
    return result;
  }

  result.visibility = solution.visibility;
  result.spec_weight = solution.specular_throughput;
  result.specular_vertex_count = solution.specular_vertex_count;
  result.bounce_count = solution.specular_vertex_count;
  for (int i = 0; i < solution.specular_vertex_count; ++i) {
    result.specular_vertices[i] = solution.specular_vertices[i];
  }

  const MpgSpecularVertex &exit_vertex =
      solution.specular_vertices[solution.specular_vertex_count - 1];
  const bool exit_is_refraction = exit_vertex.is_refraction;
#ifdef WITH_CYCLES_DEBUG
  DCHECK(solution.is_refraction == exit_is_refraction);
#endif

  LightSample light_sample = seed.light_sample;
  const float pdf_selection = light_sample.pdf_selection;
  if (!(isfinite_safe(pdf_selection) && pdf_selection > 0.0f)) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
    result.light_pdf = 0.0f;
    return result;
  }

  /* Validate the stored selection probability before reusing it. The Mitsuba
   * reference keeps the receiver-side probability, so only perform light-link
   * compatibility checks here to preserve existing failure codes. */
  /* Light linking is evaluated at the diffuse receiver. Keep the original object id
   * from the shading point instead of the specular vertex to avoid rejecting valid
   * connections when intermediate specular surfaces belong to a different object. */
  const int receiver_object = light_link_receiver_nee(kg, &sd);
  const int emitter_object = light_sample.object;

#ifdef __LIGHT_TREE__
  if (kernel_data.integrator.use_light_tree) {
    if (light_sample.emitter_id < 0) {
      result.attempt_count = attempt_count;
      result.failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
      result.light_pdf = 0.0f;
      return result;
    }
  }
  else
#endif
  {
    if (!light_link_object_match(kg, receiver_object, emitter_object)) {
      result.attempt_count = attempt_count;
      result.failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
      result.light_pdf = 0.0f;
      return result;
    }
  }

  /* `light_sample_update` expects `ls->pdf` without the selection term. */
  light_sample.pdf /= pdf_selection;

  uint32_t updated_path_flag = path_flag;
  if (exit_is_refraction) {
    updated_path_flag |= PATH_RAY_MIS_HAD_TRANSMISSION;
  }

  light_sample.pdf_selection = 1.0f;
  light_sample_update(kg, &light_sample, exit_vertex.position, exit_vertex.normal, updated_path_flag);
  /* Restore the receiver-side selection probability rather than recomputing
   * it at the specular vertex. This matches the Mitsuba reference measure and
   * keeps the technique PDF consistent with the stored seed. */
  light_sample.pdf *= pdf_selection;
  light_sample.pdf_selection = pdf_selection;

  LightSample tmp = seed.light_sample;
  /* Recompute the pre-MPG NEE pdf at the receiver. Strip the selection probability while the
   * light sample is updated so it is re-applied exactly once. */
  tmp.pdf /= pdf_selection;
  tmp.pdf_selection = 1.0f;
  light_sample_update(kg, &tmp, sd.P, sd.N, path_flag);
  float nee_pdf_sa = mpg_light_sample_pdf_solid(kg, sd, tmp);
  nee_pdf_sa *= pdf_selection;
  if (!isfinite_safe(nee_pdf_sa)) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_NEE_PDF;
    return result;
  }
  if (nee_pdf_sa <= 0.0f) {
    nee_pdf_sa = 0.0f;
  }
  result.nee_pdf = nee_pdf_sa;
  tmp.pdf = nee_pdf_sa;

  float p_light = mpg_light_sample_pdf_solid(kg, sd, light_sample);
  result.light_pdf = p_light;
  if (!isfinite_safe(p_light) || p_light <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
    return result;
  }
  light_sample.pdf = p_light;

  result.visibility = compute_visibility_after_update(kg, sd, light_sample, result);

  const float p_seed = fmaxf(seed.seed_pdf, 1.0e-16f);
  result.seed_pdf = p_seed;
#ifdef WITH_CYCLES_DEBUG
  const int total_trials = seed.trial_count;
  const float expected_trials =
      (isfinite_safe(seed.seed_resample_factor) && seed.seed_resample_factor > 0.0f) ?
          seed.seed_resample_factor :
          0.0f;
  const float acceptance_probability = (expected_trials > 0.0f) ?
                                           (1.0f / expected_trials) :
                                           0.0f;
  const float mitsuba_seed_pdf = mpg_rebuild_seed_pdf(seed);
  if (total_trials > 0 && mitsuba_seed_pdf > 0.0f) {
    if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
      DCHECK(seed.scatter != MPG_SEED_SCATTER_NONE);
      DCHECK(seed.seed_scatter_pdf > 0.0f);
      LOG_DEBUG << "MPG seed pdf parity (trials=" << total_trials
                << "): cycles=" << p_seed << ", Mitsuba=" << mitsuba_seed_pdf
                << ", raw=" << seed.seed_pdf_raw
                << ", expected_trials=" << expected_trials
                << ", accept_p=" << acceptance_probability
                << ", branch=" << seed.seed_branch_pdf
                << ", dir=" << seed.seed_direction_pdf
                << ", scatter=" << seed.seed_scatter_pdf
                << ", scatter_branch=" << static_cast<int>(seed.scatter)
                << ", bounce=" << seed.bounce_pdf << " (folded into seed)"
                << ", bounce_raw=" << seed.bounce_pdf_raw;
    }
    if (isfinite_safe(p_seed) && isfinite_safe(mitsuba_seed_pdf)) {
      const float tolerance = fmaxf(fabsf(mitsuba_seed_pdf), 1.0e-16f) * 1.0e-4f;
      DCHECK(fabsf(p_seed - mitsuba_seed_pdf) <= tolerance);
    }
  }
#endif
  if (!isfinite_safe(p_seed) || p_seed <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return result;
  }

  const float raw_jacobian = solution.jacobian_total;
  const float J_total = fabsf(raw_jacobian);
  if (!isfinite_safe(J_total) || J_total <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    result.jacobian_total = raw_jacobian;
    return result;
  }
  result.jacobian_total = J_total;

  /* Technique PDF = seed_pdf * jacobian * light_pdf.
   * The Jacobian converts seed sampling from specular surface parameter space
   * to solid angle measure at the receiver. Store light_pdf WITHOUT Jacobian
   * so it can be compared correctly with NEE PDF in MIS weighting. */
  const float technique_pdf = p_seed * J_total * p_light;
  result.pdf = technique_pdf;
  result.light_pdf = p_light;

  if (!isfinite_safe(seed.bounce_pdf) || seed.bounce_pdf <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_PDF;
    return result;
  }
  result.bounce_pdf = seed.bounce_pdf;
  result.bounce_pdf_raw = seed.bounce_pdf_raw;

  if (!isfinite_safe(technique_pdf) || technique_pdf <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_PDF;
    return result;
  }
#ifdef WITH_CYCLES_DEBUG
  if (LOG_IS_ON(LOG_LEVEL_DEBUG) && seed.bounce_count != result.bounce_count) {
    LOG_DEBUG << "MPG solver adjusted bounce count (seed=" << seed.bounce_count
              << ", solution=" << result.bounce_count
              << ", scatter=" << static_cast<int>(seed.scatter)
              << ", seed_pdf=" << result.seed_pdf
              << ", light_pdf=" << result.light_pdf
              << ", jacobian=" << result.jacobian_total
              << ", technique_pdf=" << technique_pdf
              << ", bounce_pdf=" << result.bounce_pdf << ")";
  }
#endif

#ifdef WITH_CYCLES_DEBUG
  {
    float eval_pdf = 0.0f;
    const bool eval_success = mpg_evaluate_pdf(kg, sd, bsdf, guide, seed, solution, eval_pdf);
    DCHECK(eval_success);
    if (eval_success) {
      const float tolerance = fmaxf(fabsf(result.pdf), 1.0e-16f) * 1.0e-4f;
      DCHECK(fabsf(eval_pdf - result.pdf) <= tolerance);
    }
  }
#endif

  /* Final sanity check - ensure all PDFs and Jacobian are valid before returning success */
  if (!isfinite_safe(result.jacobian_total) || result.jacobian_total <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    return result;
  }
  if (!isfinite_safe(result.pdf) || result.pdf <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_PDF;
    return result;
  }

  result.success = true;
  result.failure_code = MPG_FAILURE_NONE;
  result.wi = solution.wi;
  result.seed_pdf = p_seed;
  result.seed_pdf_raw = seed.seed_pdf_raw;
  result.bounce_pdf = seed.bounce_pdf;
  result.bounce_pdf_raw = seed.bounce_pdf_raw;
  result.light = light_sample;
  result.attempt_count = attempt_count;
  result.seed_trial_count = seed.trial_count;
  result.seed_accepted_trial_count = seed.accepted_trial_count;

#ifdef WITH_CYCLES_DEBUG
  if (bootstrap_gate_pass && LOG_IS_ON(LOG_LEVEL_DEBUG)) {
    LOG_DEBUG << "MPG bootstrap gate solver attempts=" << attempt_count
              << " (used_bootstrap_seed=" << (used_bootstrap_seed ? "yes" : "no")
              << ", success)";
  }
  if ((result.gate_mask & MPG_GATE_MASK_STRICT_PASS) != 0) {
    DCHECK(isfinite_safe(result.seed_pdf) && result.seed_pdf > 0.0f);
    DCHECK(isfinite_safe(result.light_pdf) && result.light_pdf > 0.0f);
    DCHECK(isfinite_safe(result.bounce_pdf) && result.bounce_pdf > 0.0f);
    DCHECK(isfinite_safe(result.jacobian_total) && result.jacobian_total > 0.0f);
    DCHECK(isfinite_safe(result.pdf) && result.pdf > 0.0f);
    if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
      LOG_DEBUG << "MPG strict gate factors: seed(with bounce)=" << result.seed_pdf
                << ", bounce=" << result.bounce_pdf
                << ", light_receiver=" << result.light_pdf
                << " (light_spec=" << p_light << ")"
                << ", J=" << result.jacobian_total << ", pdf=" << result.pdf;
    }
  }
#endif

if constexpr (MPG_DEBUG::SUCCESS) {
  printf("████████████████████████████████████████\n");
  printf("MPG_TRY_CONNECT: RETURNING SUCCESS TO INTEGRATOR\n");
  printf("████████████████████████████████████████\n");
  printf("  result.success = %d\n", result.success);
  printf("  result.failure_code = %d (0=none)\n", (int)result.failure_code);
  printf("  result.visibility = %.6f\n", result.visibility);
  printf("  result.spec_weight = (%.6f, %.6f, %.6f)\n",
         result.spec_weight.x, result.spec_weight.y, result.spec_weight.z);
  printf("  result.jacobian_total = %.9e\n", result.jacobian_total);
  printf("  result.pdf (technique) = %.9e\n", result.pdf);
  printf("  result.seed_pdf = %.9e\n", result.seed_pdf);
  printf("  result.light_pdf = %.9e\n", result.light_pdf);
  printf("  result.nee_pdf = %.9e\n", result.nee_pdf);
  printf("  result.wi = (%.6f, %.6f, %.6f)\n", result.wi.x, result.wi.y, result.wi.z);
  printf("  result.bounce_count = %d\n", result.bounce_count);
  const bool spec_weight_nonzero = !is_zero(result.spec_weight);
  const bool wi_valid = !is_zero(result.wi);
  printf("  spec_weight_nonzero = %d, wi_valid = %d\n", spec_weight_nonzero, wi_valid);
  printf("████████████████████████████████████████\n\n");
}

  return result;
}


CCL_NAMESPACE_END
