/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_solve.h"
#include "manifold/mpg_seed.h"

#include "kernel/bvh/bvh.h"
#include "kernel/bvh/util.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/closure/bsdf_util.h"
#include "kernel/integrator/state.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/object.h"
#include "kernel/geom/shader_data.h"
#include "kernel/geom/triangle.h"
#include "kernel/svm/types.h"
#include "kernel/types.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#ifdef WITH_CYCLES_DEBUG
#  include "util/log.h"
#endif

CCL_NAMESPACE_BEGIN

/* Thread-local storage for current sample number (set by mpg_try_connect) */
static thread_local int g_current_sample = -1;

/* Forward declaration for debug helper (defined after MPG_DEBUG struct) */
static inline bool debug_print_enabled();

/* Debug printing toggles - set categories to true to enable specific debug output */
struct MPG_DEBUG {
  /* Sample number filter: Only print debug output for this sample number.
   * Set to -1 to print all samples, or a specific number (e.g., 127 for sample 127).
   * Example: SAMPLE_FILTER = 127 means only print on sample 127 (0-indexed). */
  static constexpr int SAMPLE_FILTER = -1;  // -1 = all samples, or set to specific sample number

  /* Base flags - these are AND'ed with sample filter automatically via helper functions below */
  static constexpr bool SEED_BASE = false;
  static constexpr bool GEOMETRY_BASE = false;
  static constexpr bool PARAMS_BASE = false;
  static constexpr bool NEWTON_BASE = false;
  static constexpr bool NEWTON_DETAIL_BASE = false;
  static constexpr bool EVAL_FAIL_BASE = false;
  static constexpr bool SUCCESS_BASE = false;

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
  /* Use geometric normal directly for entering/exiting determination.
   * For refraction: dot(normal, ray) > 0 means exiting, < 0 means entering. */
  const float3 refraction_normal = normal;

if (MPG_DEBUG::PARAMS()) {
  if (params.is_refraction) {
    printf("MPG DEBUG compute_specular: refraction direction determination\n");
    printf("  normal: (%.6f, %.6f, %.6f)\n", normal.x, normal.y, normal.z);
    printf("  dir_ds: (%.6f, %.6f, %.6f)\n", dir_ds.x, dir_ds.y, dir_ds.z);
    printf("  dot(normal, dir_ds): %.6f\n", dot(normal, dir_ds));
  }
}

  /* Use backfacing flag for robust entering/exiting determination.
   * SD_BACKFACING means ray hit back face, which for closed objects means exiting.
   * This is more reliable than dot(normal, ray) for thin geometry. */
  const bool exiting = params.backfacing;

  /* Orient normal to ensure it points toward the medium the ray came from.
   * This ensures cos_theta_i = -dot(dir_ds, oriented_normal) > 0.
   * Use the same normal (microfacet or geometric) that was used for exiting determination. */
  float3 oriented_normal = exiting ? -refraction_normal : refraction_normal;

  if (!params.is_refraction) {
    /* For reflection, use propagation direction (toward surface) */
    const float3 incoming_reflect = dir_ds;
    tir = false;
    cos_theta_i = -dot(incoming_reflect, oriented_normal);
    cos_theta_t = cos_theta_i;
    eta_used = 1.0f;
    return reflect_dir(incoming_reflect, oriented_normal);
  }

  /* Compute eta ratio for reverse ray tracing.
   * Eta = n_incident / n_transmitted for Snell's law.
   * If exiting (glass→air): eta = n_glass/n_air = IOR.
   * If entering (air→glass): eta = n_air/n_glass = 1/IOR. */
  const float safe_base_eta = fmaxf(params.base_eta, 1e-6f);
  const float eta_ratio = exiting ? safe_base_eta : (1.0f / safe_base_eta);
  const float safe_eta_ratio = fmaxf(eta_ratio, 1e-6f);

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

  /* Use the precomputed normals from load_surface_geometry.
   * For smooth shading: interpolate vertex normals
   * For flat shading: use the face normal (already computed and stored in geometry.normals) */
  if (seed.use_smooth_normals) {
    eval.normal = combine_vertex_normals(geometry, u, v);
    if (is_zero(eval.normal)) {
      /* Fallback to face normal if interpolation gives zero */
      eval.normal = geometry.normals[0];
    }
    eval.dNdu = compute_normal_derivative(geometry, u, v, eval.normal, true);
    eval.dNdv = compute_normal_derivative(geometry, u, v, eval.normal, false);
  }
  else {
    /* Use precomputed face normal from load_surface_geometry - don't recompute! */
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
  }

  float cos_theta_i = 0.0f, cos_theta_t = 0.0f, eta = 1.0f;
  const float3 spec_dir = compute_specular(
      eval.dir_ds, eval.normal, params, eval.tir, cos_theta_i, cos_theta_t, eta);
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

  /* Half-vector eta convention is opposite of Snell's law eta!
   * - Snell's law (compute_specular): eta = n_from / n_to
   *   - Entering glass: eta = 1/1.5 = 0.6667
   *   - Exiting glass: eta = 1.5/1 = 1.5
   * - Half-vector (Mitsuba): eta = n_to / n_from
   *   - Entering glass: eta = 1.5/1 = 1.5
   *   - Exiting glass: eta = 1/1.5 = 0.6667
   * Therefore we use the reciprocal of eval.eta for the half-vector. */
  const float h_eta = eval.refractive ? (1.0f / eval.eta) : 1.0f;

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

  const float3 shading_normal = safe_normalize(spec_sd.N);
  const bool has_shading_normal = !is_zero(shading_normal);

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

  float3 hemisphere_normal = shading_normal;
  bool hemisphere_normal_valid = has_shading_normal;
  if (!hemisphere_normal_valid && has_geometric_normal) {
    hemisphere_normal = geometric_normal;
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
        if (has_shading_normal) {
          refraction_params.normal = shading_normal;
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
    if (has_shading_normal && !params.has_normal) {
      params.normal = shading_normal;
      params.has_normal = true;
    }
    else if (!params.has_normal && has_geometric_normal) {
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
  if (has_shading_normal) {
    params.normal = shading_normal;
    params.has_normal = true;
  }
  else if (has_geometric_normal) {
    params.normal = geometric_normal;
    params.has_normal = true;
  }
  {
    const float3 spec_point = geometry.verts[0] * (1.0f - u - v) +
                              geometry.verts[1] * u +
                              geometry.verts[2] * v;
    const float3 dir_ds = normalize(spec_point - sd.P);
    float tmp_distance = 0.0f;
    float3 dir_sl;
    compute_light_sample_direction(seed.light_sample, spec_point, dir_sl, tmp_distance);
    const float3 incident = -dir_ds;
    const float3 outgoing = dir_sl;
    const float s = dot(params.microfacet.N, incident) * dot(params.microfacet.N, outgoing);
    const bool bad_refraction = params.is_refraction ? (s > 0.0f) : false;
    const bool bad_reflection = !params.is_refraction ? (s < 0.0f) : false;
    if (bad_refraction || bad_reflection) {
      params.microfacet.N = -params.microfacet.N;
    }
  }
  if (!params.has_normal && has_shading_normal) {
    params.normal = shading_normal;
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

/* Compute Jacobian for Mitsuba's half-vector constraint formulation.
 * Constraint: C = [dot(s, h), dot(t, h)] where h = normalize(wi + eta * wo)
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
  if (!(g_len > 1e-8f)) {
    /* Degenerate case - set Jacobian to zero */
    J[0] = make_float3(0.0f, 0.0f, 0.0f);
    J[1] = make_float3(0.0f, 0.0f, 0.0f);
    return;
  }

  /* Derivative of normalized vector: dh/dx = (dg/dx / ||g||) - h * dot(h, dg/dx) */
  const float3 dh_du = (dg_du / g_len) - h * dot(h, dg_du);
  const float3 dh_dv = (dg_dv / g_len) - h * dot(h, dg_dv);

  /* Project derivatives onto surface tangent frame to get 2D Jacobian.
   * Embed in 3D vectors with third component = 0 for compatibility with solve_step. */
  J[0] = make_float3(dot(tangent_u, dh_du), dot(tangent_v, dh_du), 0.0f);
  J[1] = make_float3(dot(tangent_u, dh_dv), dot(tangent_v, dh_dv), 0.0f);
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

  /* Get primary normal respecting flat vs smooth shading.
   * Use mesh normals directly without flipping - entering/exiting is determined by
   * dot(normal, ray) in compute_specular(), which works correctly with actual mesh normals. */
  float3 primary_normal;
  if (seed.use_smooth_normals) {
    primary_normal = combine_vertex_normals(primary_geometry, primary_u, primary_v);
  }
  else {
    primary_normal = primary_geometry.normals[0];
  }

  if (!is_zero(primary_normal)) {
    primary_normal = safe_normalize(primary_normal);
  }
  else {
    /* Fallback to geometric normal only if mesh normal is degenerate */
    primary_normal = safe_normalize(cross(primary_geometry.dPdu, primary_geometry.dPdv));
  }

  if (is_zero(primary_normal)) {
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

  /* Use primary normal as-is for refraction calculation - no flipping based on ray direction.
   * The entering/exiting determination happens in compute_specular() using dot(normal, ray). */
  float3 specular_normal = primary_normal;

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

  bool tir = false;
  float cos_theta_i = 0.0f;
  float cos_theta_t = 0.0f;
  float eta_used = 1.0f;
  const float3 dir_sl = compute_specular(
      dir_ds, specular_normal, primary_params, tir, cos_theta_i, cos_theta_t, eta_used);

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

  /* Mitsuba approach: Accept all ray-traced secondary vertices without pre-validation.
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

  float3 tangent_u, tangent_v;
  if (!build_tangent_basis(eval.primary.dXdu, eval.primary.dXdv, tangent_u, tangent_v)) {
if (MPG_DEBUG::EVAL_FAIL()) {
    printf("  evaluate_double_bounce FAIL: Primary tangent basis failed\n");
}
    return false;
  }
  eval.residual[0] = dot(eval.primary.residual, tangent_u);
  eval.residual[1] = dot(eval.primary.residual, tangent_v);

  if (!build_tangent_basis(eval.secondary.dXdu, eval.secondary.dXdv, tangent_u, tangent_v)) {
if (MPG_DEBUG::EVAL_FAIL()) {
    printf("  evaluate_double_bounce FAIL: Secondary tangent basis failed\n");
}
    return false;
  }
  eval.residual[2] = dot(eval.secondary.residual, tangent_u);
  eval.residual[3] = dot(eval.secondary.residual, tangent_v);

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

  /* === Build tangent frames for both vertices === */
  float3 primary_tangent_u, primary_tangent_v;
  if (!build_tangent_basis(eval.primary.dXdu, eval.primary.dXdv, primary_tangent_u, primary_tangent_v)) {
    return false;
  }

  float3 secondary_tangent_u, secondary_tangent_v;
  if (!build_tangent_basis(eval.secondary.dXdu, eval.secondary.dXdv, secondary_tangent_u, secondary_tangent_v)) {
    return false;
  }

  /* === Compute half-vectors for both vertices === */
  const float3 primary_wi = -eval.primary.dir_ds;
  const float3 primary_wo = eval.primary.dir_sl;
  /* Use reciprocal of Snell's law eta for half-vector (see evaluate_specular) */
  const float primary_h_eta = eval.primary.refractive ? (1.0f / eval.primary.eta) : 1.0f;

  float3 primary_g = primary_wi + primary_h_eta * primary_wo;
  if (eval.primary.refractive) {
    primary_g = -primary_g;
  }
  const float primary_g_len = len(primary_g);
  if (!(primary_g_len > 1e-8f)) {
    return false;
  }
  const float3 primary_h = primary_g / primary_g_len;

  const float3 secondary_wi = -eval.secondary.dir_ds;
  const float3 secondary_wo = eval.secondary.dir_sl;
  /* Use reciprocal of Snell's law eta for half-vector (see evaluate_specular) */
  const float secondary_h_eta = eval.secondary.refractive ? (1.0f / eval.secondary.eta) : 1.0f;

  float3 secondary_g = secondary_wi + secondary_h_eta * secondary_wo;
  if (eval.secondary.refractive) {
    secondary_g = -secondary_g;
  }
  const float secondary_g_len = len(secondary_g);
  if (!(secondary_g_len > 1e-8f)) {
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

  /* Derivative of normalized half-vector: dh/dx = (dg/dx / ||g||) - h * dot(h, dg/dx) */
  const float3 d_primary_h_du1 = (d_primary_g_du1 / primary_g_len) - primary_h * dot(primary_h, d_primary_g_du1);
  const float3 d_primary_h_dv1 = (d_primary_g_dv1 / primary_g_len) - primary_h * dot(primary_h, d_primary_g_dv1);

  /* Project onto tangent frame */
  J[0][0] = dot(primary_tangent_u, d_primary_h_du1);  // dC1/du1
  J[0][1] = dot(primary_tangent_u, d_primary_h_dv1);  // dC1/dv1
  J[1][0] = dot(primary_tangent_v, d_primary_h_du1);  // dC2/du1
  J[1][1] = dot(primary_tangent_v, d_primary_h_dv1);  // dC2/dv1

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

  const float3 d_primary_h_du2 = (d_primary_g_du2 / primary_g_len) - primary_h * dot(primary_h, d_primary_g_du2);
  const float3 d_primary_h_dv2 = (d_primary_g_dv2 / primary_g_len) - primary_h * dot(primary_h, d_primary_g_dv2);

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

  const float3 d_secondary_h_du1 = (d_secondary_g_du1 / secondary_g_len) - secondary_h * dot(secondary_h, d_secondary_g_du1);
  const float3 d_secondary_h_dv1 = (d_secondary_g_dv1 / secondary_g_len) - secondary_h * dot(secondary_h, d_secondary_g_dv1);

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

  const float3 d_secondary_h_du2 = (d_secondary_g_du2 / secondary_g_len) - secondary_h * dot(secondary_h, d_secondary_g_du2);
  const float3 d_secondary_h_dv2 = (d_secondary_g_dv2 / secondary_g_len) - secondary_h * dot(secondary_h, d_secondary_g_dv2);

  J[2][2] = dot(secondary_tangent_u, d_secondary_h_du2);  // dC3/du2
  J[2][3] = dot(secondary_tangent_u, d_secondary_h_dv2);  // dC3/dv2
  J[3][2] = dot(secondary_tangent_v, d_secondary_h_du2);  // dC4/du2
  J[3][3] = dot(secondary_tangent_v, d_secondary_h_dv2);  // dC4/dv2

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
                                               const SpecularSurfaceGeometry &geometry,
                                               const SpecularParameters &params,
                                               float proposed_u,
                                               float proposed_v,
                                               int expected_object,
                                               int expected_prim,
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
  ray.time = 0.5f;  /* Mid-shutter time */
  ray.dP = differential_zero_compact();
  ray.dD = differential_zero_compact();

  /* Skip receiver surface to avoid self-intersection */
  ray.self.object = OBJECT_NONE;
  ray.self.prim = PRIM_NONE;
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

  /* Verify we hit the expected shape (Mitsuba's key check) */
  if (isect.object != expected_object || isect.prim != expected_prim) {
    /* Hit wrong surface - reject this Newton step */
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject FAIL: Wrong surface (expected obj=%d prim=%d, got obj=%d prim=%d)\n",
           expected_object, expected_prim, isect.object, isect.prim);
}
    return false;
  }

  /* Extract barycentric coordinates from the hit.
   * This is what Mitsuba does: use the ray-traced intersection point's actual coordinates. */
  hit_u = isect.u;
  hit_v = isect.v;

if (MPG_DEBUG::NEWTON_DETAIL()) {
  printf("    reproject SUCCESS: Hit correct surface at (u=%.6f, v=%.6f)\n", hit_u, hit_v);
}

  /* All checks passed - proposed vertex is geometrically valid */
  return true;
}

/* Ray-traced reproject for double-bounce manifold path.
 *
 * Validates proposed (u1,v1) and (u2,v2) by ray-tracing the full path:
 * receiver → primary vertex → secondary vertex
 *
 * Per Mitsuba reference, must verify:
 * - Each ray hits the expected shape (same object/prim as previous iteration)
 * - Scattering is geometrically valid at each vertex
 *
 * Like single-bounce, computes 3D positions from UNCLAMPED barycentric coords
 * and returns the hit's actual barycentric coordinates.
 *
 * Returns: true if entire ray-traced path is geometrically valid, false otherwise
 * On success, updates hit coordinates with the intersections' barycentric coords */
ccl_device_inline bool reproject_double_bounce(KernelGlobals kg,
                                               const ShadingPoint &receiver,
                                               const SpecularSurfaceGeometry &primary_geometry,
                                               const SpecularParameters &primary_params,
                                               float proposed_primary_u,
                                               float proposed_primary_v,
                                               int expected_primary_object,
                                               int expected_primary_prim,
                                               float &hit_primary_u,
                                               float &hit_primary_v,
                                               const SpecularSurfaceGeometry &secondary_geometry,
                                               const SpecularParameters &secondary_params,
                                               float proposed_secondary_u,
                                               float proposed_secondary_v,
                                               int expected_secondary_object,
                                               int expected_secondary_prim,
                                               float &hit_secondary_u,
                                               float &hit_secondary_v)
{
  /* === FIRST SEGMENT: receiver → primary vertex === */

  /* Compute proposed primary 3D position */
  const float w1 = 1.0f - proposed_primary_u - proposed_primary_v;
  const float3 proposed_primary_point = primary_geometry.verts[0] * w1 +
                                         primary_geometry.verts[1] * proposed_primary_u +
                                         primary_geometry.verts[2] * proposed_primary_v;

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
  ray1.tmax = distance1 * 1.0001f;
  ray1.time = 0.5f;
  ray1.dP = differential_zero_compact();
  ray1.dD = differential_zero_compact();
  ray1.self.object = OBJECT_NONE;
  ray1.self.prim = PRIM_NONE;
  ray1.self.light_object = OBJECT_NONE;
  ray1.self.light_prim = PRIM_NONE;

  Intersection isect1;
  if (!scene_intersect(kg, &ray1, PATH_RAY_ALL_VISIBILITY, &isect1)) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double FAIL: Primary vertex not reachable\n");
}
    return false;
  }

  /* Verify we hit the expected primary shape */
  if (isect1.object != expected_primary_object || isect1.prim != expected_primary_prim) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double FAIL: Wrong primary surface (expected obj=%d prim=%d, got obj=%d prim=%d)\n",
           expected_primary_object, expected_primary_prim, isect1.object, isect1.prim);
}
    return false;
  }

  /* Extract primary hit's barycentric coordinates */
  hit_primary_u = isect1.u;
  hit_primary_v = isect1.v;

  /* === SECOND SEGMENT: primary vertex → secondary vertex === */

  /* Compute proposed secondary 3D position */
  const float w2 = 1.0f - proposed_secondary_u - proposed_secondary_v;
  const float3 proposed_secondary_point = secondary_geometry.verts[0] * w2 +
                                           secondary_geometry.verts[1] * proposed_secondary_u +
                                           secondary_geometry.verts[2] * proposed_secondary_v;

  /* Use the ACTUAL hit primary position for the second ray segment.
   * Mitsuba uses the reprojected positions for subsequent segments. */
  const float w1_hit = 1.0f - hit_primary_u - hit_primary_v;
  const float3 hit_primary_point = primary_geometry.verts[0] * w1_hit +
                                    primary_geometry.verts[1] * hit_primary_u +
                                    primary_geometry.verts[2] * hit_primary_v;

  /* Ray-trace from actual primary hit toward proposed secondary */
  Ray ray2;
  ray2.P = hit_primary_point;
  const float3 direction2 = proposed_secondary_point - hit_primary_point;
  const float distance2 = len(direction2);
  if (!(distance2 > 1e-6f)) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double FAIL: Primary and secondary too close\n");
}
    return false;
  }
  ray2.D = direction2 / distance2;
  ray2.tmin = 0.0f;
  ray2.tmax = FLT_MAX;  /* Don't limit - let it find whatever it hits */
  ray2.time = 0.5f;
  ray2.dP = differential_zero_compact();
  ray2.dD = differential_zero_compact();
  /* Skip primary surface to avoid self-intersection */
  ray2.self.object = expected_primary_object;
  ray2.self.prim = expected_primary_prim;
  ray2.self.light_object = OBJECT_NONE;
  ray2.self.light_prim = PRIM_NONE;

  Intersection isect2;
  if (!scene_intersect(kg, &ray2, PATH_RAY_ALL_VISIBILITY, &isect2)) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double FAIL: Secondary vertex not reachable from primary\n");
}
    return false;
  }

  /* Verify we hit the expected secondary shape */
  if (isect2.object != expected_secondary_object || isect2.prim != expected_secondary_prim) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
    printf("    reproject_double FAIL: Wrong secondary surface (expected obj=%d prim=%d, got obj=%d prim=%d)\n",
           expected_secondary_object, expected_secondary_prim, isect2.object, isect2.prim);
}
    return false;
  }

  /* Extract secondary hit's barycentric coordinates */
  hit_secondary_u = isect2.u;
  hit_secondary_v = isect2.v;

if (MPG_DEBUG::NEWTON_DETAIL()) {
  printf("    reproject_double SUCCESS: prim(%.6f,%.6f) sec(%.6f,%.6f)\n",
         hit_primary_u, hit_primary_v, hit_secondary_u, hit_secondary_v);
}

  /* All checks passed - proposed path is geometrically valid */
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

  float u = clamp(seed.bary_u, 1e-4f, 1.0f - 1e-4f);
  float v = clamp(seed.bary_v, 1e-4f, 1.0f - 1e-4f);
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

  /* Mitsuba's half-vector constraint: project half-vector onto surface tangent frame.
   * C = [dot(s, h), dot(t, h)] where s,t are surface tangents.
   * This formulation is proven to converge to 1e-4 in Mitsuba. */
  float3 tangent_u, tangent_v;
  if (!build_tangent_basis(eval.dXdu, eval.dXdv, tangent_u, tangent_v)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: Degenerate normals at build_tangent_basis\n");
    printf("  dXdu=[%.6f, %.6f, %.6f] len=%.9f\n", eval.dXdu.x, eval.dXdu.y, eval.dXdu.z, len(eval.dXdu));
    printf("  dXdv=[%.6f, %.6f, %.6f] len=%.9f\n", eval.dXdv.x, eval.dXdv.y, eval.dXdv.z, len(eval.dXdv));
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  /* Compute generalized half-vector (matching Mitsuba) */
  const float3 wi = -eval.dir_ds;
  const float3 wo = eval.dir_sl;
  float h_eta = params.is_refraction ? params.base_eta : 1.0f;
  if (params.is_refraction) {
    /* Use backfacing flag from params for entering/exiting determination.
     * This matches compute_specular() and is robust for thin geometry.
     *
     * CRITICAL: Half-vector eta is INVERSE of Snell's law eta!
     * - Snell's law: entering = 1/IOR, exiting = IOR
     * - Half-vector: entering = IOR, exiting = 1/IOR
     *
     * Therefore, invert when exiting (backfacing). */
    if (params.backfacing) {
      /* Exiting (backfacing): use inverse IOR for half-vector */
      h_eta = 1.0f / fmaxf(h_eta, 1e-6f);
    }
    /* Entering (!backfacing): keep h_eta = base_eta */
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
   * For perfect specular (roughness=0): N = 0, so C = H
   * For glossy (roughness>0): N sampled from microfacet distribution
   * TODO: Implement microfacet-based offset sampling for glossy surfaces */
  const float2 offset_2d = make_float2(0.0f, 0.0f);  /* Zero offset for perfect specular */

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

  for (int iter = 0; iter < options.max_iters; ++iter) {
    /* Now using Mitsuba's half-vector constraint formulation directly.
     * Should converge to 1e-4 like Mitsuba (threshold from their reference). */
    if (residual_norm < 1e-4f) {
      break;
    }

    /* Only recompute Jacobian and solve when needed (avoid redundant computation) */
    if (needs_step_update) {
      float3 J_cols[2];
      compute_halfvector_jacobian(shading_point, seed, geometry, eval, h, h_eta, tangent_u, tangent_v, J_cols);

      /* Embed 2D constraint in 3D vector for solve_step compatibility */
      const float3 residual_3d = make_float3(residual_2d_u, residual_2d_v, 0.0f);

      if (!solve_step(J_cols[0], J_cols[1], residual_3d, delta) ||
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
    float new_u, new_v;
    if (!reproject_single_bounce(kg, shading_point, geometry, params, proposed_u, proposed_v,
                                  seed.object, seed.prim, new_u, new_v)) {
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    /* Use the reprojected barycentric coordinates for subsequent evaluation */
    SpecularParameters new_params;
    if (!specular_parameters_from_surface(kg, sd, geometry, seed, 0, new_u, new_v, new_params)) {
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    SpecularEval new_eval;
    evaluate_specular(shading_point, seed, geometry, new_params, new_u, new_v, new_eval);
    if (new_eval.tir) {
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    /* Compute Mitsuba's half-vector constraint for new evaluation */
    float3 new_tangent_u, new_tangent_v;
    if (!build_tangent_basis(new_eval.dXdu, new_eval.dXdv, new_tangent_u, new_tangent_v)) {
      beta *= 0.5f;
      needs_step_update = false;
      continue;
    }

    const float3 new_wi = -new_eval.dir_ds;
    const float3 new_wo = new_eval.dir_sl;
    float new_h_eta = new_params.is_refraction ? new_params.base_eta : 1.0f;
    if (new_params.is_refraction) {
      /* Use backfacing flag for entering/exiting - see lines 2426-2441. */
      if (new_params.backfacing) {
        new_h_eta = 1.0f / fmaxf(new_h_eta, 1e-6f);
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

    /* Use zero offset for perfect specular (matches Mitsuba for roughness=0) */
    const float2 new_offset_2d = make_float2(0.0f, 0.0f);

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
  result.visibility = compute_visibility(kg, sd, seed, eval);
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
  vertex.object = seed.object;
  vertex.prim = seed.prim;

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

  float residual_matrix[2][2];
  if (!compute_residual_matrix(shading_point, seed, geometry, eval, residual_matrix)) {
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

  float primary_u = clamp(seed.bary_u, 1.0e-4f, 1.0f - 1.0e-4f);
  float primary_v = clamp(seed.bary_v, 1.0e-4f, 1.0f - 1.0e-4f);
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

  float secondary_u = clamp(secondary_seed.bary_u, 1.0e-4f, 1.0f - 1.0e-4f);
  float secondary_v = clamp(secondary_seed.bary_v, 1.0e-4f, 1.0f - 1.0e-4f);
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
      if (!compute_double_bounce_jacobian(kg,
                                          sd,
                                          seed,
                                          secondary_seed,
                                          receiver,
                                          primary_geometry,
                                          secondary_geometry,
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
    float new_primary_u, new_primary_v, new_secondary_u, new_secondary_v;
    if (!reproject_double_bounce(kg, receiver,
                                  primary_geometry, primary_params, proposed_primary_u, proposed_primary_v,
                                  seed.object, seed.prim, new_primary_u, new_primary_v,
                                  secondary_geometry, secondary_params, proposed_secondary_u, proposed_secondary_v,
                                  secondary_seed.object, secondary_seed.prim, new_secondary_u, new_secondary_v)) {
if (MPG_DEBUG::NEWTON_DETAIL()) {
      printf("  Iter %2d: REJECT #0 - reproject failed, beta %.6f→%.6f\n", iter + 1, beta, beta * 0.5f);
}
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    /* Use the reprojected barycentric coordinates for subsequent evaluation */
    SpecularParameters new_primary_params;
    if (!specular_parameters_from_surface(
            kg, sd, primary_geometry, seed, 0, new_primary_u, new_primary_v, new_primary_params))
    {
if (MPG_DEBUG::NEWTON_DETAIL()) {
      printf("  Iter %2d: REJECT #1 - primary params extraction failed, beta %.6f→%.6f\n", iter + 1, beta, beta * 0.5f);
}
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    ShaderData new_primary_sd;
    if (!build_primary_shading_data(sd, primary_geometry, seed, new_primary_u, new_primary_v, new_primary_sd)) {
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
                                          secondary_geometry,
                                          secondary_seed,
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
                                primary_geometry,
                                new_primary_params,
                                secondary_geometry,
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

  if (!specular_parameters_from_surface(kg, sd, primary_geometry, seed, 0, primary_u, primary_v, primary_params)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: specular_parameters_from_surface (post-Newton, primary) (line 2518)\n");
}
    failure_code = MPG_FAILURE_NO_SPECULAR;
    return false;
  }

  if (!build_primary_shading_data(sd, primary_geometry, seed, primary_u, primary_v, primary_sd)) {
if (MPG_DEBUG::PARAMS()) {
    printf("MPG FAILURE: build_primary_shading_data (post-Newton) (line 2523)\n");
}
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  if (!specular_parameters_from_surface(kg,
                                        primary_sd,
                                        secondary_geometry,
                                        secondary_seed,
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
  primary_vertex.object = seed.object;
  primary_vertex.prim = seed.prim;

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
  secondary_vertex.object = secondary_seed.object;
  secondary_vertex.prim = secondary_seed.prim;

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

CCL_NAMESPACE_END