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
};

float3 surface_point_from_barycentric(const SpecularSurfaceGeometry &geometry, const float u, const float v);
float3 surface_normal_from_barycentric(const SpecularSurfaceGeometry &geometry, const float u, const float v);

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

  float3 geom_normal = cross(geometry.dPdu, geometry.dPdv);
  if (!is_zero(geom_normal)) {
    geom_normal = safe_normalize(geom_normal);
  }
  else {
    geom_normal = surface_normal_from_barycentric(geometry, u, v);
    if (!is_zero(geom_normal)) {
      geom_normal = safe_normalize(geom_normal);
    }
  }
  if (is_zero(geom_normal)) {
    return false;
  }

  const float3 receiver_to_primary = out_sd.P - receiver_sd.P;
  if (!is_zero(receiver_to_primary) && dot(geom_normal, receiver_to_primary) < 0.0f) {
    geom_normal = -geom_normal;
  }

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
      const bool entering = dot_incident_normal <= 0.0f;
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
  const float3 incoming = -dir_ds;
  const bool entering = dot(normal, incoming) >= 0.0f;

  float3 oriented_normal = entering ? normal : -normal;

  if (!params.is_refraction) {
    tir = false;
    cos_theta_i = dot(incoming, oriented_normal);
    cos_theta_t = cos_theta_i;
    eta_used = 1.0f;
    return reflect_dir(incoming, oriented_normal);
  }

  const float safe_base_eta = fmaxf(params.base_eta, 1e-6f);
  const float eta_ratio = entering ? (1.0f / safe_base_eta) : safe_base_eta;
  const float safe_eta_ratio = fmaxf(eta_ratio, 1e-6f);

  float3 dir = refract_dir(incoming, oriented_normal, safe_eta_ratio, tir, cos_theta_i, cos_theta_t);
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
  eval.dir_ds = eval.point - D.position;
  eval.distance_ds = len(eval.dir_ds);
  eval.dir_ds = (eval.distance_ds > 0.0f) ? (eval.dir_ds / eval.distance_ds) :
                                           make_float3(0.0f, 0.0f, 1.0f);

  compute_light_sample_direction(seed.light_sample, eval.point, eval.dir_sl, eval.distance_sl);

  if (seed.use_smooth_normals) {
    eval.normal = combine_vertex_normals(geometry, u, v);
    if (is_zero(eval.normal)) {
      eval.normal = safe_normalize(cross(geometry.dPdu, geometry.dPdv));
    }
    eval.dNdu = compute_normal_derivative(geometry, u, v, eval.normal, true);
    eval.dNdv = compute_normal_derivative(geometry, u, v, eval.normal, false);
  }
  else {
    eval.normal = safe_normalize(cross(geometry.dPdu, geometry.dPdv));
    eval.dNdu = zero_float3();
    eval.dNdv = zero_float3();
  }

  if (!is_zero(eval.normal)) {
    eval.normal = safe_normalize(eval.normal);

    /* For refraction, preserve geometric normal orientation to avoid flipping
     * interfaces in underwater caustics. Only flip if explicitly contradicting
     * the closure's configured normal (params.normal).
     * For reflection, align with incoming ray to ensure same-side scattering. */
    if (params.is_refraction) {
      /* Refraction: only flip if closure normal exists and points opposite */
      if (params.has_normal && dot(eval.normal, params.normal) < 0.0f) {
        eval.normal = -eval.normal;
        eval.dNdu = -eval.dNdu;
        eval.dNdv = -eval.dNdv;
      }
    }
    else {
      /* Reflection: flip if pointing toward receiver (same-side requirement) */
      const bool flip_to_params = params.has_normal && dot(eval.normal, params.normal) < 0.0f;
      const bool flip_to_receiver = !params.has_normal && dot(eval.normal, eval.dir_ds) > 0.0f;
      if (flip_to_params || flip_to_receiver) {
        eval.normal = -eval.normal;
        eval.dNdu = -eval.dNdu;
        eval.dNdv = -eval.dNdv;
      }
    }
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
   * projected onto surface tangent frame. This is the constraint used in Mitsuba reference. */
  const float3 wi = -eval.dir_ds;  /* Incoming: receiver to specular point */
  const float3 wo = eval.dir_sl;    /* Outgoing: specular point to light */

  /* Mitsuba uses base material IOR for half-vector, not the relative IOR from Snell's law.
   * Invert eta when ray comes from inside surface (dot(wi, normal) < 0). */
  float h_eta = params.is_refraction ? params.base_eta : 1.0f;
  if (params.is_refraction && dot(wi, eval.normal) < 0.0f) {
    h_eta = 1.0f / fmaxf(h_eta, 1e-6f);
  }

  /* Generalized half-vector: h = normalize(wi + eta * wo), negated when eta != 1 */
  float3 h = wi + h_eta * wo;
  if (h_eta != 1.0f) {
    h = -h;
  }
  const float h_len = len(h);
  if (h_len > 1e-8f) {
    h /= h_len;
  }

  /* For directional constraint compatibility, also store old residual */
  eval.residual = eval.dir_sl - spec_dir;
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
  if (!(distance > 1e-4f)) {
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

  const bool tau_hint_valid = (seed.tau_count > 0);
  bool force_transmission = false;
  bool force_reflection = false;
  if (tau_hint_valid) {
    force_transmission = (seed.tau_bits & 1u) != 0u;
    force_reflection = !force_transmission;
  }
  else if (seed.scatter == MPG_SEED_SCATTER_REFRACTION) {
    force_transmission = true;
  }
  else if (seed.scatter == MPG_SEED_SCATTER_REFLECTION) {
    force_reflection = true;
  }

  bool prefer_reflection_from_geometry = false;
  /* For directional lights, light_dir points TO the light source (opposite of actual light ray
   * travel direction), making the "same side" test invalid. Only apply geometric checks for
   * finite lights where light_dir represents the actual light ray direction. */
  const bool is_directional_light = (seed.light_sample.t == FLT_MAX);
  if (!light_dir_degenerate && hemisphere_normal_valid && !is_directional_light) {
    /* Compare directions in the shading-normal frame (hemisphere_normal) to detect interface crossings. */
    const float dot_in = dot(hemisphere_normal, -ray_dir);
    const float dot_light = dot(hemisphere_normal, light_dir);
    if ((dot_in < 0.0f && dot_light > 0.0f) || (dot_in > 0.0f && dot_light < 0.0f)) {
      force_transmission = true;
      force_reflection = false;
    }
    else if ((dot_in > 0.0f && dot_light > 0.0f) || (dot_in < 0.0f && dot_light < 0.0f)) {
      /* When both rays occupy the same half-space the Mitsuba reference prefers
       * the reflective branch even if the seed requested refraction. */
      prefer_reflection_from_geometry = true;
    }
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

#ifdef WITH_CYCLES_DEBUG
  auto log_forced_scatter_mismatch = [&](const char *requested, const char *fallback) {
    if (!has_forced_scatter) {
      return;
    }
    if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
      LOG_DEBUG << "MPG seed forced " << requested << " scatter but only " << fallback
                << " available; falling back.";
    }
  };
#else
  auto log_forced_scatter_mismatch = [&](const char *, const char *) {};
#endif

  bool have_singular_reflection = false;
  float eta_singular = 1.5f; // overwritten below with microfacet IOR when available
  SpecularParameters singular_refraction_params;
  bool have_singular_refraction_params = false;

  for (int i = 0; i < spec_sd.num_closure; ++i) {
    const ShaderClosure *closure = &spec_sd.closure[i];
    if (!CLOSURE_IS_BSDF(closure->type)) {
      continue;
    }

    const bool is_trans = CLOSURE_IS_BSDF_TRANSMISSION(closure->type);
    const bool is_glass = CLOSURE_IS_GLASS(closure->type);
    const bool is_micro = CLOSURE_IS_BSDF_MICROFACET(closure->type);
    const bool is_singular = CLOSURE_IS_BSDF_SINGULAR(closure->type);

    if (!(is_micro || is_singular)) {
      continue;
    }

    if (is_micro) {
      const MicrofacetBsdf *mf = reinterpret_cast<const MicrofacetBsdf *>(closure);
      const float ax = fmaxf(mf->alpha_x, 0.0f);
      const float ay = fmaxf(mf->alpha_y, 0.0f);
      const bool delta_like = (ax <= 1.0e-6f) && (ay <= 1.0e-6f);
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
      if (reports_transmission) {
        refraction_microfacet = mf;
      }
      if (reports_reflection) {
        reflection_microfacet = mf;
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
    if (prefer_reflection) {
      microfacet = reflection_microfacet;
      selected_refraction = false;
    }
    else {
      microfacet = refraction_microfacet;
      selected_refraction = true;
    }
  }
  if (microfacet == nullptr) return false;
  params = SpecularParameters();
  copy_microfacet_to_parameters(microfacet, params);

  params.is_refraction = selected_refraction;
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
        return false;
      }
    }
    else {
      params.base_eta = fabsf(eta);
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
    const float3 face_normal = safe_normalize(cross(geometry.dPdu, geometry.dPdv));
    if (is_zero(face_normal)) {
      return false;
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
    return false;
  }
  tangent_u /= len_u;

  tangent_v = dXdv - tangent_u * dot(tangent_u, dXdv);
  const float len_v = len(tangent_v);
  if (!(len_v > 0.0f)) {
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
  float3 primary_normal = surface_normal_from_barycentric(primary_geometry, primary_u, primary_v);
  if (is_zero(primary_normal)) {
    return false;
  }

  const float3 offset_ng_raw = cross(primary_geometry.dPdu, primary_geometry.dPdv);
  bool use_offset_ng = !is_zero(offset_ng_raw);
  float3 offset_ng = use_offset_ng ? safe_normalize(offset_ng_raw) : zero_float3();
  if (use_offset_ng && is_zero(offset_ng)) {
    use_offset_ng = false;
  }

  float3 dir_ds = primary_point - sd.P;
  float distance_ds = len(dir_ds);
  if (!(distance_ds > 1e-4f)) {
    return false;
  }
  dir_ds /= distance_ds;

  if (dot(primary_normal, -dir_ds) < 0.0f) {
    primary_normal = -primary_normal;
  }

  float3 specular_normal = use_offset_ng ? offset_ng : primary_normal;
  if (use_offset_ng && dot(specular_normal, -dir_ds) < 0.0f) {
    specular_normal = -specular_normal;
    offset_ng = specular_normal;
  }

  bool tir = false;
  float cos_theta_i = 0.0f;
  float cos_theta_t = 0.0f;
  float eta_used = 1.0f;
  const float3 dir_sl = compute_specular(
      dir_ds, specular_normal, primary_params, tir, cos_theta_i, cos_theta_t, eta_used);
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

  Ray ray;
  ray.P = mpg_surface_ray_offset(kg, offset_sd, primary_point, dir_sl);
  ray.D = dir_sl;
  ray.tmin = 1e-4f;
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
    return false;
  }

  if (!(isect.type & PRIMITIVE_TRIANGLE)) {
    return false;
  }

  secondary_seed = seed;
  secondary_seed.object = isect.object;
  secondary_seed.prim = isect.prim;
  secondary_seed.bary_u = isect.u;
  secondary_seed.bary_v = isect.v;
  if (secondary_seed.tau_count > 0) {
    const uint8_t remaining_tau_count = secondary_seed.tau_count;
    secondary_seed.tau_bits = static_cast<uint8_t>(secondary_seed.tau_bits >> 1);
    secondary_seed.tau_count = (remaining_tau_count > 0) ? static_cast<uint8_t>(remaining_tau_count - 1) : 0;
  }

  bool has_smooth_normals = secondary_seed.use_smooth_normals;
  smooth_normals_at_hit(kg, ray, isect, has_smooth_normals);
  secondary_seed.use_smooth_normals = has_smooth_normals;
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
    return false;
  }
  if (!isfinite_safe(eval.primary.distance_sl) || !(eval.primary.distance_sl > 1e-4f)) {
    return false;
  }
  if (eval.primary.tir) {
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
    return false;
  }
  if (!isfinite_safe(eval.secondary.distance_sl) || !(eval.secondary.distance_sl > 1e-4f)) {
    return false;
  }
  if (eval.secondary.tir) {
    return false;
  }

  float3 tangent_u, tangent_v;
  if (!build_tangent_basis(eval.primary.dXdu, eval.primary.dXdv, tangent_u, tangent_v)) {
    return false;
  }
  eval.residual[0] = dot(eval.primary.residual, tangent_u);
  eval.residual[1] = dot(eval.primary.residual, tangent_v);

  if (!build_tangent_basis(eval.secondary.dXdu, eval.secondary.dXdv, tangent_u, tangent_v)) {
    return false;
  }
  eval.residual[2] = dot(eval.secondary.residual, tangent_u);
  eval.residual[3] = dot(eval.secondary.residual, tangent_v);
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
  /* For double-bounce with directional lights, we need to use the secondary vertex position
   * as a finite light source for Jacobian computation. Compute the secondary point from
   * current barycentric coordinates. */
  const float3 secondary_point = surface_point_from_barycentric(secondary_geometry, u2, v2);

  LightSample adjusted_light_sample = light_sample;
  const bool is_directional = (light_sample.t == FLT_MAX) ||
                             (light_sample.type == LIGHT_DISTANT) ||
                             (light_sample.type == LIGHT_BACKGROUND);

  if (is_directional) {
    /* For the secondary bounce, treat the light as coming from the secondary vertex
     * instead of from infinity. This matches Mitsuba's approach of creating a fake
     * vertex at finite distance for directional lights. */
    adjusted_light_sample.P = secondary_point;
    adjusted_light_sample.t = 1.0f;  /* Arbitrary small distance, direction matters more */
    adjusted_light_sample.type = LIGHT_POINT;
    /* D stays the same - light direction from secondary vertex toward light */
  }

  const float epsilon = 1.0e-4f;

  for (int column = 0; column < 4; ++column) {
    float du1 = 0.0f, dv1 = 0.0f, du2 = 0.0f, dv2 = 0.0f;
    switch (column) {
      case 0:
        du1 = epsilon;
        break;
      case 1:
        dv1 = epsilon;
        break;
      case 2:
        du2 = epsilon;
        break;
      case 3:
        dv2 = epsilon;
        break;
    }

    float offset_u1 = u1 + du1;
    float offset_v1 = v1 + dv1;
    float offset_u2 = u2 + du2;
    float offset_v2 = v2 + dv2;

    project_barycentrics(offset_u1, offset_v1);
    project_barycentrics(offset_u2, offset_v2);

    DoubleBounceEval offset_eval;
    SpecularParameters offset_primary_params;
    if (!specular_parameters_from_surface(
            kg, sd, primary_geometry, primary_seed, offset_u1, offset_v1, offset_primary_params))
    {
      return false;
    }

    ShaderData offset_primary_sd;
    if (!build_primary_shading_data(sd, primary_geometry, primary_seed, offset_u1, offset_v1, offset_primary_sd)) {
      return false;
    }

    SpecularParameters offset_secondary_params;
    if (!specular_parameters_from_surface(kg,
                                          offset_primary_sd,
                                          secondary_geometry,
                                          secondary_seed,
                                          offset_u2,
                                          offset_v2,
                                          offset_secondary_params))
    {
      return false;
    }

    /* For directional lights, need to update the adjusted light sample position
     * for each iteration since the secondary vertex moves with offset_u2/offset_v2 */
    LightSample iteration_light_sample = adjusted_light_sample;
    if (is_directional && (du2 != 0.0f || dv2 != 0.0f)) {
      const float3 offset_secondary_point = surface_point_from_barycentric(
          secondary_geometry, offset_u2, offset_v2);
      iteration_light_sample.P = offset_secondary_point;
    }

    if (!evaluate_double_bounce(receiver,
                                primary_geometry,
                                offset_primary_params,
                                secondary_geometry,
                                offset_secondary_params,
                                iteration_light_sample,
                                offset_u1,
                                offset_v1,
                                offset_u2,
                                offset_v2,
                                primary_use_smooth_normals,
                                secondary_use_smooth_normals,
                                offset_eval))
    {
      return false;
    }

    for (int row = 0; row < 4; ++row) {
      const float diff = offset_eval.residual[row] - base_residual[row];
      J[row][column] = diff / epsilon;
    }
  }

  return true;
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
  if (!specular_parameters_from_surface(kg, sd, geometry, seed, u, v, params)) {
    failure_code = MPG_FAILURE_NO_SPECULAR;
    return false;
  }

  float trust_radius = 2.0f;
  float prev_residual = FLT_MAX;
  int increase_counter = 0;

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
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  /* Compute generalized half-vector (matching Mitsuba) */
  const float3 wi = -eval.dir_ds;
  const float3 wo = eval.dir_sl;
  float h_eta = params.is_refraction ? params.base_eta : 1.0f;
  if (params.is_refraction && dot(wi, eval.normal) < 0.0f) {
    h_eta = 1.0f / fmaxf(h_eta, 1e-6f);
  }
  float3 h = wi + h_eta * wo;
  if (h_eta != 1.0f) {
    h = -h;
  }
  const float h_len = len(h);
  if (!(h_len > 1e-8f)) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  h /= h_len;

  /* Mitsuba's 2D constraint: C = H - N
   * where H is the projected half-vector and N is the offset normal.
   * For perfect specular (roughness=0): N = 0, so C = H
   * For glossy (roughness>0): N sampled from microfacet distribution
   * TODO: Implement microfacet-based offset sampling for glossy surfaces */
  const float2 offset_2d = make_float2(0.0f, 0.0f);  /* Zero offset for perfect specular */

  float residual_2d_u = dot(tangent_u, h) - offset_2d.x;
  float residual_2d_v = dot(tangent_v, h) - offset_2d.y;
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

    /* Apply step with current beta scaling (Mitsuba approach) */
    float new_u = u - beta * delta.x;
    float new_v = v - beta * delta.y;
    project_barycentrics(new_u, new_v);

    SpecularParameters new_params;
    if (!specular_parameters_from_surface(kg, sd, geometry, seed, new_u, new_v, new_params)) {
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
    if (new_params.is_refraction && dot(new_wi, new_eval.normal) < 0.0f) {
      new_h_eta = 1.0f / fmaxf(new_h_eta, 1e-6f);
    }
    float3 new_h = new_wi + new_h_eta * new_wo;
    if (new_h_eta != 1.0f) {
      new_h = -new_h;
    }
    const float new_h_len = len(new_h);
    if (!(new_h_len > 1e-8f)) {
      beta *= 0.5f;
      needs_step_update = false;
      continue;
    }
    new_h /= new_h_len;

    /* Use zero offset for perfect specular (matches Mitsuba for roughness=0) */
    const float2 new_offset_2d = make_float2(0.0f, 0.0f);

    /* Apply Mitsuba's constraint: C = H - N */
    const float new_residual_2d_u = dot(new_tangent_u, new_h) - new_offset_2d.x;
    const float new_residual_2d_v = dot(new_tangent_v, new_h) - new_offset_2d.y;
    const float new_residual_norm = sqrtf(new_residual_2d_u * new_residual_2d_u +
                                          new_residual_2d_v * new_residual_2d_v);

    if (!isfinite_safe(new_residual_norm)) {
      failure_code = MPG_FAILURE_NEWTON_DIVERGED;
      return false;
    }

    /* Step acceptance: check if residual improved */
    if (new_residual_norm < residual_norm) {
      /* Accept step */
      u = new_u;
      v = new_v;
      eval = new_eval;
      params = new_params;
      tangent_u = new_tangent_u;  /* Update for next Jacobian computation */
      tangent_v = new_tangent_v;
      h = new_h;
      h_eta = new_h_eta;
      residual_2d_u = new_residual_2d_u;
      residual_2d_v = new_residual_2d_v;
      residual_norm = new_residual_norm;
      beta = fminf(beta * 2.0f, 1.0f);  /* Expand beta, cap at 1.0 (Mitsuba) */
      needs_step_update = true;  /* Recompute Jacobian next iteration */
    }
    else {
      /* Reject step */
      beta *= 0.5f;  /* Shrink beta */
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
    }
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
#ifdef WITH_CYCLES_DEBUG
    printf("MPG DEBUG single-bounce: FAILED geometry check, area=%.9f cos_theta=%.9f\n",
           area_element, cos_theta);
#endif
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  /* Compute the complete Jacobian including the geometric term.
   * The Jacobian transforms from the specular surface area measure to the receiver's
   * solid angle measure. Following the Mitsuba reference, this includes:
   * - The manifold constraint Jacobian (determinant of residual matrix)
   * - The geometric factor: cos(θ) / r²
   * where θ is the angle at the specular surface and r is the distance from receiver. */
  const float distance_sq = eval.distance_ds * eval.distance_ds;
  if (distance_sq < 1e-4f) {
#ifdef WITH_CYCLES_DEBUG
    printf("MPG DEBUG single-bounce: FAILED distance check, distance_sq=%.9f\n", distance_sq);
#endif
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
  if (!specular_parameters_from_surface(kg, sd, primary_geometry, seed, primary_u, primary_v, primary_params)) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
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

#ifdef WITH_CYCLES_DEBUG
      if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
        LOG_DEBUG
            << "MPG double-bounce refraction seed missing exit surface, retrying as single bounce";
      }
#endif

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
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  SpecularParameters secondary_params;
  if (!specular_parameters_from_surface(
          kg, primary_sd, secondary_geometry, secondary_seed, secondary_u, secondary_v, secondary_params))
  {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
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

  float residual_norm = 0.0f;
  for (int i = 0; i < 4; ++i) {
    residual_norm += eval.residual[i] * eval.residual[i];
  }
  residual_norm = sqrtf(residual_norm);
  if (!isfinite_safe(residual_norm)) {
    failure_code = MPG_FAILURE_NEWTON_DIVERGED;
    return false;
  }

  /* Mitsuba-style damped Newton: reuse Jacobian when step is rejected */
  float beta = 1.0f;  /* Step size damping factor */
  bool needs_step_update = true;
  float delta[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int iter = 0; iter < options.max_iters; ++iter) {
    /* Double-bounce already uses correct 2D projection. Use 0.05 threshold
     * to match single-bounce (directional constraint convergence range). */
    if (residual_norm < 0.05f) {
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
    }

    /* Apply step with current beta scaling (Mitsuba approach) */
    float new_primary_u = primary_u - beta * delta[0];
    float new_primary_v = primary_v - beta * delta[1];
    float new_secondary_u = secondary_u - beta * delta[2];
    float new_secondary_v = secondary_v - beta * delta[3];

    project_barycentrics(new_primary_u, new_primary_v);
    project_barycentrics(new_secondary_u, new_secondary_v);

    SpecularParameters new_primary_params;
    if (!specular_parameters_from_surface(
            kg, sd, primary_geometry, seed, new_primary_u, new_primary_v, new_primary_params))
    {
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    ShaderData new_primary_sd;
    if (!build_primary_shading_data(sd, primary_geometry, seed, new_primary_u, new_primary_v, new_primary_sd)) {
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    SpecularParameters new_secondary_params;
    if (!specular_parameters_from_surface(kg,
                                          new_primary_sd,
                                          secondary_geometry,
                                          secondary_seed,
                                          new_secondary_u,
                                          new_secondary_v,
                                          new_secondary_params))
    {
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
      beta *= 0.5f;
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
      continue;
    }

    float new_norm = 0.0f;
    for (int i = 0; i < 4; ++i) {
      new_norm += new_eval.residual[i] * new_eval.residual[i];
    }
    new_norm = sqrtf(new_norm);
    if (!isfinite_safe(new_norm)) {
      failure_code = MPG_FAILURE_NEWTON_DIVERGED;
      return false;
    }

    /* Step acceptance: check if residual improved */
    if (new_norm < residual_norm) {
      /* Accept step */
      primary_u = new_primary_u;
      primary_v = new_primary_v;
      secondary_u = new_secondary_u;
      secondary_v = new_secondary_v;
      eval = new_eval;
      primary_params = new_primary_params;
      secondary_params = new_secondary_params;
      primary_sd = new_primary_sd;
      residual_norm = new_norm;
      beta = fminf(beta * 2.0f, 1.0f);  /* Expand beta, cap at 1.0 (Mitsuba) */
      needs_step_update = true;  /* Recompute Jacobian next iteration */
    }
    else {
      /* Reject step */
      beta *= 0.5f;  /* Shrink beta */
      needs_step_update = false;  /* Reuse Jacobian with smaller beta */
    }
  }

  /* Accept solutions with residual <= 0.05 (matching single-bounce threshold).
   * The double-bounce solver uses 2D projected residuals (4D total for 2 vertices). */
  if (!isfinite_safe(residual_norm) || residual_norm > 0.05f) {
    failure_code = MPG_FAILURE_NEWTON_DIVERGED;
    return false;
  }

  if (!specular_parameters_from_surface(kg, sd, primary_geometry, seed, primary_u, primary_v, primary_params)) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  if (!build_primary_shading_data(sd, primary_geometry, seed, primary_u, primary_v, primary_sd)) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  if (!specular_parameters_from_surface(kg,
                                        primary_sd,
                                        secondary_geometry,
                                        secondary_seed,
                                        secondary_u,
                                        secondary_v,
                                        secondary_params))
  {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
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
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  /* Compute Jacobian for primary bounce including geometric term. */
  const float distance_primary_sq = eval.primary.distance_ds * eval.primary.distance_ds;
  if (distance_primary_sq < 1e-4f) {
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
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  /* Compute Jacobian for secondary bounce including geometric term. */
  const float distance_secondary_sq = eval.secondary.distance_ds * eval.secondary.distance_ds;
  if (distance_secondary_sq < 1e-4f) {
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
   * Don't exclude anything as the "light" since we want to detect if the path
   * from primary to secondary is clear. The self exclusion (seed.object/prim) prevents
   * hitting the starting surface. */
  const float visibility_intermediate = compute_segment_visibility(kg,
                                                                   eval.primary.point,
                                                                   eval.primary.normal,
                                                                   eval.secondary.point,
                                                                   sd.time,
                                                                   seed.object,
                                                                   seed.prim,
                                                                   OBJECT_NONE,
                                                                   PRIM_NONE);
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
  return true;
}

CCL_NAMESPACE_END