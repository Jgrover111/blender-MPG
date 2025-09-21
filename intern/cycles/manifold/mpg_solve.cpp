/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_solve.h"

#include "kernel/bvh/bvh.h"
#include "kernel/bvh/util.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/integrator/state.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/object.h"
#include "kernel/geom/shader_data.h"
#include "kernel/geom/triangle.h"
#include "kernel/svm/types.h"
#include "kernel/types.h"

#include <cfloat>

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
                                  const float3 &dir_sl)
{
  if (!params.has_microfacet) {
    return zero_spectrum();
  }

  const MicrofacetBsdf &microfacet = params.microfacet;
  const float cos_NI = dot(microfacet.N, dir_ds);
  const float3 incident_dir = params.is_refraction ? dir_sl : -dir_sl;
  const float cos_NO = dot(microfacet.N, incident_dir);

  if (!(fabsf(cos_NI) > 1e-6f && fabsf(cos_NO) > 1e-6f)) {
    return zero_spectrum();
  }

  if (params.is_refraction) {
    if (cos_NI * cos_NO >= 0.0f) {
      return zero_spectrum();
    }
  }
  else {
    if (cos_NI <= 0.0f || cos_NO <= 0.0f) {
      return zero_spectrum();
    }
  }

  Spectrum reflectance = zero_spectrum();
  Spectrum transmittance = zero_spectrum();
  microfacet_fresnel(kg, &microfacet, cos_NI, nullptr, &reflectance, &transmittance);

  const Spectrum fresnel_weight = params.is_refraction ? transmittance : reflectance;
  return microfacet.weight * fresnel_weight;
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
  if (norm_raw == 0.0f) {
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
  float3 dir_out;
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
  if (!params.is_refraction) {
    tir = false;
    cos_theta_i = fabsf(dot(-dir_ds, normal));
    cos_theta_t = cos_theta_i;
    eta_used = 1.0f;
    return reflect_dir(dir_ds, normal);
  }

  float3 oriented_normal = normal;
  float eta = fmaxf(params.base_eta, 1e-6f);
  if (dot(-dir_ds, normal) < 0.0f) {
    oriented_normal = -normal;
    eta = 1.0f / eta;
  }

  float3 dir = refract_dir(dir_ds, oriented_normal, eta, tir, cos_theta_i, cos_theta_t);
  eta_used = eta;
  if (tir) {
    cos_theta_i = fabsf(cos_theta_i);
    cos_theta_t = 0.0f;
    return dir;
  }

  cos_theta_i = fabsf(cos_theta_i);
  cos_theta_t = fabsf(cos_theta_t);
  return dir;
}

float3 derivative_normalized(const float3 &vector, const float3 &d_vector)
{
  const float len_v = len(vector);
  if (len_v == 0.0f) {
    return zero_float3();
  }
  const float3 v_hat = vector / len_v;
  return (d_vector * len_v - vector * (dot(vector, d_vector) / len_v)) / (len_v * len_v);
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

  eval.dir_sl = seed.light_sample.P - eval.point;
  eval.distance_sl = len(eval.dir_sl);
  eval.dir_sl = (eval.distance_sl > 0.0f) ? (eval.dir_sl / eval.distance_sl) :
                                           make_float3(0.0f, 0.0f, 1.0f);

  eval.normal = combine_vertex_normals(geometry, u, v);
  eval.dNdu = compute_normal_derivative(geometry, u, v, eval.normal, true);
  eval.dNdv = compute_normal_derivative(geometry, u, v, eval.normal, false);

  float cos_theta_i = 0.0f, cos_theta_t = 0.0f, eta = 1.0f;
  const float3 spec_dir = compute_specular(
      -eval.dir_ds, eval.normal, params, eval.tir, cos_theta_i, cos_theta_t, eta);
  eval.refractive = params.is_refraction;
  eval.eta = eta;
  eval.cos_theta_i = cos_theta_i;
  eval.cos_theta_t = cos_theta_t;
  eval.residual = eval.dir_sl - spec_dir;
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
  float3 ray_dir = spec_point - sd.P;
  const float distance = len(ray_dir);
  if (!(distance > 1e-6f)) {
    return false;
  }
  ray_dir /= distance;

  Ray ray;
  ray.P = sd.P;
  ray.D = ray_dir;
  ray.tmin = 0.0f;
  ray.tmax = distance;
  ray.time = sd.time;
  ray.self.prim = sd.prim;
  ray.self.object = sd.object;
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
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW>(
      kg, integrator_state, &spec_sd, nullptr, PATH_RAY_DIFFUSE, true);

  const MicrofacetBsdf *reflection_microfacet = nullptr;
  const MicrofacetBsdf *refraction_microfacet = nullptr;

  for (int i = 0; i < spec_sd.num_closure; ++i) {
    const ShaderClosure *closure = &spec_sd.closure[i];
    if (!CLOSURE_IS_BSDF(closure->type)) {
      continue;
    }
    if (!(closure->sample_weight > 0.0f)) {
      continue;
    }

    const bool closure_is_refraction = CLOSURE_IS_REFRACTION(closure->type) ||
                                       CLOSURE_IS_GLASS(closure->type);
    const bool closure_is_reflection = (closure->type == CLOSURE_BSDF_MICROFACET_GGX_ID ||
                                        closure->type == CLOSURE_BSDF_MICROFACET_BECKMANN_ID ||
                                        closure->type == CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID);
    if (!(closure_is_refraction || closure_is_reflection)) {
      continue;
    }

    const MicrofacetBsdf *microfacet = reinterpret_cast<const MicrofacetBsdf *>(closure);
    const float roughness_sq = microfacet->alpha_x * microfacet->alpha_y;
    if (roughness_sq > BSDF_ROUGHNESS_SQ_THRESH) {
      continue;
    }

    if (closure_is_refraction) {
      refraction_microfacet = microfacet;
      break;
    }

    if (reflection_microfacet == nullptr) {
      reflection_microfacet = microfacet;
    }
  }

  const MicrofacetBsdf *microfacet = refraction_microfacet ? refraction_microfacet :
                                                                  reflection_microfacet;
  if (microfacet == nullptr) {
    return false;
  }

  copy_microfacet_to_parameters(microfacet, params);

  params.is_refraction = (microfacet == refraction_microfacet);
  if (params.is_refraction) {
    float eta = microfacet->ior;
    if (fabsf(eta) <= 1e-6f) {
      return false;
    }
    if (spec_sd.flag & SD_BACKFACING) {
      eta = 1.0f / eta;
    }
    params.base_eta = fabsf(eta);
  }
  else {
    params.base_eta = 1.0f;
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
  shadow_ray.tmin = 0.0f;
  shadow_ray.tmax = fmaxf(eval.distance_sl - 1e-4f, 0.0f);
  shadow_ray.time = sd.time;
  shadow_ray.self.prim = seed.prim;
  shadow_ray.self.object = seed.object;
  shadow_ray.self.light_prim = PRIM_NONE;
  shadow_ray.self.light_object = OBJECT_NONE;

  const bool occluded = scene_intersect_shadow(kg, &shadow_ray, PATH_RAY_SHADOW);
  return occluded ? 0.0f : 1.0f;
}

float3 derivative_specular_reflection(const float3 &dir_in,
                                      const float3 &d_dir_in,
                                      const float3 &normal,
                                      const float3 &d_normal)
{
  const float dot_in_n = dot(dir_in, normal);
  const float d_dot = dot(d_dir_in, normal) + dot(dir_in, d_normal);
  const float3 d_reflect = d_dir_in - 2.0f * (d_dot * normal + dot_in_n * d_normal);
  return d_reflect;
}

float3 derivative_specular_refraction(const float3 &dir_in,
                                      const float3 &d_dir_in,
                                      const float3 &normal,
                                      const float3 &d_normal,
                                      const float eta,
                                      const float cos_theta_i,
                                      const float cos_theta_t,
                                      const float sin_theta_i,
                                      const float sin_theta_t)
{
  const float d_cos_theta_i = -dot(d_dir_in, normal) - dot(dir_in, d_normal);
  const float denom = fmaxf(1e-8f, cos_theta_t);
  const float d_cos_theta_t = -(eta * eta * sin_theta_i * d_cos_theta_i) / denom;
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

  const float3 d_dir_sl_du = -derivative_normalized(seed.light_sample.P - eval.point, -geometry.dPdu);
  const float3 d_dir_sl_dv = -derivative_normalized(seed.light_sample.P - eval.point, -geometry.dPdv);

  float3 d_spec_du, d_spec_dv;
  if (!eval.refractive) {
    d_spec_du = derivative_specular_reflection(-eval.dir_ds, -d_dir_ds_du, eval.normal, eval.dNdu);
    d_spec_dv = derivative_specular_reflection(-eval.dir_ds, -d_dir_ds_dv, eval.normal, eval.dNdv);
  }
  else {
    const float sin_theta_i = sqrtf(fmaxf(0.0f, 1.0f - eval.cos_theta_i * eval.cos_theta_i));
    const float sin_theta_t = sqrtf(fmaxf(0.0f, 1.0f - eval.cos_theta_t * eval.cos_theta_t));
    d_spec_du = derivative_specular_refraction(-eval.dir_ds,
                                               -d_dir_ds_du,
                                               eval.normal,
                                               eval.dNdu,
                                               eval.eta,
                                               eval.cos_theta_i,
                                               eval.cos_theta_t,
                                               sin_theta_i,
                                               sin_theta_t);
    d_spec_dv = derivative_specular_refraction(-eval.dir_ds,
                                               -d_dir_ds_dv,
                                               eval.normal,
                                               eval.dNdv,
                                               eval.eta,
                                               eval.cos_theta_i,
                                               eval.cos_theta_t,
                                               sin_theta_i,
                                               sin_theta_t);
  }

  J[0] = d_dir_sl_du - d_spec_du;
  J[1] = d_dir_sl_dv - d_spec_dv;
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

  const float det = a00 * a11 - a01 * a01;
  if (fabsf(det) < 1e-10f) {
    return false;
  }

  delta.x = (-a11 * b0 + a01 * b1) / det;
  delta.y = (a01 * b0 - a00 * b1) / det;
  return true;
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
  if (sum == 0.0f) {
    bary = make_float3(1.0f, 0.0f, 0.0f);
  }
  bary /= sum;
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

}  // namespace

bool mpg_solve_single_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
                             const MpgOptions &options,
                             RNGState &rng_state,
                             MpgSolverOutput &result)
{
  (void)rng_state;
  (void)bsdf;

  result = MpgSolverOutput();

  if (seed.prim < 0 || seed.object < 0) {
    return false;
  }

  SpecularSurfaceGeometry geometry;
  if (!load_surface_geometry(kg, sd, seed, geometry)) {
    return false;
  }

  float u = clamp(seed.bary_u, 1e-4f, 1.0f - 1e-4f);
  float v = clamp(seed.bary_v, 1e-4f, 1.0f - 1e-4f);
  project_barycentrics(u, v);

  SpecularParameters params;
  if (!specular_parameters_from_surface(kg, sd, geometry, seed, u, v, params)) {
    return false;
  }

  float trust_radius = 0.25f;
  float prev_residual = FLT_MAX;
  int increase_counter = 0;

  const ShadingPoint shading_point = shading_point_from_shader_data(sd);

  SpecularEval eval;
  evaluate_specular(shading_point, seed, geometry, params, u, v, eval);
  if (eval.tir) {
    return false;
  }

  float residual_norm = len(eval.residual);

  for (int iter = 0; iter < options.max_iters; ++iter) {
    if (residual_norm < 1e-5f) {
      break;
    }

    float3 J_cols[2];
    compute_jacobian(shading_point, seed, geometry, eval, J_cols);

    float2 delta;
    if (!solve_step(J_cols[0], J_cols[1], eval.residual, delta)) {
      break;
    }

    const float step_norm = len(make_float3(delta.x, delta.y, 0.0f));
    if (step_norm > trust_radius) {
      const float scale = trust_radius / (step_norm + 1e-8f);
      delta *= scale;
    }

    float new_u = u - delta.x;
    float new_v = v - delta.y;
    project_barycentrics(new_u, new_v);

    SpecularEval new_eval;
    evaluate_specular(shading_point, seed, geometry, params, new_u, new_v, new_eval);
    if (new_eval.tir) {
      trust_radius *= 0.5f;
      if (trust_radius < 1e-6f) {
        return false;
      }
      continue;
    }

    const float new_residual_norm = len(new_eval.residual);
    if (new_residual_norm < residual_norm) {
      u = new_u;
      v = new_v;
      eval = new_eval;
      residual_norm = new_residual_norm;
      trust_radius = fminf(trust_radius * 1.5f, 1.0f);
      if (prev_residual - residual_norm < 1e-6f) {
        ++increase_counter;
      }
      else {
        increase_counter = 0;
      }
      prev_residual = residual_norm;
    }
    else {
      trust_radius *= 0.5f;
      if (trust_radius < 1e-6f) {
        return false;
      }
      ++increase_counter;
    }

    if (increase_counter >= 2) {
      break;
    }
  }

  if (residual_norm > 1e-4f) {
    return false;
  }

  result.success = true;
  result.specular_point = eval.point;
  result.specular_normal = eval.normal;
  result.dir_ds = -eval.dir_ds;
  result.dir_sl = eval.dir_sl;
  result.distance_ds = eval.distance_ds;
  result.distance_sl = eval.distance_sl;
  result.dXdu = eval.dXdu;
  result.dXdv = eval.dXdv;
  result.dNdu = eval.dNdu;
  result.dNdv = eval.dNdv;
  result.u = u;
  result.v = v;
  result.visibility = compute_visibility(kg, sd, seed, eval);
  result.wi = normalize(eval.point - shading_point.position);
  result.object = seed.object;
  result.prim = seed.prim;
  result.is_refraction = params.is_refraction;

  result.spec_weight = evaluate_specular_weight(kg, params, result.dir_ds, result.dir_sl);
  if (is_zero(result.spec_weight)) {
    return false;
  }

  float residual_matrix[2][2];
  if (!compute_residual_matrix(shading_point, seed, geometry, eval, residual_matrix)) {
    return false;
  }

  const float determinant =
      residual_matrix[0][0] * residual_matrix[1][1] - residual_matrix[0][1] * residual_matrix[1][0];
  if (!isfinite_safe(determinant) || determinant <= 0.0f) {
    return false;
  }

  const float area_element = len(cross(eval.dXdu, eval.dXdv));
  const float cos_theta = fabsf(dot(eval.normal, -result.wi));
  const float dist2 = fmaxf(result.distance_ds * result.distance_ds, 1e-8f);
  if (area_element <= 0.0f || cos_theta <= 0.0f) {
    return false;
  }

  const float area_to_solid = area_element * cos_theta / dist2;
  if (!isfinite_safe(area_to_solid) || area_to_solid <= 0.0f) {
    return false;
  }

  result.jacobian = fabsf(determinant) * area_to_solid;

  return true;
}

CCL_NAMESPACE_END
