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

#include <algorithm>
#include <cfloat>
#include <cmath>

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

static inline float pow5f(float x) { const float x2=x*x; return x2*x2*x; }

Spectrum evaluate_specular_weight(KernelGlobals kg,
                                  const SpecularParameters &params,
                                  const float3 &dir_ds,
                                  const float3 &dir_sl)
{
  if (params.has_microfacet) {
    const MicrofacetBsdf &mf = params.microfacet;
    const float cos_NI = dot(mf.N, dir_ds);
    const float3 incident_dir = params.is_refraction ? dir_sl : -dir_sl;
    const float cos_NO = dot(mf.N, incident_dir);

    if (!(fabsf(cos_NI) > 1e-7f && fabsf(cos_NO) > 1e-7f)) {
      return zero_spectrum();
    }
    if (params.is_refraction) {
      if (cos_NI * cos_NO >= 0.0f) return zero_spectrum();
    } else {
      if (cos_NI <= 0.0f || cos_NO <= 0.0f) return zero_spectrum();
    }

    Spectrum F_refl=zero_spectrum(), F_trans=zero_spectrum();
    microfacet_fresnel(kg, &mf, cos_NI, nullptr, &F_refl, &F_trans);
    const Spectrum fw = params.is_refraction ? F_trans : F_refl;
    return mf.weight * fw;
  }
  else {
    const float n = params.base_eta;
    const float F0 = sqr((n - 1.0f) / (n + 1.0f));

    const float c = 0.5f;
    const float F = F0 + (1.0f - F0) * pow5f(1.0f - c);

    return params.is_refraction ? make_spectrum(1.0f - F) : make_spectrum(F);
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

  eval.normal = safe_normalize(cross(geometry.dPdu, geometry.dPdv));
  eval.dNdu = zero_float3();
  eval.dNdv = zero_float3();

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
  float3 n_off = (dot(sd.Ng, ray_dir) >= 0.0f) ? sd.Ng : -sd.Ng;
  ray.P = ray_offset(sd.P, n_off);
  ray.D = ray_dir;
  ray.tmin = 0.0f;
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

  const MicrofacetBsdf *reflection_microfacet = nullptr;
  const MicrofacetBsdf *refraction_microfacet = nullptr;

  bool have_singular_refraction = false;
  bool have_singular_reflection = false;
  float eta_singular = 1.5f; // overwritten below with microfacet IOR when available

  for (int i = 0; i < spec_sd.num_closure; ++i) {
    const ShaderClosure *closure = &spec_sd.closure[i];
    if (!CLOSURE_IS_BSDF(closure->type)) {
      continue;
    }

    const bool is_trans = CLOSURE_IS_BSDF_TRANSMISSION(closure->type);
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
        if (is_trans) {
          have_singular_refraction = true;
        }
        else {
          have_singular_reflection = true;
        }
        if (mf->ior > 0.0f) {
          eta_singular = mf->ior;
        }
      }
      if (is_trans) {
        refraction_microfacet = mf;
      } else {
        reflection_microfacet = mf;
      }
    }
    if (is_singular) {
      if (is_trans) {
        params = SpecularParameters();
        params.has_microfacet = false;
        params.is_refraction = true;

        float eta = 1.45f;
        if (refraction_microfacet) {
          eta = fmaxf(1.0e-6f, refraction_microfacet->ior);
        }
        if (spec_sd.flag &SD_BACKFACING) eta = (eta > 1e-6f) ? 1.0f / eta : eta;
        params.base_eta = fabsf(eta);
        return true;
      }
      have_singular_reflection = true;
      continue;
    }
  }
  if (have_singular_reflection) {
    params = SpecularParameters();
    params.has_microfacet = false;
    params.is_refraction  = false;
    params.base_eta       = 1.0f;
    return true;
  }
  const MicrofacetBsdf *microfacet = refraction_microfacet ? refraction_microfacet : reflection_microfacet;
  if (microfacet == nullptr) return false;
  params = SpecularParameters();
  copy_microfacet_to_parameters(microfacet, params);

  params.is_refraction = (microfacet == refraction_microfacet);
  if (params.is_refraction) {
    float eta = microfacet->ior;
    if (fabsf(eta) <= 1e-6f) return false;
    if (spec_sd.flag & SD_BACKFACING) eta = 1.0f / eta;
    params.base_eta = fabsf(eta);
  }
  else {
    params.base_eta = 1.0f;
  }
  params.microfacet.N = normalize(params.microfacet.N);
  {
    const float3 spec_point = geometry.verts[0] * (1.0f - u - v) +
                              geometry.verts[1] * u +
                              geometry.verts[2] * v;
    const float3 dir_ds = normalize(spec_point - sd.P);
    const float3 dir_sl = normalize(seed.light_sample.P - spec_point);
    const float3 incident  = dir_ds;
    const float3 outgoing  = params.is_refraction ? dir_sl : -dir_sl;
    const float s = dot(params.microfacet.N, incident) * dot(params.microfacet.N, outgoing);
    const bool bad_refraction =  params.is_refraction ? (s > 0.0f) : false;
    const bool bad_reflection = !params.is_refraction ? (s < 0.0f) : false;
    if (bad_refraction || bad_reflection) {
      params.microfacet.N = -params.microfacet.N;
    }
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
                                      const float sin_theta_i)
{
  const float d_cos_theta_i = -dot(d_dir_in, normal) - dot(dir_in, d_normal);
  const float denom = fmaxf(1e-8f, cos_theta_t);
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

  const float3 d_dir_sl_du = derivative_normalized(seed.light_sample.P - eval.point, -geometry.dPdu);
  const float3 d_dir_sl_dv = derivative_normalized(seed.light_sample.P - eval.point, -geometry.dPdv);

  float3 d_spec_du, d_spec_dv;
  if (!eval.refractive) {
    d_spec_du = derivative_specular_reflection(-eval.dir_ds, -d_dir_ds_du, eval.normal, eval.dNdu);
    d_spec_dv = derivative_specular_reflection(-eval.dir_ds, -d_dir_ds_dv, eval.normal, eval.dNdv);
  }
  else {
    const float sin_theta_i = sqrtf(fmaxf(0.0f, 1.0f - eval.cos_theta_i * eval.cos_theta_i));
    d_spec_du = derivative_specular_refraction(-eval.dir_ds,
                                               -d_dir_ds_du,
                                               eval.normal,
                                               eval.dNdu,
                                               eval.eta,
                                               eval.cos_theta_i,
                                               eval.cos_theta_t,
                                               sin_theta_i);
    d_spec_dv = derivative_specular_refraction(-eval.dir_ds,
                                               -d_dir_ds_dv,
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
  if (!(isfinite_safe(det)) || fabsf(det) < 1e-12f) {
    // Diagonal damping proportional to trace for scale invariance
    const float trace = a00 + a11 + 1e-20f;
    const float lambda = 1e-6f * trace;

    const float a00d = a00 + lambda;
    const float a11d = a11 + lambda;
    det = a00d * a11d - a01 * a01;

    if (!(isfinite_safe(det)) || fabsf(det) < 1e-20f) {
      return false;
    }

    delta.x = (-a11 * b0 + a01 * b1) / det;
    delta.y = (a01 * b0 - a00 * b1) / det;
  }
  else {
    delta.x = (-a11 * b0 + a01 * b1) / det;
    delta.y = (a01 * b0 - a00 * b1) / det;
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

  float3 dir_ds = sd.P - primary_point;
  float distance_ds = len(dir_ds);
  if (!(distance_ds > 1e-6f)) {
    return false;
  }
  dir_ds /= distance_ds;

  if (dot(primary_normal, -dir_ds) < 0.0f) {
    primary_normal = -primary_normal;
  }

  bool tir = false;
  float cos_theta_i = 0.0f;
  float cos_theta_t = 0.0f;
  float eta_used = 1.0f;
  const float3 dir_sl = compute_specular(dir_ds, primary_normal, primary_params, tir, cos_theta_i, cos_theta_t, eta_used);
  (void)cos_theta_i;
  (void)cos_theta_t;
  (void)eta_used;
  if (tir || is_zero(dir_sl)) {
    return false;
  }

  Ray ray;
  float3 offset_n = primary_normal;
  if (dot(offset_n, dir_sl) < 0.0f) {
    offset_n = -offset_n;
  }
  ray.P = ray_offset(primary_point, offset_n);
  ray.D = dir_sl;
  ray.tmin = 0.0f;
  const float light_distance = len(seed.light_sample.P - primary_point);
  ray.tmax = (std::isfinite(light_distance) && light_distance > 0.0f) ? light_distance : FLT_MAX;
  ray.time = sd.time;
  ray.self.prim = seed.prim;
  ray.self.object = seed.object;
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
  secondary_seed.use_smooth_normals = false;
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
                            const float3 &light_point,
                            const float u1,
                            const float v1,
                            const float u2,
                            const float v2,
                            DoubleBounceEval &eval)
{
  const float3 secondary_point = surface_point_from_barycentric(secondary_geometry, u2, v2);

  MpgSeedRay primary_seed = {};
  primary_seed.light_sample.P = secondary_point;

  evaluate_specular(receiver, primary_seed, primary_geometry, primary_params, u1, v1, eval.primary);
  if (!isfinite_safe(eval.primary.distance_ds) || !(eval.primary.distance_ds > 1e-6f)) {
    return false;
  }
  if (!isfinite_safe(eval.primary.distance_sl) || !(eval.primary.distance_sl > 1e-6f)) {
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
  secondary_seed.light_sample.P = light_point;

  evaluate_specular(intermediate_point, secondary_seed, secondary_geometry, secondary_params, u2, v2, eval.secondary);
  if (!isfinite_safe(eval.secondary.distance_ds) || !(eval.secondary.distance_ds > 1e-6f)) {
    return false;
  }
  if (!isfinite_safe(eval.secondary.distance_sl) || !(eval.secondary.distance_sl > 1e-6f)) {
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

bool compute_double_bounce_jacobian(const ShadingPoint &receiver,
                                    const SpecularSurfaceGeometry &primary_geometry,
                                    const SpecularParameters &primary_params,
                                    const SpecularSurfaceGeometry &secondary_geometry,
                                    const SpecularParameters &secondary_params,
                                    const float3 &light_point,
                                    const float u1,
                                    const float v1,
                                    const float u2,
                                    const float v2,
                                    const float base_residual[4],
                                    float J[4][4])
{
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
    if (!evaluate_double_bounce(receiver,
                                primary_geometry,
                                primary_params,
                                secondary_geometry,
                                secondary_params,
                                light_point,
                                offset_u1,
                                offset_v1,
                                offset_u2,
                                offset_v2,
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
                                 const int skip_prim)
{
  float3 dir = end_point - start_point;
  const float distance = len(dir);
  if (!(distance > 1e-6f)) {
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
  ray.tmin = 0.0f;
  ray.tmax = fmaxf(distance - 1.0e-4f, 0.0f);
  ray.time = time;
  ray.self.prim = skip_prim;
  ray.self.object = skip_object;
  ray.self.light_prim = PRIM_NONE;
  ray.self.light_object = OBJECT_NONE;

  const bool occluded = scene_intersect_shadow(kg, &ray, PATH_RAY_SHADOW);
  return occluded ? 0.0f : 1.0f;
}

}  // namespace

bool mpg_solve_single_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
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

  float trust_radius = 0.25f;
  float prev_residual = FLT_MAX;
  int increase_counter = 0;

  const ShadingPoint shading_point = shading_point_from_shader_data(sd);

  SpecularEval eval;
  evaluate_specular(shading_point, seed, geometry, params, u, v, eval);
  if (eval.tir) {
    failure_code = MPG_FAILURE_TOTAL_INTERNAL_REFLECTION;
    return false;
  }

  float residual_norm = len(eval.residual);
  if (!isfinite_safe(residual_norm)) {
    failure_code = MPG_FAILURE_NEWTON_DIVERGED;
    return false;
  }

  for (int iter = 0; iter < options.max_iters; ++iter) {
    if (residual_norm < 1e-5f) {
      break;
    }

    float3 J_cols[2];
    compute_jacobian(shading_point, seed, geometry, eval, J_cols);

    float2 delta;
    if (!solve_step(J_cols[0], J_cols[1], eval.residual, delta) ||
        !isfinite_safe(delta.x) ||
        !isfinite_safe(delta.y))
    {
      if (failure_code == MPG_FAILURE_NONE) {
        failure_code = MPG_FAILURE_JACOBIAN_ZERO;
      }
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
        failure_code = MPG_FAILURE_TOTAL_INTERNAL_REFLECTION;
        return false;
      }
      continue;
    }

    const float new_residual_norm = len(new_eval.residual);
    if (!isfinite_safe(new_residual_norm)) {
      failure_code = MPG_FAILURE_NEWTON_DIVERGED;
      return false;
    }

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
        failure_code = (failure_code != MPG_FAILURE_NONE) ? failure_code : MPG_FAILURE_NEWTON_DIVERGED;
        return false;
      }
      ++increase_counter;
    }

    if (increase_counter >= 2) {
      break;
    }
  }

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
  vertex.dir_in = eval.dir_ds;
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

  result.spec_weight = evaluate_specular_weight(kg, params, result.dir_ds, result.dir_sl);
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
  const float cos_theta = fabsf(dot(eval.normal, -result.wi));
  const float dist2 = fmaxf(result.distance_ds * result.distance_ds, 1e-8f);
  if (area_element <= 0.0f || cos_theta <= 0.0f) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  const float area_to_solid = area_element * cos_theta / dist2;
  if (!isfinite_safe(area_to_solid) || area_to_solid <= 0.0f) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  result.jacobian_total = fabsf(determinant) * area_to_solid;
  result.jacobian = result.jacobian_total;
  vertex.jacobian = result.jacobian_total;

  failure_code = MPG_FAILURE_NONE;

  return true;
}

bool mpg_solve_double_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
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

  ShaderData primary_sd = {};
  primary_sd.P = surface_point_from_barycentric(primary_geometry, primary_u, primary_v);
  primary_sd.time = sd.time;
  primary_sd.prim = seed.prim;
  primary_sd.object = seed.object;

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
                              seed.light_sample.P,
                              primary_u,
                              primary_v,
                              secondary_u,
                              secondary_v,
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

  float trust_radius = 0.25f;
  float prev_residual = FLT_MAX;
  int increase_counter = 0;

  for (int iter = 0; iter < options.max_iters; ++iter) {
    if (residual_norm < 1.0e-5f) {
      break;
    }

    float J[4][4];
    if (!compute_double_bounce_jacobian(receiver,
                                        primary_geometry,
                                        primary_params,
                                        secondary_geometry,
                                        secondary_params,
                                        seed.light_sample.P,
                                        primary_u,
                                        primary_v,
                                        secondary_u,
                                        secondary_v,
                                        eval.residual,
                                        J))
    {
      failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
      return false;
    }

    float delta[4];
    if (!solve_linear_system_4x4(J, eval.residual, delta)) {
      failure_code = MPG_FAILURE_JACOBIAN_ZERO;
      return false;
    }

    float step_norm = 0.0f;
    for (int i = 0; i < 4; ++i) {
      step_norm += delta[i] * delta[i];
    }
    step_norm = sqrtf(step_norm);

    if (step_norm > trust_radius && step_norm > 0.0f) {
      const float scale = trust_radius / (step_norm + 1.0e-8f);
      for (int i = 0; i < 4; ++i) {
        delta[i] *= scale;
      }
      step_norm = trust_radius;
    }

    float new_primary_u = primary_u - delta[0];
    float new_primary_v = primary_v - delta[1];
    float new_secondary_u = secondary_u - delta[2];
    float new_secondary_v = secondary_v - delta[3];

    project_barycentrics(new_primary_u, new_primary_v);
    project_barycentrics(new_secondary_u, new_secondary_v);

    DoubleBounceEval new_eval;
    if (!evaluate_double_bounce(receiver,
                                primary_geometry,
                                primary_params,
                                secondary_geometry,
                                secondary_params,
                                seed.light_sample.P,
                                new_primary_u,
                                new_primary_v,
                                new_secondary_u,
                                new_secondary_v,
                                new_eval))
    {
      trust_radius *= 0.5f;
      if (trust_radius < 1.0e-6f) {
        failure_code = (new_eval.primary.tir || new_eval.secondary.tir) ?
                            MPG_FAILURE_TOTAL_INTERNAL_REFLECTION :
                            MPG_FAILURE_NEWTON_DIVERGED;
        return false;
      }
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

    if (new_norm < residual_norm) {
      primary_u = new_primary_u;
      primary_v = new_primary_v;
      secondary_u = new_secondary_u;
      secondary_v = new_secondary_v;
      eval = new_eval;
      residual_norm = new_norm;
      trust_radius = fminf(trust_radius * 1.5f, 1.0f);
      if (prev_residual - residual_norm < 1.0e-6f) {
        ++increase_counter;
      }
      else {
        increase_counter = 0;
      }
      prev_residual = residual_norm;
    }
    else {
      trust_radius *= 0.5f;
      if (trust_radius < 1.0e-6f) {
        failure_code = MPG_FAILURE_NEWTON_DIVERGED;
        return false;
      }
      ++increase_counter;
    }

    if (increase_counter >= 3) {
      break;
    }
  }

  if (!isfinite_safe(residual_norm) || residual_norm > 1.0e-4f) {
    failure_code = MPG_FAILURE_NEWTON_DIVERGED;
    return false;
  }

  if (!specular_parameters_from_surface(kg, sd, primary_geometry, seed, primary_u, primary_v, primary_params)) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  primary_sd.P = surface_point_from_barycentric(primary_geometry, primary_u, primary_v);
  if (!specular_parameters_from_surface(
          kg, primary_sd, secondary_geometry, secondary_seed, secondary_u, secondary_v, secondary_params))
  {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  if (!evaluate_double_bounce(receiver,
                              primary_geometry,
                              primary_params,
                              secondary_geometry,
                              secondary_params,
                              seed.light_sample.P,
                              primary_u,
                              primary_v,
                              secondary_u,
                              secondary_v,
                              eval))
  {
    failure_code = (eval.primary.tir || eval.secondary.tir) ? MPG_FAILURE_TOTAL_INTERNAL_REFLECTION :
                                                                 MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }

  MpgSeedRay primary_seed_for_jacobian = seed;
  primary_seed_for_jacobian.light_sample.P = eval.secondary.point;

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
  const float cos_primary = fabsf(dot(eval.primary.normal, -eval.primary.dir_ds));
  const float dist_primary_sq = fmaxf(eval.primary.distance_ds * eval.primary.distance_ds, 1.0e-8f);
  if (!(area_primary > 0.0f) || !(cos_primary > 0.0f)) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  const float area_to_solid_primary = area_primary * cos_primary / dist_primary_sq;
  if (!isfinite_safe(area_to_solid_primary) || area_to_solid_primary <= 0.0f) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  const float jacobian_primary = fabsf(determinant_primary) * area_to_solid_primary;

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
  const float cos_secondary = fabsf(dot(eval.secondary.normal, -eval.secondary.dir_sl));
  const float dist_secondary_sq = fmaxf(eval.secondary.distance_sl * eval.secondary.distance_sl, 1.0e-8f);
  if (!(area_secondary > 0.0f) || !(cos_secondary > 0.0f)) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  const float area_to_solid_secondary = area_secondary * cos_secondary / dist_secondary_sq;
  if (!isfinite_safe(area_to_solid_secondary) || area_to_solid_secondary <= 0.0f) {
    failure_code = MPG_FAILURE_DEGENERATE_NORMALS;
    return false;
  }
  const float jacobian_secondary = fabsf(determinant_secondary) * area_to_solid_secondary;

  const float jacobian_total = jacobian_primary * jacobian_secondary;
  if (!isfinite_safe(jacobian_total) || fabsf(jacobian_total) <= 1.0e-12f) {
    failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    return false;
  }

  const float3 dir_primary_in = -eval.primary.dir_ds;
  const float3 dir_primary_out = eval.primary.dir_sl;
  const Spectrum primary_weight = evaluate_specular_weight(kg, primary_params, dir_primary_in, dir_primary_out);
  if (is_zero(primary_weight)) {
    failure_code = MPG_FAILURE_ZERO_THROUGHPUT;
    return false;
  }

  const float3 dir_secondary_in = -eval.secondary.dir_ds;
  const float3 dir_secondary_out = eval.secondary.dir_sl;
  const Spectrum secondary_weight = evaluate_specular_weight(kg, secondary_params, dir_secondary_in, dir_secondary_out);
  if (is_zero(secondary_weight)) {
    failure_code = MPG_FAILURE_ZERO_THROUGHPUT;
    return false;
  }

  const Spectrum spec_throughput = primary_weight * secondary_weight;
  if (is_zero(spec_throughput)) {
    failure_code = MPG_FAILURE_ZERO_THROUGHPUT;
    return false;
  }

  const float visibility_primary = compute_segment_visibility(
      kg, sd.P, sd.Ng, eval.primary.point, sd.time, sd.object, sd.prim);
  const float visibility_intermediate = compute_segment_visibility(
      kg, eval.primary.point, eval.primary.normal, eval.secondary.point, sd.time, seed.object, seed.prim);
  const float visibility_secondary = compute_segment_visibility(kg,
                                                                eval.secondary.point,
                                                                eval.secondary.normal,
                                                                seed.light_sample.P,
                                                                sd.time,
                                                                secondary_seed.object,
                                                                secondary_seed.prim);
  const float visibility = visibility_primary * visibility_intermediate * visibility_secondary;

  result.success = true;
  result.specular_vertex_count = 2;
  result.wi = eval.primary.dir_ds;
  result.visibility = visibility;
  result.jacobian_total = jacobian_total;
  result.jacobian = jacobian_total;
  result.specular_throughput = spec_throughput;
  result.spec_weight = spec_throughput;
  result.is_refraction = primary_params.is_refraction || secondary_params.is_refraction;

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
