/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_solve.h"

#include "kernel/bvh/bvh.h"
#include "kernel/bvh/util.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/object.h"
#include "kernel/geom/shader_data.h"
#include "kernel/geom/triangle.h"
#include "kernel/svm/types.h"
#include "kernel/types.h"

#include "util/math_matrix.h"

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

SpecularParameters specular_parameters_from_closure(const ShaderClosure &bsdf)
{
  SpecularParameters params;
  params.is_refraction = CLOSURE_IS_REFRACTION(bsdf.type) || CLOSURE_IS_GLASS(bsdf.type);
  if (params.is_refraction) {
    const MicrofacetBsdf *microfacet = reinterpret_cast<const MicrofacetBsdf *>(&bsdf);
    params.base_eta = fmaxf(microfacet->ior, 1e-6f);
  }
  return params;
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
  shadow_ray.P = ray_offset(eval.point, eval.normal);
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

  result = MpgSolverOutput();

  if (seed.prim < 0 || seed.object < 0) {
    return false;
  }

  SpecularSurfaceGeometry geometry;
  if (!load_surface_geometry(kg, sd, seed, geometry)) {
    return false;
  }

  const SpecularParameters params = specular_parameters_from_closure(bsdf);

  float u = clamp(seed.bary_u, 1e-4f, 1.0f - 1e-4f);
  float v = clamp(seed.bary_v, 1e-4f, 1.0f - 1e-4f);
  project_barycentrics(u, v);

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

  const float area_element = len(cross(eval.dXdu, eval.dXdv));
  const float cos_theta = fabsf(dot(eval.normal, -result.wi));
  const float dist2 = fmaxf(result.distance_ds * result.distance_ds, 1e-8f);
  if (area_element <= 0.0f || cos_theta <= 0.0f) {
    return false;
  }
  result.jacobian = area_element * cos_theta / dist2;

  return true;
}

CCL_NAMESPACE_END
