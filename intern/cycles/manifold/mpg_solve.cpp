/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_solve.h"

#include "util/math_matrix.h"

#include <cfloat>

CCL_NAMESPACE_BEGIN

namespace {
struct SpecularEval {
  float3 point;
  float3 normal;
  float3 dir_ds;
  float3 dir_sl;
  float distance_ds;
  float distance_sl;
  float3 dXdu;
  float3 dXdv;
  float3 dNdu;
  float3 dNdv;
  float3 residual;
  bool tir = false;
};

float3 combine_vertex_normals(const MpgSeedRay &seed, const float u, const float v)
{
  const float w = 1.0f - u - v;
  float3 n = seed.tri_n0 * w + seed.tri_n1 * u + seed.tri_n2 * v;
  if (is_zero(n)) {
    return normalize(seed.tri_n0);
  }
  return normalize(n);
}

float3 compute_normal_derivative(const MpgSeedRay &seed,
                                 const float u,
                                 const float v,
                                 const float3 &normal,
                                 const bool du)
{
  const float w = 1.0f - u - v;
  const float3 raw = seed.tri_n0 * w + seed.tri_n1 * u + seed.tri_n2 * v;
  const float norm_raw = len(raw);
  if (norm_raw == 0.0f) {
    return zero_float3();
  }
  const float3 d_raw = du ? (seed.tri_n1 - seed.tri_n0) : (seed.tri_n2 - seed.tri_n0);
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
                        const bool is_refraction,
                        const float eta,
                        bool &tir,
                        float &cos_theta_i,
                        float &cos_theta_t)
{
  if (!is_refraction) {
    tir = false;
    cos_theta_i = dot(-dir_ds, normal);
    cos_theta_t = cos_theta_i;
    return reflect_dir(dir_ds, normal);
  }
  return refract_dir(dir_ds, normal, eta, tir, cos_theta_i, cos_theta_t);
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
                       const float u,
                       const float v,
                       SpecularEval &eval)
{
  const float w = 1.0f - u - v;
  eval.point = seed.tri_v0 * w + seed.tri_v1 * u + seed.tri_v2 * v;
  eval.dXdu = seed.tri_v1 - seed.tri_v0;
  eval.dXdv = seed.tri_v2 - seed.tri_v0;
  eval.dir_ds = eval.point - D.position;
  eval.distance_ds = len(eval.dir_ds);
  eval.dir_ds = (eval.distance_ds > 0.0f) ? (eval.dir_ds / eval.distance_ds) : make_float3(0.0f, 0.0f, 1.0f);

  eval.dir_sl = seed.emitter_position - eval.point;
  eval.distance_sl = len(eval.dir_sl);
  eval.dir_sl = (eval.distance_sl > 0.0f) ? (eval.dir_sl / eval.distance_sl) : make_float3(0.0f, 0.0f, 1.0f);

  eval.normal = combine_vertex_normals(seed, u, v);
  eval.dNdu = compute_normal_derivative(seed, u, v, eval.normal, true);
  eval.dNdv = compute_normal_derivative(seed, u, v, eval.normal, false);

  float cos_theta_i = 0.0f, cos_theta_t = 0.0f;
  eval.residual = eval.dir_sl -
                  compute_specular(-eval.dir_ds, eval.normal, seed.is_refraction, seed.eta, eval.tir, cos_theta_i, cos_theta_t);
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
                      const SpecularEval &eval,
                      float3 J[2])
{
  const float3 d_dir_ds_du = derivative_normalized(eval.point - D.position, eval.dXdu);
  const float3 d_dir_ds_dv = derivative_normalized(eval.point - D.position, eval.dXdv);

  const float3 d_dir_sl_du = -derivative_normalized(seed.emitter_position - eval.point, -eval.dXdu);
  const float3 d_dir_sl_dv = -derivative_normalized(seed.emitter_position - eval.point, -eval.dXdv);

  float cos_theta_i = 0.0f, cos_theta_t = 0.0f;
  bool tir = false;
  const float3 spec_dir =
      compute_specular(-eval.dir_ds, eval.normal, seed.is_refraction, seed.eta, tir, cos_theta_i, cos_theta_t);
  const float sin_theta_i = sqrtf(fmaxf(0.0f, 1.0f - cos_theta_i * cos_theta_i));
  const float sin_theta_t = sqrtf(fmaxf(0.0f, 1.0f - cos_theta_t * cos_theta_t));

  float3 d_spec_du, d_spec_dv;
  if (!seed.is_refraction) {
    d_spec_du = derivative_specular_reflection(-eval.dir_ds, -d_dir_ds_du, eval.normal, eval.dNdu);
    d_spec_dv = derivative_specular_reflection(-eval.dir_ds, -d_dir_ds_dv, eval.normal, eval.dNdv);
  }
  else {
    d_spec_du = derivative_specular_refraction(-eval.dir_ds,
                                               -d_dir_ds_du,
                                               eval.normal,
                                               eval.dNdu,
                                               seed.eta,
                                               cos_theta_i,
                                               cos_theta_t,
                                               sin_theta_i,
                                               sin_theta_t);
    d_spec_dv = derivative_specular_refraction(-eval.dir_ds,
                                               -d_dir_ds_dv,
                                               eval.normal,
                                               eval.dNdv,
                                               seed.eta,
                                               cos_theta_i,
                                               cos_theta_t,
                                               sin_theta_i,
                                               sin_theta_t);
  }

  (void)spec_dir;
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
  (void)kg;
  (void)bsdf;
  (void)rng_state;

  result = MpgSolverOutput();

  if (seed.prim < 0 || seed.object < 0) {
    return false;
  }

  float u = clamp(seed.bary_u, 1e-4f, 1.0f - 1e-4f);
  float v = clamp(seed.bary_v, 1e-4f, 1.0f - 1e-4f);
  project_barycentrics(u, v);

  float trust_radius = 0.25f;
  float prev_residual = FLT_MAX;
  int increase_counter = 0;

  const ShadingPoint shading_point = shading_point_from_shader_data(sd);

  SpecularEval eval;
  evaluate_specular(shading_point, seed, u, v, eval);
  if (eval.tir) {
    return false;
  }

  float residual_norm = len(eval.residual);

  for (int iter = 0; iter < options.max_iters; ++iter) {
    if (residual_norm < 1e-5f) {
      break;
    }

    float3 J_cols[2];
    compute_jacobian(shading_point, seed, eval, J_cols);

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
    evaluate_specular(shading_point, seed, new_u, new_v, new_eval);
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
  result.visibility = seed.visibility;
  result.wi = normalize(eval.point - shading_point.position);
  result.object = seed.object;
  result.prim = seed.prim;
  result.light = seed.light;

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
