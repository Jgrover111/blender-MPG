/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg.h"

#include "manifold/mpg_types.h"

#include "manifold/mpg_pdf.h"
#include "manifold/mpg_seed.h"
#include "manifold/mpg_solve.h"

#include "kernel/device/cpu/globals.h"
#include "kernel/integrator/path_state.h"
#include "kernel/types.h"

#include <cfloat>

CCL_NAMESPACE_BEGIN

/* -------------------------------------------------------------------- */
/** \name RNG implementation
 * \{ */

RNG::RNG(uint64_t seed) : engine_(seed), dist_(0.0f, 1.0f)
{
}

float RNG::uniform_float()
{
  return dist_(engine_);
}

float2 RNG::uniform_float2()
{
  return make_float2(uniform_float(), uniform_float());
}

float3 RNG::uniform_direction()
{
  const float2 rand = uniform_float2();
  const float z = 1.0f - 2.0f * rand.x;
  const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
  const float phi = 2.0f * M_PI_F * rand.y;
  return make_float3(r * cosf(phi), r * sinf(phi), z);
}

/* \} */

/* -------------------------------------------------------------------- */
/** \name Lights implementation
 * \{ */

float3 Lights::SpecularTriangle::normal(const float u, const float v) const
{
  const float w = 1.0f - u - v;
  float3 n = n0 * w + n1 * u + n2 * v;
  if (is_zero(n)) {
    n = cross(v1 - v0, v2 - v0);
  }
  return normalize(n);
}

float3 Lights::SpecularTriangle::position(const float u, const float v) const
{
  const float w = 1.0f - u - v;
  return v0 * w + v1 * u + v2 * v;
}

float3 Lights::SpecularTriangle::dXdu() const
{
  return v1 - v0;
}

float3 Lights::SpecularTriangle::dXdv() const
{
  return v2 - v0;
}

void Lights::add_specular_triangle(const SpecularTriangle &tri)
{
  triangles_.push_back(tri);
}

void Lights::add_emitter(const Emitter &emitter)
{
  emitters_.push_back(emitter);
}

bool Lights::intersect_specular(const float3 &origin,
                                const float3 &direction,
                                int &triangle_index,
                                float &t,
                                float &u,
                                float &v) const
{
  bool hit = false;
  float best_t = FLT_MAX;
  int best_index = -1;
  float best_u = 0.0f, best_v = 0.0f;

  for (int i = 0; i < triangles_.size(); ++i) {
    const SpecularTriangle &tri = triangles_[i];
    if (!tri.enabled) {
      continue;
    }

    const float3 e1 = tri.v1 - tri.v0;
    const float3 e2 = tri.v2 - tri.v0;
    const float3 pvec = cross(direction, e2);
    const float det = dot(e1, pvec);
    if (fabsf(det) < 1e-8f) {
      continue;
    }
    const float inv_det = 1.0f / det;
    const float3 tvec = origin - tri.v0;
    const float uu = dot(tvec, pvec) * inv_det;
    if (uu < 0.0f || uu > 1.0f) {
      continue;
    }
    const float3 qvec = cross(tvec, e1);
    const float vv = dot(direction, qvec) * inv_det;
    if (vv < 0.0f || uu + vv > 1.0f) {
      continue;
    }
    const float tt = dot(e2, qvec) * inv_det;
    if (tt <= 1e-4f || tt >= best_t) {
      continue;
    }
    best_t = tt;
    best_index = i;
    best_u = uu;
    best_v = vv;
    hit = true;
  }

  if (hit) {
    triangle_index = best_index;
    t = best_t;
    u = best_u;
    v = best_v;
  }
  return hit;
}

bool Lights::sample_emitter(RNG &rng, int &index, float3 &position, float3 &normal, float &pdf) const
{
  index = -1;
  position = zero_float3();
  normal = zero_float3();
  pdf = 0.0f;

  float total_area = 0.0f;
  int enabled_count = 0;
  for (const Emitter &em : emitters_) {
    if (!em.enabled) {
      continue;
    }
    total_area += fmaxf(em.area, 0.0f);
    ++enabled_count;
  }

  if (enabled_count == 0) {
    return false;
  }

  if (total_area <= 0.0f) {
    /* Uniform selection. */
    const int choice = clamp(int(rng.uniform_float() * enabled_count), 0, enabled_count - 1);
    int counter = 0;
    for (int i = 0; i < emitters_.size(); ++i) {
      const Emitter &em = emitters_[i];
      if (!em.enabled) {
        continue;
      }
      if (counter == choice) {
        index = i;
        position = em.position;
        normal = em.normal;
        pdf = 1.0f / enabled_count;
        return true;
      }
      ++counter;
    }
    return false;
  }

  const float target = rng.uniform_float() * total_area;
  float accum = 0.0f;
  for (int i = 0; i < emitters_.size(); ++i) {
    const Emitter &em = emitters_[i];
    if (!em.enabled) {
      continue;
    }
    const float area = fmaxf(em.area, 0.0f);
    accum += area;
    if (target <= accum) {
      index = i;
      position = em.position;
      normal = em.normal;
      pdf = area / total_area;
      return true;
    }
  }

  /* Fallback to last enabled emitter. */
  for (int i = emitters_.size() - 1; i >= 0; --i) {
    const Emitter &em = emitters_[i];
    if (!em.enabled) {
      continue;
    }
    index = i;
    position = em.position;
    normal = em.normal;
    pdf = fmaxf(em.area, 0.0f) / total_area;
    return true;
  }
  return false;
}

const Lights::SpecularTriangle &Lights::triangle(const int index) const
{
  return triangles_[index];
}

const Lights::Emitter &Lights::emitter(const int index) const
{
  return emitters_[index];
}

int Lights::num_triangles() const
{
  return triangles_.size();
}

int Lights::num_emitters() const
{
  return emitters_.size();
}

/* \} */

static MpgResult mpg_try_connect_impl(const ShadingPoint &D,
                                      const ClosureBSDF &bsdf,
                                      const Lights &lights,
                                      const GuideSummary &g,
                                      const MpgOptions &opt,
                                      RNG &rng)
{
  MpgResult result;
  if (opt.max_bounces <= 0) {
    return result;
  }

  MpgSeedRay seed;
  if (!mpg_generate_seed(D, bsdf, lights, g, opt, rng, seed)) {
    return result;
  }

  MpgSolverOutput solution;
  if (!mpg_solve_single_bounce(D, bsdf, lights, seed, opt, rng, solution)) {
    return result;
  }

  float pdf = 0.0f;
  if (!mpg_evaluate_pdf(D, bsdf, seed, solution, pdf) || pdf <= 0.0f) {
    return result;
  }

  result.success = true;
  result.wi = solution.wi;
  result.pdf = pdf;
  result.visibility = solution.visibility;
  return result;
}

MpgResult mpg_try_connect(KernelGlobals kg,
                          const ShaderData &sd,
                          const ShaderClosure &bsdf,
                          const GuideSummary &g,
                          const MpgOptions &opt,
                          RNGState &rng_state)
{
  (void)kg;
  (void)bsdf;

  ShadingPoint shading_point;
  shading_point.position = sd.P;
  shading_point.geometric_normal = sd.Ng;
  shading_point.shading_normal = sd.N;
  shading_point.wo = -sd.wi;
  shading_point.time = sd.time;

  ClosureBSDF closure;
  closure.type = ClosureBSDF::Type::Reflection;
  closure.eta = 1.0f;
  closure.normal = is_zero(sd.N) ? shading_point.shading_normal : normalize(sd.N);
  make_orthonormals(closure.normal, closure.tangent, closure.bitangent);

  Lights lights;

  const uint64_t seed = ((uint64_t)rng_state.rng_pixel << 32) ^
                        ((uint64_t)rng_state.sample << 16) ^
                        (uint64_t)rng_state.rng_offset;
  RNG rng(seed);

  return mpg_try_connect_impl(shading_point, closure, lights, g, opt, rng);
}

CCL_NAMESPACE_END