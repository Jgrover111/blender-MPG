/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_seed.h"

#include "util/math_float4.h"

CCL_NAMESPACE_BEGIN

namespace {
float vmf_normalization(const float kappa)
{
  if (kappa <= 0.0f) {
    return 1.0f / (4.0f * M_PI_F);
  }
  const float sinh_k = sinhf(kappa);
  if (sinh_k <= 0.0f) {
    return 0.0f;
  }
  return kappa / (4.0f * M_PI_F * sinh_k);
}

float vmf_pdf(const float3 &dir, const float3 &mean, const float kappa)
{
  const float3 mu = normalize(mean);
  const float c = vmf_normalization(kappa);
  if (c == 0.0f) {
    return 0.0f;
  }
  const float dot_v = dot(dir, mu);
  return c * expf(kappa * dot_v);
}

float3 jitter_direction(const float3 &mean, const float jitter, RNG &rng)
{
  const float3 mu = (is_zero(mean)) ? make_float3(0.0f, 0.0f, 1.0f) : normalize(mean);
  float3 tangent, bitangent;
  make_orthonormals(mu, tangent, bitangent);

  const float2 rand = rng.uniform_float2();
  const float angle = jitter * sqrtf(fmaxf(rand.x, 0.0f));
  const float phi = 2.0f * M_PI_F * rand.y;
  const float sin_theta = sinf(angle);
  const float cos_theta = cosf(angle);

  return normalize(cos_theta * mu + sin_theta * (cosf(phi) * tangent + sinf(phi) * bitangent));
}

}  // namespace

bool mpg_generate_seed(const ShadingPoint &D,
                       const ClosureBSDF &bsdf,
                       const Lights &lights,
                       const GuideSummary &guide,
                       const MpgOptions &options,
                       RNG &rng,
                       MpgSeedRay &seed)
{
  seed = MpgSeedRay();

  if (guide.peak_weight < options.gate_w || guide.kappa < options.gate_kappa) {
    return false;
  }

  if (lights.num_triangles() == 0 || lights.num_emitters() == 0) {
    return false;
  }

  float3 emitter_pos, emitter_norm;
  float emitter_pdf = 0.0f;
  int emitter_index = -1;
  if (!lights.sample_emitter(rng, emitter_index, emitter_pos, emitter_norm, emitter_pdf) ||
      emitter_pdf <= 0.0f)
  {
    return false;
  }

  const float3 dir = jitter_direction(guide.mean_dir, options.angular_jitter, rng);
  int tri_index = -1;
  float t = 0.0f, u = 0.0f, v = 0.0f;
  if (!lights.intersect_specular(D.position + 1e-4f * D.geometric_normal, dir, tri_index, t, u, v)) {
    return false;
  }

  if (tri_index < 0) {
    return false;
  }

  const Lights::SpecularTriangle &tri = lights.triangle(tri_index);
  if (!tri.enabled) {
    return false;
  }

  seed.direction = dir;
  seed.seed_pdf = vmf_pdf(dir, guide.mean_dir, guide.kappa);
  if (seed.seed_pdf <= 0.0f) {
    return false;
  }

  seed.triangle_index = tri_index;
  seed.bary_u = u;
  seed.bary_v = v;
  seed.emitter_position = emitter_pos;
  seed.emitter_normal = emitter_norm;
  seed.emitter_pdf = emitter_pdf;
  seed.emitter_index = emitter_index;
  seed.is_refraction = tri.is_refraction || bsdf.is_refraction();
  seed.eta = tri.is_refraction ? tri.eta : bsdf.eta;
  seed.visibility = 1.0f;
  return true;
}

CCL_NAMESPACE_END