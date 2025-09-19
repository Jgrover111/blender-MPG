/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_pgl_summary.h"

#if defined(WITH_PATH_GUIDING)

#  include "BLI_rand.h"

#  include <openpgl/cpp/SurfaceSamplingDistribution.h>

#  include <algorithm>
#  include <cmath>
#  include <vector>

CCL_NAMESPACE_BEGIN

namespace {

inline float3 safe_normalize_or_zero(const float3 &v, float &length)
{
  float3 n = safe_normalize_len(v, &length);
  if (length == 0.0f) {
    return zero_float3();
  }
  return n;
}

inline float estimate_kappa_from_rbar(const float rbar)
{
  const float clamped_rbar = clamp(rbar, 0.0f, 1.0f);
  if (clamped_rbar <= 1.0e-6f) {
    return 0.0f;
  }
  if (clamped_rbar >= 1.0f - 1.0e-6f) {
    return 1.0e4f;
  }

  const float r_sq = clamped_rbar * clamped_rbar;
  const float denom = max(1.0f - r_sq, 1.0e-6f);
  float kappa = (clamped_rbar * (3.0f - r_sq)) / denom;
  kappa = clamp(kappa, 0.0f, 1.0e4f);

  if (!(kappa > 0.0f)) {
    return 0.0f;
  }

  if (!std::isfinite(kappa)) {
    return 0.0f;
  }

  /* Optional Newton refinement of A(kappa) = rbar for vMF on S^2.
   * A(kappa) = coth(kappa) - 1/kappa. */
  const float inv_k = 1.0f / kappa;
  const float coth_k = 1.0f / tanhf(kappa);
  const float A_k = coth_k - inv_k;
  const float derivative = 1.0f - (A_k * A_k) - (A_k * inv_k);
  if (fabsf(derivative) > 1.0e-6f) {
    const float newton = kappa - (A_k - clamped_rbar) / derivative;
    if (std::isfinite(newton)) {
      kappa = clamp(newton, 0.0f, 1.0e4f);
    }
  }

  return clamp(kappa, 0.0f, 1.0e4f);
}

}  // namespace

bool pgl_estimate_summary(const OpenPGLSurfaceDistribution &dist_world,
                          const float3 &Ng_world,
                          RNG &rng,
                          GuideSummary &out,
                          int n_samples,
                          float cone_half_angle_rad)
{
  out.mean_dir = zero_float3();
  out.peak_weight = 0.0f;
  out.kappa = 0.0f;
  out.rbar = 0.0f;

  n_samples = max(n_samples, 1);
  cone_half_angle_rad = clamp(cone_half_angle_rad, 0.0f, M_PI_F);

  const bool have_normal = !is_zero(Ng_world);
  const float3 hemisphere_normal = have_normal ? safe_normalize(Ng_world) : zero_float3();
  const bool constrain_hemisphere = have_normal && !is_zero(hemisphere_normal);

  std::vector<float3> samples;
  samples.reserve(n_samples);

  const int max_attempts = std::max(n_samples * 8, n_samples);
  int attempts = 0;
  while ((int)samples.size() < n_samples && attempts < max_attempts) {
    attempts++;
    const pgl_point2f sample_uv = {BLI_rng_get_float(&rng), BLI_rng_get_float(&rng)};
    const pgl_vec3f dir = dist_world.Sample(sample_uv);
    const float3 direction = make_float3(dir.x, dir.y, dir.z);

    if (!isfinite(direction.x) || !isfinite(direction.y) || !isfinite(direction.z)) {
      continue;
    }

    if (constrain_hemisphere && dot(direction, hemisphere_normal) <= 0.0f) {
      continue;
    }

    const float length_sq = dot(direction, direction);
    if (!(length_sq > 0.0f)) {
      continue;
    }

    samples.push_back(direction / sqrtf(length_sq));
  }

  const int count = samples.size();
  if (count == 0) {
    return false;
  }

  float3 mean = zero_float3();
  for (const float3 &d : samples) {
    mean += d;
  }
  mean /= float(count);

  float rbar = 0.0f;
  const float3 mean_dir = safe_normalize_or_zero(mean, rbar);
  if (rbar < 1.0e-3f) {
    return false;
  }

  const float cone_cos = cosf(cone_half_angle_rad);
  int in_cone = 0;
  for (const float3 &d : samples) {
    if (dot(d, mean_dir) >= cone_cos) {
      in_cone++;
    }
  }

  const float peak_weight = float(in_cone) / float(count);
  const float kappa = estimate_kappa_from_rbar(rbar);

  out.mean_dir = mean_dir;
  out.peak_weight = peak_weight;
  out.kappa = kappa;
  out.rbar = rbar;
  return true;
}

CCL_NAMESPACE_END

#endif