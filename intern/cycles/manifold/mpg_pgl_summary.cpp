/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_pgl_summary.h"

#if defined(WITH_PATH_GUIDING)

#  include <openpgl/cpp/SurfaceSamplingDistribution.h>

#  include <algorithm>
#  include <cmath>
#  include <vector>

#  include "util/hash.h"

CCL_NAMESPACE_BEGIN

namespace {

class HashSequence2D {
public:
  explicit HashSequence2D(const uint32_t seed) : seed_(seed), counter_(0)
  {
  }

  float2 next()
  {
    const uint32_t idx = counter_++;
    const uint32_t hash_x = hash_uint3(seed_, idx, 0u);
    const uint32_t hash_y = hash_uint3(seed_, idx, 1u);
    return make_float2(uint_to_float_excl(hash_x), uint_to_float_excl(hash_y));
  }

private:
  uint32_t seed_;
  uint32_t counter_;
};

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
                          uint32_t rng_seed,
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

  HashSequence2D rng(rng_seed);
  const int max_attempts = std::max(n_samples * 8, n_samples);
  int attempts = 0;
  while ((int)samples.size() < n_samples && attempts < max_attempts) {
    attempts++;
    const float2 sample_uv = rng.next();
    const pgl_point2f sample = {sample_uv.x, sample_uv.y};
    const pgl_vec3f dir = dist_world.Sample(sample);
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

  const float cone_cos = cosf(cone_half_angle_rad);
  int best_count = 0;
  float3 best_sum = zero_float3();
  float best_sum_len_sq = 0.0f;

  for (int i = 0; i < count; i++) {
    const float3 &pivot = samples[i];
    float3 cluster_sum = zero_float3();
    int cluster_count = 0;

    for (int j = 0; j < count; j++) {
      const float3 &candidate = samples[j];
      if (dot(candidate, pivot) >= cone_cos) {
        cluster_sum += candidate;
        cluster_count++;
      }
    }

    const float cluster_sum_len_sq = dot(cluster_sum, cluster_sum);
    if (cluster_count > best_count ||
        (cluster_count == best_count && cluster_sum_len_sq > best_sum_len_sq)) {
      best_count = cluster_count;
      best_sum = cluster_sum;
      best_sum_len_sq = cluster_sum_len_sq;
    }
  }

  float3 mean_dir = zero_float3();
  float rbar = 0.0f;
  float peak_weight = 0.0f;

  const bool have_cluster = (best_count > 0) && (best_sum_len_sq > 1.0e-12f) &&
                            std::isfinite(best_sum_len_sq);
  if (have_cluster) {
    const float cluster_sum_len = sqrtf(best_sum_len_sq);
    mean_dir = best_sum / cluster_sum_len;
    rbar = clamp(cluster_sum_len / float(best_count), 0.0f, 1.0f);
    peak_weight = float(best_count) / float(count);
  }
  else {
    rbar = 0.0f;
    peak_weight = 0.0f;
  }

  const bool have_direction = (rbar >= 1.0e-3f) && !is_zero(mean_dir);
  if (!have_direction) {
    mean_dir = zero_float3();
    rbar = 0.0f;
    peak_weight = 0.0f;
  }

  const float kappa = have_direction ? estimate_kappa_from_rbar(rbar) : 0.0f;

  out.mean_dir = mean_dir;
  out.peak_weight = peak_weight;
  out.kappa = kappa;
  out.rbar = rbar;
  return true;
}

CCL_NAMESPACE_END

#endif