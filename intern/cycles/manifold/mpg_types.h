/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/light/common.h"

#include "util/math.h"

#include <cstdint>

CCL_NAMESPACE_BEGIN

/* Length used when tracing visibility rays toward directional emitters. The value only
 * affects shadow rays and stays large enough to exit typical scene bounds without
 * perturbing the Jacobian computations that operate on normalized segments. */
static constexpr float MPG_DISTANT_LIGHT_VISIBILITY_DISTANCE = 1.0e6f;

enum MpgFailureCode : int {
  MPG_FAILURE_NONE = 0,
  MPG_FAILURE_SEED = 1,
  MPG_FAILURE_TOTAL_INTERNAL_REFLECTION = 2,
  MPG_FAILURE_DEGENERATE_NORMALS = 3,
  MPG_FAILURE_NEWTON_DIVERGED = 4,
  MPG_FAILURE_JACOBIAN_ZERO = 5,
  MPG_FAILURE_GATE = 6,
  MPG_FAILURE_GEOMETRY = 7,
  MPG_FAILURE_ZERO_THROUGHPUT = 8,
  MPG_FAILURE_INVALID_LIGHT_PDF = 9,
  MPG_FAILURE_INVALID_SEED_PDF = 10,
  MPG_FAILURE_INVALID_NEE_PDF = 11,
  MPG_FAILURE_INVALID_PDF = 12,
  MPG_FAILURE_NO_SPECULAR = 13,
  MPG_FAILURE_UNSUPPORTED = 14,
  MPG_FAILURE_UNKNOWN = 15
};

enum MpgGateMask : uint32_t {
  MPG_GATE_MASK_NONE = 0,
  MPG_GATE_MASK_ACTIVE = 1u << 0,
  MPG_GATE_MASK_HAS_DIRECTION_RELAXED = 1u << 1,
  MPG_GATE_MASK_HAS_DIRECTION_STRICT = 1u << 2,
  MPG_GATE_MASK_STRICT_PASS = 1u << 3,
  MPG_GATE_MASK_RELAX_PASS = 1u << 4,
  MPG_GATE_MASK_BOOTSTRAP_PASS = 1u << 5
};

enum MpgSeedBranch : int {
  MPG_SEED_BRANCH_NONE = 0,
  MPG_SEED_BRANCH_GUIDED = 1,
  MPG_SEED_BRANCH_FALLBACK = 2,
};

struct ShadingPoint {
  float3 position = zero_float3();
  float3 geometric_normal = zero_float3();
  float3 shading_normal = zero_float3();
  float3 wo = zero_float3();
  float time = 0.0f;
};

struct MpgSeedRay {
  float3 direction = zero_float3();
  float seed_pdf = 0.0f;
  float seed_pdf_raw = 0.0f;
  float seed_branch_pdf = 0.0f;
  float seed_direction_pdf = 0.0f;
  float bounce_pdf = 1.0f;
  float bounce_pdf_raw = 1.0f;
  LightSample light_sample = {};
  uint32_t path_flag = 0;
  int object = -1;
  int prim = -1;
  float bary_u = 0.0f;
  float bary_v = 0.0f;
  int trial_count = 0;
  int accepted_trial_count = 0;
  int guided_trial_count = 0;
  int fallback_trial_count = 0;
  float seed_resample_factor = 0.0f; /* Expected trials before re-discovering seed. */
  MpgSeedBranch branch = MPG_SEED_BRANCH_NONE;
  int bounce_count = 1;
  bool use_smooth_normals = false;
};

struct MpgSpecularVertex {
  float3 position = zero_float3();
  float3 normal = zero_float3();
  float3 dir_in = zero_float3();
  float3 dir_out = zero_float3();
  float distance_in = 0.0f;
  float distance_out = 0.0f;
  float eta = 1.0f;
  float cos_theta_in = 0.0f;
  float cos_theta_out = 0.0f;
  Spectrum throughput = zero_spectrum();
  float3 dXdu = zero_float3();
  float3 dXdv = zero_float3();
  float3 dNdu = zero_float3();
  float3 dNdv = zero_float3();
  float u = 0.0f;
  float v = 0.0f;
  float jacobian = 0.0f;
  bool is_refraction = false;
  bool total_internal_reflection = false;
  int object = -1;
  int prim = -1;
};

struct MpgSolverOutput {
  bool success = false;
  int specular_vertex_count = 0;
  float3 wi = zero_float3();
  float visibility = 1.0f;
  float jacobian_total = 0.0f;
  Spectrum specular_throughput = zero_spectrum();
  float seed_resample_factor = 0.0f; /* Expected trials before re-discovering seed. */
  MpgSeedBranch seed_branch = MPG_SEED_BRANCH_NONE;
  float seed_branch_pdf = 0.0f;
  float seed_direction_pdf = 0.0f;
  int seed_guided_trial_count = 0;
  int seed_fallback_trial_count = 0;
  MpgSpecularVertex specular_vertices[2];

  /* Legacy single-bounce fields kept for callers that have not yet
   * transitioned to the multi-vertex representation. These mirror the
   * contents of the first entry in `specular_vertices` when present. */
  float3 specular_point = zero_float3();
  float3 specular_normal = zero_float3();
  float3 dir_ds = zero_float3();
  float3 dir_sl = zero_float3();
  float distance_ds = 0.0f;
  float distance_sl = 0.0f;
  bool is_refraction = false;
  Spectrum spec_weight = zero_spectrum();
  float3 dXdu = zero_float3();
  float3 dXdv = zero_float3();
  float3 dNdu = zero_float3();
  float3 dNdv = zero_float3();
  float u = 0.0f;
  float v = 0.0f;
  float jacobian = 0.0f;
  int object = -1;
  int prim = -1;
};

CCL_NAMESPACE_END