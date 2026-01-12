/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Manifold Path Guiding (MPG) - Cycles implementation
 *
 * This file matches the structure of the Mitsuba MPG reference implementation:
 * https://github.com/mollnn/manifold-path-guiding/tree/main/mitsuba/src/integrators/MPG
 *
 * File corresponds to: manifold_path_guiding.h in the Mitsuba reference
 *
 * Key differences from Mitsuba:
 * - Uses Cycles/Blender types (float3, Spectrum) instead of Mitsuba types
 * - Integrates with OpenPGL for spatial/directional distributions (see dtree.h)
 * - CPU-only implementation (no GPU kernel variants)
 */

#pragma once

#include "integrator/guiding.h"

#include "kernel/device/cpu/globals.h"
#include "kernel/integrator/path_state.h"
#include "kernel/light/common.h"
#include "kernel/light/sample.h"
#include "kernel/types.h"

#include "util/math.h"

#include <cstdint>

CCL_NAMESPACE_BEGIN

/* ========================================================================
 * Configuration and Options
 * ======================================================================== */

/* Configuration structure for MPG solver and sampling
 * Corresponds to ManifoldPathGuidingConfig in Mitsuba reference */
struct MpgOptions {
  /* Number of specular bounces supported by the solver. */
  int max_bounces = 2;
  int max_iters = 20;
  float gate_w = 0.35f;
  float gate_kappa = 40.0f;
  /* Newton solver step scaling factor (matches Mitsuba's m_config.step_scale).
   * Multiplies the Newton step size: new_param = param - step_scale * beta * delta.
   * Default 1.0 for standard Newton steps. */
  float step_scale = 1.0f;
  /* Angular jitter removed - Mitsuba reference uses uniform sampling without cone restrictions.
   * Cone-based sampling is replaced with uniform sphere/hemisphere sampling to match reference. */
  /* Enable relax_gate by default to allow bootstrap sampling when guide isn't ready yet.
   * This prevents failure code 302 (guide not ready) from blocking MPG entirely. */
  bool relax_gate = true;
  /* Increased from 8 to match Mitsuba's more generous retry budget.
   * Mitsuba allows up to 1e6 trials, we use 64 as a practical compromise. */
  int max_seed_repeat_trials = 64;
};

/* ========================================================================
 * Failure Codes and Status
 * ======================================================================== */

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

enum MpgSeedScatter : int {
  MPG_SEED_SCATTER_NONE = 0,
  MPG_SEED_SCATTER_REFLECTION = 1,
  MPG_SEED_SCATTER_REFRACTION = 2,
};

/* ========================================================================
 * Vertex and Path Structures
 * Corresponds to ManifoldVertex and EmitterInteraction in Mitsuba
 * ======================================================================== */

struct ShadingPoint {
  float3 position = zero_float3();
  float3 geometric_normal = zero_float3();
  float3 shading_normal = zero_float3();
  float3 wo = zero_float3();
  float time = 0.0f;
};

/* Seed ray proposal for manifold walk
 * Combines direction sampling and light sampling */
struct MpgSeedRay {
  float3 direction = zero_float3();
  float3 direction_normalized = zero_float3();
  float seed_pdf = 0.0f;
  float seed_pdf_raw = 0.0f;
  float seed_branch_pdf = 0.0f;
  float seed_direction_pdf = 0.0f;
  float seed_scatter_pdf = 0.0f;
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
  MpgSeedScatter scatter = MPG_SEED_SCATTER_NONE;
  int bounce_count = 1;
  uint8_t tau_bits = 0;
  uint8_t tau_count = 0;
  bool use_smooth_normals = false;
};

/* Specular vertex on the manifold
 * Corresponds to ManifoldVertex in Mitsuba */
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
  bool is_refraction = false; /* True when this specular vertex is refractive. */
  bool total_internal_reflection = false;
  int object = -1;
  int prim = -1;
};

/* ========================================================================
 * Solver Output
 * Corresponds to the result of manifold walk in Mitsuba
 * ======================================================================== */

struct MpgSolverOutput {
  bool success = false;
  int specular_vertex_count = 0;
  float3 wi = zero_float3();
  float visibility = 1.0f;
  float jacobian_total = 0.0f;
  Spectrum specular_throughput = zero_spectrum();
  float seed_resample_factor = 0.0f; /* Expected trials before re-discovering seed. */
  MpgSeedBranch seed_branch = MPG_SEED_BRANCH_NONE;
  MpgSeedScatter seed_scatter = MPG_SEED_SCATTER_NONE;
  float seed_branch_pdf = 0.0f;
  float seed_direction_pdf = 0.0f;
  float seed_scatter_pdf = 0.0f;
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
  bool is_refraction = false; /* True only when the terminal specular vertex is refractive. */
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

/* ========================================================================
 * Main Result Structure
 * ======================================================================== */

struct MpgResult {
  bool success = false;
  float3 wi = zero_float3();
  float pdf = 0.0f;
  float nee_pdf = 0.0f;
  float visibility = 0.0f;
  float jacobian_total = 0.0f;
  float seed_pdf = 0.0f;
  float seed_pdf_raw = 0.0f;
  float seed_resample_factor = 0.0f; /* Expected trials before re-discovering seed. */
  float seed_branch_pdf = 0.0f;
  float seed_direction_pdf = 0.0f;
  float seed_scatter_pdf = 0.0f;
  float3 seed_direction_normalized = zero_float3();
  uint8_t seed_tau_bits = 0;
  uint8_t seed_tau_count = 0;
  float bounce_pdf = 1.0f;
  float bounce_pdf_raw = 1.0f;
  /* Light pdf converted to receiver solid angle (includes jacobian_total). */
  float light_pdf = 0.0f;
  Spectrum spec_weight = zero_spectrum();
  LightSample light = {};
  int specular_vertex_count = 0;
  MpgSpecularVertex specular_vertices[2];
  int attempt_count = 0;
  int seed_trial_count = 0;
  int seed_accepted_trial_count = 0;
  int seed_guided_trial_count = 0;
  int seed_fallback_trial_count = 0;
  MpgSeedBranch seed_branch = MPG_SEED_BRANCH_NONE;
  MpgSeedScatter seed_scatter = MPG_SEED_SCATTER_NONE;
  int bounce_count = 1;
  uint32_t gate_mask = MPG_GATE_MASK_NONE;
  MpgFailureCode failure_code = MPG_FAILURE_NONE;
};

/* ========================================================================
 * Main Entry Point
 * Corresponds to the main manifold sampling function in Mitsuba
 * ======================================================================== */

/* Main entry point for Manifold Path Guiding
 * Attempts to connect a non-specular surface to a light through a specular chain
 * Returns a complete MpgResult with sampled direction, PDFs, and throughput */
MpgResult mpg_try_connect(KernelGlobals kg,
                          const ShaderData &sd,
                          const ShaderClosure &bsdf,
                          const GuideSummary &g,
                          const MpgOptions &opt,
                          const uint32_t path_flag,
                          const int bounce,
                          RNGState &rng_state);

/* ========================================================================
 * Internal Functions (for implementation, exposed for testing)
 * ======================================================================== */

/* Seed generation - corresponds to emitter sampling in Mitsuba */
bool mpg_generate_seed(KernelGlobals kg,
                       const ShaderData &sd,
                       const ShaderClosure &bsdf,
                       const GuideSummary &guide,
                       const MpgOptions &options,
                       const uint32_t path_flag,
                       const int bounce,
                       const RNGState &rng_state,
                       MpgSeedRay &seed,
                       MpgFailureCode &failure_code,
                       const int rng_branch_offset = 0);

/* Manifold walk solvers - corresponds to Manifold_Walk in Mitsuba */
bool mpg_solve_single_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
                             const GuideSummary &guide,
                             const MpgOptions &options,
                             RNGState &rng_state,
                             MpgSolverOutput &result,
                             MpgFailureCode &failure_code);

bool mpg_solve_double_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
                             const GuideSummary &guide,
                             const MpgOptions &options,
                             RNGState &rng_state,
                             MpgSolverOutput &result,
                             MpgFailureCode &failure_code);

/* PDF evaluation and utilities */
bool mpg_evaluate_pdf(KernelGlobals kg,
                      const ShaderData &sd,
                      const ShaderClosure &bsdf,
                      const GuideSummary &guide,
                      const MpgSeedRay &seed,
                      const MpgSolverOutput &solution,
                      float &pdf);

float mpg_light_sample_pdf_solid(KernelGlobals kg,
                                 const ShaderData &sd,
                                 const LightSample &light_sample);

float3 mpg_surface_ray_offset(KernelGlobals kg,
                              const ShaderData &sd,
                              const float3 ray_P,
                              const float3 ray_D);

float mpg_rebuild_seed_pdf(const MpgSeedRay &seed);

float mpg_seed_branch_probability(const int guided_attempt_budget,
                                  const int fallback_attempt_budget,
                                  MpgSeedBranch branch);

/* Visibility computation */
float mpg_compute_segment_visibility(KernelGlobals kg,
                                     const float3 &start_point,
                                     const float3 &start_normal,
                                     const float3 &end_point,
                                     float time,
                                     int skip_object,
                                     int skip_prim,
                                     int skip_light_object,
                                     int skip_light_prim);

/* Debug helper: Set current sample number for filtered debug output */
void mpg_set_current_sample(int sample);

CCL_NAMESPACE_END
