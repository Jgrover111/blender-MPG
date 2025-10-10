/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "integrator/guiding.h"

#include "kernel/device/cpu/globals.h"
#include "kernel/integrator/path_state.h"
#include "kernel/light/common.h"
#include "kernel/light/sample.h"
#include "kernel/types.h"

#include "manifold/mpg_types.h"

#include "util/math.h"

CCL_NAMESPACE_BEGIN

struct MpgOptions {
  /* Number of specular bounces supported by the solver. */
  int max_bounces = 2;
  int max_iters = 6;
  float gate_w = 0.35f;
  float gate_kappa = 40.0f;
  float angular_jitter = 0.02f;
  bool relax_gate = false;
  int max_seed_repeat_trials = 8;
};

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
  int bounce_count = 1;
  uint32_t gate_mask = MPG_GATE_MASK_NONE;
  MpgFailureCode failure_code = MPG_FAILURE_NONE;
};

struct MpgSeedRay;
struct MpgSolverOutput;

MpgResult mpg_try_connect(KernelGlobals kg,
                          const ShaderData &sd,
                          const ShaderClosure &bsdf,
                          const GuideSummary &g,
                          const MpgOptions &opt,
                          const uint32_t path_flag,
                          const int bounce,
                          RNGState &rng_state);

CCL_NAMESPACE_END