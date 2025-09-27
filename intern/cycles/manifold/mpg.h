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

#include "util/math.h"

CCL_NAMESPACE_BEGIN

struct MpgOptions {
  int max_bounces = 1;
  int max_iters = 6;
  float gate_w = 0.35f;
  float gate_kappa = 40.0f;
  float angular_jitter = 0.02f;
  bool relax_gate = false;
};

struct MpgResult {
  bool success = false;
  float3 wi = zero_float3();
  float pdf = 0.0f;
  float nee_pdf = 0.0f;
  float visibility = 0.0f;
  Spectrum spec_weight = zero_spectrum();
  LightSample light = {};
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