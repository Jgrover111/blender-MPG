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

CCL_NAMESPACE_BEGIN

MpgResult mpg_try_connect(KernelGlobals kg,
                          const ShaderData &sd,
                          const ShaderClosure &bsdf,
                          const GuideSummary &g,
                          const MpgOptions &opt,
                          RNGState &rng_state)
{
  MpgResult result;
  if (opt.max_bounces <= 0) {
    return result;
  }

  if (CLOSURE_IS_DELTA(bsdf.type)) {
    return result;
  }

  if (g.peak_weight < opt.gate_w || g.kappa < opt.gate_kappa) {
    return result;
  }

  MpgSeedRay seed;
  if (!mpg_generate_seed(kg, sd, bsdf, g, opt, rng_state, seed)) {
    return result;
  }

  MpgSolverOutput solution;
  if (!mpg_solve_single_bounce(kg, sd, bsdf, seed, opt, rng_state, solution)) {
    return result;
  }

  float pdf = 0.0f;
  if (!mpg_evaluate_pdf(kg, sd, bsdf, g, seed, solution, pdf) || pdf <= 0.0f) {
    return result;
  }

  result.success = true;
  result.wi = solution.wi;
  result.pdf = pdf;
  result.visibility = solution.visibility;
  result.light = seed.light_sample;
  return result;
}

CCL_NAMESPACE_END