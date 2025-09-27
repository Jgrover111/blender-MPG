/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg.h"

#include "manifold/mpg_types.h"

#include "manifold/mpg_pdf.h"
#include "manifold/mpg_seed.h"
#include "manifold/mpg_solve.h"

#include "kernel/light/common.h"
#include "kernel/device/cpu/globals.h"
#include "kernel/integrator/path_state.h"
#include "kernel/svm/types.h"
#include "kernel/types.h"

#include <algorithm>
#include <cfloat>

CCL_NAMESPACE_BEGIN

MpgResult mpg_try_connect(KernelGlobals kg,
                          const ShaderData &sd,
                          const ShaderClosure &bsdf,
                          const GuideSummary &g,
                          const MpgOptions &opt,
                          const uint32_t path_flag,
                          const int bounce,
                          RNGState &rng_state)
{
  MpgResult result;
  if (opt.max_bounces <= 0) {
    return result;
  }

  const int solver_bounce_count = std::min(opt.max_bounces, 2);

  if (CLOSURE_IS_BSDF_SINGULAR(bsdf.type) && !CLOSURE_IS_RAY_PORTAL(bsdf.type)) {
    return result;
  }

  const bool gate_active = (opt.gate_w > 0.0f) || (opt.gate_kappa > 0.0f);
  if (gate_active) {
    const bool has_direction_relaxed = (g.rbar > 1.0e-4f);
    const bool has_direction_strict = (g.rbar > 1.0e-3f);
    const bool strict_gate = has_direction_strict && (g.peak_weight >= opt.gate_w) &&
                             (g.kappa >= opt.gate_kappa);
    const bool relaxed_gate = opt.relax_gate && has_direction_relaxed;
    const bool bootstrap_gate = opt.relax_gate && !has_direction_relaxed;
    if (!strict_gate && !relaxed_gate && !bootstrap_gate) {
      return result;
    }
  }
  else if (g.rbar <= 1.0e-5f) {
    /* Without guiding gate we still require a numerically stable direction. */
    return result;
  }

  MpgSeedRay seed;
  if (!mpg_generate_seed(kg, sd, bsdf, g, opt, path_flag, bounce, rng_state, seed)) {
    return result;
  }

  MpgSolverOutput solution;
  bool solved = false;

  switch (solver_bounce_count) {
    case 1:
      solved = mpg_solve_single_bounce(kg, sd, bsdf, seed, opt, rng_state, solution);
      break;
    case 2:
      /* Two-bounce solving is not implemented yet. Guard the branch so the plumbing compiles
       * ahead of the solver landing. */
      kernel_assert(false && "Two-bounce MPG solver not implemented yet");
      return result;
    default:
      kernel_assert(false && "Unsupported MPG bounce count");
      return result;
  }

  if (!solved) {
    return result;
  }

  // float pdf = 0.0f;
  // if (!mpg_evaluate_pdf(kg, sd, bsdf, g, seed, solution, pdf) || pdf <= 0.0f) {
  //   return result;
  // }

  const float nee_pdf = seed.light_sample.pdf;
  if (!isfinite_safe(nee_pdf) || nee_pdf <= 0.0f) {
    return result;
  }

  if (solution.specular_vertex_count <= 0) {
    return result;
  }

  const MpgSpecularVertex &exit_vertex =
      solution.specular_vertices[solution.specular_vertex_count - 1];

  LightSample light_sample = seed.light_sample;
  const float pdf_selection = light_sample.pdf_selection;
  if (pdf_selection != 0.0f) {
    /* `light_sample_update` expects `ls->pdf` without the selection term. */
    light_sample.pdf /= pdf_selection;
  }

  uint32_t updated_path_flag = path_flag;
  if (solution.is_refraction) {
    updated_path_flag |= PATH_RAY_MIS_HAD_TRANSMISSION;
  }

  light_sample_update(kg, &light_sample, exit_vertex.position, exit_vertex.normal, updated_path_flag);

  /* `light_sample_update()` re-applies the selection term, so `light_sample.pdf` already
   * contains the probability of picking this emitter. Do not multiply by it again or the
   * MPG technique PDF gets scaled by an extra factor of `pdf_selection`, driving the MIS
   * weight towards zero. */
  const float light_pdf_solid = light_sample.pdf;

  float pdf = seed.seed_pdf * light_pdf_solid * solution.jacobian_total;

  if (!isfinite_safe(pdf) || pdf <= 0.0f) {
    return result;
  }

  result.success = true;
  result.wi = solution.wi;
  result.pdf = pdf;
  result.nee_pdf = nee_pdf;
  result.visibility = solution.visibility;
  result.jacobian_total = solution.jacobian_total;
  result.spec_weight = solution.specular_throughput;
  result.specular_vertex_count = solution.specular_vertex_count;
  for (int i = 0; i < solution.specular_vertex_count; ++i) {
    result.specular_vertices[i] = solution.specular_vertices[i];
  }
  result.light = light_sample;
  return result;
}

CCL_NAMESPACE_END