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

  /* Multi-bounce solving is not implemented yet. */
  if (opt.max_bounces > 1) {
    return result;
  }

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
    if (!strict_gate && !relaxed_gate) {
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
  if (!mpg_solve_single_bounce(kg, sd, bsdf, seed, opt, rng_state, solution)) {
    return result;
  }

  // float pdf = 0.0f;
  // if (!mpg_evaluate_pdf(kg, sd, bsdf, g, seed, solution, pdf) || pdf <= 0.0f) {
  //   return result;
  // }

  LightSample light_sample = seed.light_sample;
  const float pdf_selection = light_sample.pdf_selection;
  float nee_pdf = seed.light_pdf_receiver;
  if (pdf_selection != 0.0f) {
    /* `light_sample_update` expects `ls->pdf` without the selection term. */
    light_sample.pdf /= pdf_selection;
  }

  uint32_t updated_path_flag = path_flag;
  if (solution.is_refraction) {
    updated_path_flag |= PATH_RAY_MIS_HAD_TRANSMISSION;
  }

  light_sample_update(kg,
                      &light_sample,
                      solution.specular_point,
                      solution.specular_normal,
                      updated_path_flag);

  float light_pdf_solid = light_sample.pdf;
  if (pdf_selection != 0.0f)
    light_pdf_solid *= pdf_selection;

  float pdf = seed.seed_pdf * light_pdf_solid * solution.jacobian;

  if (!isfinite_safe(pdf) || pdf <= 0.0f) {
    return result;
  }

  if (nee_pdf > 0.0f) {
    if (seed.light_sample.type == LIGHT_AREA || seed.light_sample.type == LIGHT_TRIANGLE) {
      const float3 light_normal = make_float3(
          seed.light_sample.Ng.x, seed.light_sample.Ng.y, seed.light_sample.Ng.z);
      const float area_to_solid = light_pdf_area_to_solid_angle(
          light_normal, -seed.light_sample.D, seed.light_sample.t);
      if (isfinite_safe(area_to_solid) && area_to_solid > 0.0f) {
        nee_pdf *= area_to_solid;  // MPG_FIX: convert NEE pdf to solid angle at the receiver.
      }
      else {
        nee_pdf = 0.0f;
      }
    }
  }

  result.success = true;
  result.wi = solution.wi;
  result.pdf = pdf;
  result.visibility = solution.visibility;
  result.spec_weight = solution.spec_weight;
  result.light = light_sample;
  result.nee_pdf = nee_pdf;  // MPG_FIX: carry receiver-space light pdf for MIS balance.
  return result;
}

CCL_NAMESPACE_END