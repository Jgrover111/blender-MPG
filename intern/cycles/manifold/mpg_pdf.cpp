/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_pdf.h"

#include "kernel/light/common.h"
#include "kernel/light/sample.h"
#include "kernel/types.h"

#include <cfloat>

#include "util/math.h"

CCL_NAMESPACE_BEGIN

float mpg_light_sample_pdf_solid(KernelGlobals kg,
                                 const ShaderData &sd,
                                 const LightSample &light_sample)
{
  LightSample ls = light_sample;
  light_sample_update(kg, &ls, sd.P, sd.Ng, PATH_RAY_SHADOW);
  return (isfinite_safe(ls.pdf) && ls.pdf > 0.0f) ? ls.pdf : 0.0f;
}

bool mpg_evaluate_pdf(KernelGlobals kg,
                      const ShaderData &sd,
                      const ShaderClosure &bsdf,
                      const GuideSummary &guide,
                      const MpgSeedRay &seed,
                      const MpgSolverOutput &solution,
                      float &pdf)
{
  pdf = 0.0f;
  const float p_seed  = fmaxf(seed.seed_pdf, 1.0e-16f);
  const float p_light = mpg_light_sample_pdf_solid(kg, sd, seed.light_sample);
  if (!(isfinite_safe(p_light) && p_light > 0.0f)) {
    return false;
  }

  const float J = fabsf(solution.jacobian_total);
  if (!(isfinite_safe(J) && J > 0.0f)) {
    return false;
  }

  const float p = p_seed * p_light * J;
  if (!(isfinite_safe(p) && p > 0.0f)) {
    return false;
  }
  pdf = p;
  return true;
}

CCL_NAMESPACE_END