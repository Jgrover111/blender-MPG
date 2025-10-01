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
  if (!(isfinite_safe(light_sample.pdf)  && light_sample.pdf > 0.0f)) {
    return 0.0f;
  }

  float pdf = light_sample.pdf;

  const bool is_area_like = (len_squared(light_sample.Ng) > 0.0f);

  if (is_area_like) {
    const float3 L = light_sample.P - sd.P;
    const float dist2 = fmaxf(dot(L, L), 1.0e-8f);
    const float cos_l = fabsf(dot(light_sample.Ng, -light_sample.D));
    if (!(isfinite(cos_l) && cos_l > 1.0e-8f)) {
      return 0.0f;
    }
    pdf = pdf * (dist2 / cos_l);
  }
  return (isfinite(pdf) && pdf > 0.0f) ? pdf : 0.0f;
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