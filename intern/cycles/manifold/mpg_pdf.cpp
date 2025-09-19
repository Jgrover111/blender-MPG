/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_pdf.h"

CCL_NAMESPACE_BEGIN

bool mpg_evaluate_pdf(KernelGlobals kg,
                      const ShaderData &sd,
                      const ShaderClosure &bsdf,
                      const GuideSummary &guide,
                      const MpgSeedRay &seed,
                      const MpgSolverOutput &solution,
                      float &pdf)
{
  (void)kg;
  (void)sd;
  (void)bsdf;
  (void)guide;
  pdf = 0.0f;
  if (!solution.success) {
    return false;
  }
  if (seed.seed_pdf <= 0.0f || seed.emitter_pdf <= 0.0f || solution.jacobian <= 0.0f) {
    return false;
  }

  const float geo_term = fabsf(dot(solution.specular_normal, -solution.wi));
  if (geo_term <= 0.0f) {
    return false;
  }

  pdf = seed.seed_pdf * seed.emitter_pdf * solution.jacobian;
  if (!isfinite_safe(pdf) || pdf <= 0.0f) {
    pdf = 0.0f;
    return false;
  }
  return true;
}

CCL_NAMESPACE_END