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
  if (!isfinite_safe(solution.visibility) || solution.visibility <= 0.0f) {
    return false;
  }

  if (!isfinite_safe(seed.seed_pdf) || seed.seed_pdf <= 0.0f) {
    return false;
  }

  if (!isfinite_safe(solution.jacobian) || solution.jacobian <= 0.0f) {
    return false;
  }

  if (!isfinite_safe(solution.distance_sl) || solution.distance_sl <= 0.0f) {
    return false;
  }

  const float3 emitter_normal = make_float3(
      seed.light_sample.Ng.x, seed.light_sample.Ng.y, seed.light_sample.Ng.z);
  const float cos_light = fabsf(dot(emitter_normal, -solution.dir_sl));
  if (!isfinite_safe(cos_light) || cos_light <= 0.0f) {
    return false;
  }

  const float spec_geo_term = fabsf(dot(solution.specular_normal, -solution.wi));
  if (!isfinite_safe(spec_geo_term) || spec_geo_term <= 0.0f) {
    return false;
  }

  const float light_pdf_solid = seed.light_sample.pdf;
  if (!isfinite_safe(light_pdf_solid) || light_pdf_solid <= 0.0f) {
    return false;
  }

  /* Compose the solid-angle pdf used in MIS with BSDF/guided/NEE:
  *   p = p_seed(ω_d) * p_light(ω_l) * |det dF/duv| * |dX/du x dX/dv| / r_ds^2 */
  pdf = seed.seed_pdf * light_pdf_solid * solution.jacobian;
  if (!isfinite_safe(pdf) || pdf <= 0.0f) {
    pdf = 0.0f;
    return false;
  }
  return true;
}

CCL_NAMESPACE_END