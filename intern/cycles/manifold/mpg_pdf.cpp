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

  if (!isfinite_safe(seed.light_sample.pdf) || seed.light_sample.pdf <= 0.0f) {
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

  const float distance_sl_sq = solution.distance_sl * solution.distance_sl;
  if (!isfinite_safe(distance_sl_sq) || distance_sl_sq <= 0.0f) {
    return false;
  }

  const float emitter_area_pdf = seed.light_sample.pdf * cos_light / distance_sl_sq;
  if (!isfinite_safe(emitter_area_pdf) || emitter_area_pdf <= 0.0f) {
    return false;
  }

  const float spec_geo_term = fabsf(dot(solution.specular_normal, -solution.wi));
  if (!isfinite_safe(spec_geo_term) || spec_geo_term <= 0.0f) {
    return false;
  }

  pdf = seed.seed_pdf * emitter_area_pdf * solution.jacobian;
  if (!isfinite_safe(pdf) || pdf <= 0.0f) {
    pdf = 0.0f;
    return false;
  }
  return true;
}

CCL_NAMESPACE_END