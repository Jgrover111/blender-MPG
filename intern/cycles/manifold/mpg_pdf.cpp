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
  (void)kg;
  (void)sd;

  if (!(isfinite_safe(light_sample.pdf) && light_sample.pdf > 0.0f)) {
    return 0.0f;
  }

  float pdf_solid = light_sample.pdf;

  if (light_sample.t != FLT_MAX) {
    bool needs_conversion = false;

    switch (light_sample.type) {
      case LIGHT_POINT:
      case LIGHT_SPOT:
        needs_conversion = true;
        break;
      case LIGHT_AREA:
        if (light_sample.prim != PRIM_NONE) {
          const ccl_global KernelLight *klight = &kernel_data_fetch(lights, light_sample.prim);
          /* Area lights that sample from an elliptical shape keep the PDF in area
           * measure even when the spread is zero. Convert those (and any spread
           * configurations) back to solid angle to match Mitsuba's measure. */
          const bool is_elliptical = (klight->area.invarea < 0.0f);
          const bool has_spread = (klight->area.tan_half_spread != 0.0f);
          needs_conversion = (has_spread || is_elliptical);
        }
        break;
      default:
        break;
    }

    if (needs_conversion) {
      const float jacobian = light_pdf_area_to_solid_angle(
          light_sample.Ng, -light_sample.D, light_sample.t);

      if (!(isfinite_safe(jacobian) && jacobian > 0.0f)) {
        return 0.0f;
      }

      pdf_solid *= jacobian;
    }
  }

  return pdf_solid;
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