/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_pdf.h"

#include "manifold/mpg_seed.h"

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
      float3 light_dir = light_sample.D;
      if (is_zero(light_dir)) {
        return 0.0f;
      }
      light_dir = normalize(light_dir);

      const float jacobian = light_pdf_area_to_solid_angle(
          light_sample.Ng, -light_dir, light_sample.t);

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
  (void)bsdf;
  (void)guide;

  pdf = 0.0f;

  const float p_seed = seed.seed_pdf;
  if (!(isfinite_safe(p_seed) && p_seed > 0.0f)) {
    return false;
  }

  const int vertex_count = solution.specular_vertex_count;
  if (vertex_count <= 0) {
    return false;
  }

  LightSample light_sample = seed.light_sample;
  const float pdf_selection = light_sample.pdf_selection;
  if (!(isfinite_safe(pdf_selection) && pdf_selection > 0.0f)) {
    return false;
  }

  light_sample.pdf /= pdf_selection;
  light_sample.pdf_selection = 1.0f;

  const MpgSpecularVertex &exit_vertex = solution.specular_vertices[vertex_count - 1];
  uint32_t updated_path_flag = seed.path_flag;
  if (solution.is_refraction) {
    updated_path_flag |= PATH_RAY_MIS_HAD_TRANSMISSION;
  }
  light_sample_update(kg, &light_sample, exit_vertex.position, exit_vertex.normal, updated_path_flag);

  light_sample.pdf *= pdf_selection;
  light_sample.pdf_selection = pdf_selection;

  float p_light = mpg_light_sample_pdf_solid(kg, sd, light_sample);
  if (!isfinite_safe(p_light) || p_light <= 0.0f) {
    return false;
  }
  p_light = fmaxf(p_light, 1.0e-16f);

  float J = fabsf(solution.jacobian_total);
  if (!isfinite_safe(J) || J <= 0.0f) {
    return false;
  }
  J = fmaxf(J, 1.0e-16f);

  const float pdf_product = p_seed * p_light * J;
  if (!isfinite_safe(pdf_product) || pdf_product <= 0.0f) {
    return false;
  }

  pdf = fmaxf(pdf_product, 1.0e-16f);
  return true;
}

CCL_NAMESPACE_END