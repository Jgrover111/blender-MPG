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

  const float pdf_native = light_sample.pdf;
  if (!isfinite_safe(pdf_native) || pdf_native <= 0.0f) {
    return 0.0f;
  }

  const bool finite_distance = isfinite_safe(light_sample.t) && light_sample.t != FLT_MAX;
  const LightType light_type = light_sample.type;

  if (!finite_distance || light_type == LIGHT_BACKGROUND || light_type == LIGHT_DISTANT) {
    return pdf_native;
  }

  const float3 Ng = make_float3(light_sample.Ng.x, light_sample.Ng.y, light_sample.Ng.z);
  const float3 I = -light_sample.D;
  const float area_to_solid = light_pdf_area_to_solid_angle(Ng, I, light_sample.t);
  if (!isfinite_safe(area_to_solid) || area_to_solid <= 0.0f) {
    return 0.0f;
  }

  return pdf_native * area_to_solid;
}

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

  if (solution.specular_vertex_count <= 0) {
    return false;
  }

  if (!isfinite_safe(solution.jacobian_total) || solution.jacobian_total <= 0.0f) {
    return false;
  }

  const MpgSpecularVertex &entry_vertex = solution.specular_vertices[0];
  const MpgSpecularVertex &exit_vertex =
      solution.specular_vertices[solution.specular_vertex_count - 1];

  if (!isfinite_safe(exit_vertex.distance_out) || exit_vertex.distance_out <= 0.0f) {
    return false;
  }

  const float3 emitter_normal = make_float3(
      seed.light_sample.Ng.x, seed.light_sample.Ng.y, seed.light_sample.Ng.z);
  const float cos_light = fabsf(dot(emitter_normal, -exit_vertex.dir_out));
  if (!isfinite_safe(cos_light) || cos_light <= 0.0f) {
    return false;
  }

  const float spec_geo_term = fabsf(dot(entry_vertex.normal, -solution.wi));
  if (!isfinite_safe(spec_geo_term) || spec_geo_term <= 0.0f) {
    return false;
  }

  LightSample light = seed.light_sample;
  const float pdf_selection = light.pdf_selection;
  if (pdf_selection != 0.0f) {
    light.pdf /= pdf_selection;
  }

  uint32_t updated_path_flag = seed.path_flag;
  if (solution.is_refraction) {
    updated_path_flag |= PATH_RAY_MIS_HAD_TRANSMISSION;
  }

  light_sample_update(kg, &light, exit_vertex.position, exit_vertex.normal, updated_path_flag);

  /* `light_sample_update()` already folded in the selection probability. Re-applying it would
   * shrink the MPG technique pdf and bias MIS towards the other proposals. */
  const float light_pdf_solid = mpg_light_sample_pdf_solid(kg, sd, light);
  if (light_pdf_solid <= 0.0f) {
    return false;
  }

  /* Compose the solid-angle pdf used in MIS with BSDF/guided/NEE:
   *   p = p_seed(ω_d) * p_light(ω_l) * |det dF/duv| * |dX/du x dX/dv| / r_ds^2 */
  pdf = seed.seed_pdf * light_pdf_solid * solution.jacobian_total;
  if (!isfinite_safe(pdf) || pdf <= 0.0f) {
    pdf = 0.0f;
    return false;
  }
  return true;
}

CCL_NAMESPACE_END