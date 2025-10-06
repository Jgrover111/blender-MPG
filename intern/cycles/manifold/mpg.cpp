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
#include <cmath>

CCL_NAMESPACE_BEGIN

namespace {

float3 compute_distant_light_endpoint(const LightSample &light_sample,
                                      const float3 &origin)
{
  float3 dir = light_sample.D;
  if (is_zero(dir)) {
    return origin;
  }
  dir = normalize(dir);
  const float distant_length = 1.0e6f;
  return origin + dir * distant_length;
}

float compute_visibility_after_update(KernelGlobals kg,
                                      const ShaderData &sd,
                                      const LightSample &light_sample,
                                      const MpgResult &result)
{
  if (result.specular_vertex_count <= 0) {
    return 0.0f;
  }

  float visibility = 1.0f;
  float3 segment_start = sd.P;
  float3 segment_normal = sd.Ng;
  int skip_object = sd.object;
  int skip_prim = sd.prim;

  for (int i = 0; i < result.specular_vertex_count; ++i) {
    const MpgSpecularVertex &vertex = result.specular_vertices[i];
    visibility *= mpg_compute_segment_visibility(kg,
                                                 segment_start,
                                                 segment_normal,
                                                 vertex.position,
                                                 sd.time,
                                                 skip_object,
                                                 skip_prim);
    if (visibility == 0.0f) {
      return 0.0f;
    }

    segment_start = vertex.position;
    segment_normal = vertex.normal;
    skip_object = vertex.object;
    skip_prim = vertex.prim;
  }

  float3 light_point = light_sample.P;
  if (light_sample.t == FLT_MAX) {
    light_point = compute_distant_light_endpoint(light_sample, segment_start);
  }

  visibility *= mpg_compute_segment_visibility(kg,
                                               segment_start,
                                               segment_normal,
                                               light_point,
                                               sd.time,
                                               skip_object,
                                               skip_prim);

  return visibility;
}

}  // namespace

MpgResult mpg_try_connect(KernelGlobals kg,
                          const ShaderData &sd,
                          const ShaderClosure &bsdf,
                          const GuideSummary &g,
                          const MpgOptions &opt,
                          const uint32_t path_flag,
                          const int bounce,
                          RNGState &rng_state)
{
  MpgResult result{};
  result.failure_code = MPG_FAILURE_NONE;
  result.attempt_count = 0;
  result.gate_mask = MPG_GATE_MASK_NONE;

  if (opt.max_bounces <= 0) {
    result.failure_code = MPG_FAILURE_UNSUPPORTED;
    return result;
  }

  if (CLOSURE_IS_BSDF_SINGULAR(bsdf.type) && !CLOSURE_IS_RAY_PORTAL(bsdf.type)) {
    result.failure_code = MPG_FAILURE_UNSUPPORTED;
    return result;
  }

  const bool gate_active = (opt.gate_w > 0.0f) || (opt.gate_kappa > 0.0f);
  const bool has_direction_relaxed = (g.rbar > 1.0e-4f);
  const bool has_direction_strict = (g.rbar > 1.0e-3f);
  const bool strict_gate = has_direction_strict && (g.peak_weight >= opt.gate_w) &&
                           (g.kappa >= opt.gate_kappa);
  const bool relaxed_gate = opt.relax_gate && has_direction_relaxed;
  const bool bootstrap_gate = opt.relax_gate && !has_direction_relaxed;

  uint32_t gate_mask = MPG_GATE_MASK_NONE;
  if (gate_active) {
    gate_mask |= MPG_GATE_MASK_ACTIVE;
  }
  if (has_direction_relaxed) {
    gate_mask |= MPG_GATE_MASK_HAS_DIRECTION_RELAXED;
  }
  if (has_direction_strict) {
    gate_mask |= MPG_GATE_MASK_HAS_DIRECTION_STRICT;
  }
  if (strict_gate) {
    gate_mask |= MPG_GATE_MASK_STRICT_PASS;
  }
  if (relaxed_gate) {
    gate_mask |= MPG_GATE_MASK_RELAX_PASS;
  }
  if (bootstrap_gate) {
    gate_mask |= MPG_GATE_MASK_BOOTSTRAP_PASS;
  }
  if (!gate_active) {
    gate_mask |= MPG_GATE_MASK_STRICT_PASS;
  }
  result.gate_mask = gate_mask;

  if (gate_active) {
    if (!strict_gate && !relaxed_gate && !bootstrap_gate) {
      result.failure_code = MPG_FAILURE_GATE;
      result.attempt_count = 0;
      return result;
    }
  }
  else if (g.rbar <= 1.0e-5f) {
    /* Without guiding gate we still require a numerically stable direction. */
    result.failure_code = MPG_FAILURE_GATE;
    result.attempt_count = 0;
    return result;
  }

  MpgSeedRay seed;
  MpgFailureCode seed_failure = MPG_FAILURE_NONE;
  if (!mpg_generate_seed(kg, sd, bsdf, g, opt, path_flag, bounce, rng_state, seed, seed_failure)) {
    result.failure_code = (seed_failure != MPG_FAILURE_NONE) ? seed_failure : MPG_FAILURE_SEED;
    result.attempt_count = 0;
    return result;
  }
  result.seed_pdf = seed.seed_pdf;
  result.light = seed.light_sample;
  if (!is_zero(seed.direction)) {
    result.wi = normalize(seed.direction);
  }

  MpgSolverOutput solution;
  bool solved = false;
  int attempt_count = 0;
  MpgFailureCode solver_failure = MPG_FAILURE_NONE;

  if (opt.max_bounces >= 2) {
    ++attempt_count;
    solved = mpg_solve_double_bounce(kg, sd, bsdf, seed, opt, rng_state, solution, solver_failure);
    if (!solved) {
      result.failure_code = solver_failure;
    }
  }

  if (!solved) {
    ++attempt_count;
    solved = mpg_solve_single_bounce(kg, sd, bsdf, seed, opt, rng_state, solution, solver_failure);
    if (!solved) {
      result.failure_code = solver_failure;
    }
  }

  result.attempt_count = attempt_count;

  if (!solved) {
    if (result.failure_code == MPG_FAILURE_NONE) {
      result.failure_code = (solver_failure != MPG_FAILURE_NONE) ? solver_failure : MPG_FAILURE_NO_SPECULAR;
    }
    result.attempt_count = attempt_count;
    return result;
  }

  if (solution.specular_vertex_count <= 0) {
    result.failure_code = MPG_FAILURE_NO_SPECULAR;
    result.attempt_count = attempt_count;
    return result;
  }

  result.jacobian_total = solution.jacobian_total;
  result.visibility = solution.visibility;
  result.spec_weight = solution.specular_throughput;
  result.specular_vertex_count = solution.specular_vertex_count;
  for (int i = 0; i < solution.specular_vertex_count; ++i) {
    result.specular_vertices[i] = solution.specular_vertices[i];
  }

  const MpgSpecularVertex &exit_vertex =
      solution.specular_vertices[solution.specular_vertex_count - 1];

  LightSample light_sample = seed.light_sample;
  const float pdf_selection = light_sample.pdf_selection;
  const bool has_selection_pdf = pdf_selection > 0.0f;
  if (has_selection_pdf) {
    /* `light_sample_update` expects `ls->pdf` without the selection term. */
    light_sample.pdf /= pdf_selection;
  }

  uint32_t updated_path_flag = path_flag;
  if (solution.is_refraction) {
    updated_path_flag |= PATH_RAY_MIS_HAD_TRANSMISSION;
  }

  LightSample light_sa = light_sample;
  light_sample_update(kg, &light_sample, exit_vertex.position, exit_vertex.normal, updated_path_flag);
  LightSample tmp = light_sa;
  light_sample_update(kg, &tmp, sd.P, sd.N, path_flag);
  float nee_pdf_sa = mpg_light_sample_pdf_solid(kg, sd, tmp);
  if (!isfinite_safe(nee_pdf_sa)) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_NEE_PDF;
    result.nee_pdf = 0.0f;
    return result;
  }
  if (nee_pdf_sa < 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_NEE_PDF;
    result.nee_pdf = 0.0f;
    return result;
  }
  if (has_selection_pdf) {
    nee_pdf_sa *= pdf_selection;
  }
  nee_pdf_sa = fmaxf(nee_pdf_sa, 0.0f);
  tmp.pdf = nee_pdf_sa;
  result.nee_pdf = nee_pdf_sa;

  float p_light = mpg_light_sample_pdf_solid(kg, sd, light_sample);
  if (has_selection_pdf) {
    p_light *= pdf_selection;
  }
  if (p_light <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
    result.light_pdf = p_light;
    return result;
  }

  light_sample.pdf = p_light;

  result.visibility = compute_visibility_after_update(kg, sd, light_sample, result);

  const float p_seed = (isfinite_safe(seed.seed_pdf)) ? fmaxf(seed.seed_pdf, 1.0e-16f) : 0.0f;
  if (p_seed <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    result.seed_pdf = p_seed;
    return result;
  }

  const float J_total = (isfinite_safe(solution.jacobian_total)) ?
                            fmaxf(fabsf(solution.jacobian_total), 0.0f) :
                            0.0f;
  if (J_total <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    return result;
  }

  const float pdf = p_seed * p_light * J_total;
  if (!isfinite_safe(pdf) || pdf <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_PDF;
    return result;
  }

  result.success = true;
  result.failure_code = MPG_FAILURE_NONE;
  result.wi = solution.wi;
  result.pdf = pdf;
  result.seed_pdf = p_seed;
  result.light_pdf = p_light;
  result.light = light_sample;
  result.attempt_count = attempt_count;

  return result;
}

CCL_NAMESPACE_END