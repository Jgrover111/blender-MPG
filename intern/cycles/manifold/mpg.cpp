/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg.h"

#include "manifold/mpg_types.h"

#include "manifold/mpg_pdf.h"
#include "manifold/mpg_seed.h"
#include "manifold/mpg_solve.h"

#include "kernel/light/common.h"
#include "kernel/light/light.h"
#include "kernel/light/distribution.h"
#include "kernel/light/tree.h"
#include "kernel/device/cpu/globals.h"
#include "kernel/integrator/path_state.h"
#include "kernel/svm/types.h"
#include "kernel/types.h"

#include <cfloat>
#include <cmath>

#ifdef WITH_CYCLES_DEBUG
#  include "util/log.h"
#endif

CCL_NAMESPACE_BEGIN

namespace {

GuideSummary sanitize_guide_summary(const GuideSummary &input)
{
  GuideSummary result = input;

  if (!(isfinite_safe(result.mean_dir.x) && isfinite_safe(result.mean_dir.y) &&
        isfinite_safe(result.mean_dir.z)))
  {
    result.mean_dir = zero_float3();
  }

  if (!isfinite_safe(result.peak_weight)) {
    result.peak_weight = 0.0f;
  }
  else {
    result.peak_weight = fminf(fmaxf(result.peak_weight, 0.0f), 1.0f);
  }

  if (!isfinite_safe(result.kappa) || result.kappa < 0.0f) {
    result.kappa = 0.0f;
  }

  if (!isfinite_safe(result.rbar)) {
    result.rbar = 0.0f;
  }
  else {
    result.rbar = fminf(fmaxf(result.rbar, 0.0f), 1.0f);
  }

  return result;
}

float3 compute_distant_light_endpoint(const LightSample &light_sample,
                                      const float3 &origin)
{
  float3 dir = light_sample.D;
  if (is_zero(dir)) {
    return origin;
  }
  dir = normalize(dir);
  return origin + dir * MPG_DISTANT_LIGHT_VISIBILITY_DISTANCE;
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
    const float3 vertex_geometric_normal = safe_normalize(cross(vertex.dXdu, vertex.dXdv));
    const float3 visibility_normal = is_zero(vertex_geometric_normal) ? vertex.normal : vertex_geometric_normal;
    visibility *= mpg_compute_segment_visibility(kg,
                                                 segment_start,
                                                 segment_normal,
                                                 vertex.position,
                                                 sd.time,
                                                 skip_object,
                                                 skip_prim,
                                                 OBJECT_NONE,
                                                 PRIM_NONE);
    if (visibility == 0.0f) {
      return 0.0f;
    }

    segment_start = vertex.position;
    segment_normal = visibility_normal;
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
                                               skip_prim,
                                               light_sample.object,
                                               light_sample.prim);

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

  const GuideSummary guide = sanitize_guide_summary(g);

  if (opt.max_bounces <= 0) {
    result.failure_code = MPG_FAILURE_UNSUPPORTED;
    return result;
  }

  if (CLOSURE_IS_BSDF_SINGULAR(bsdf.type) && !CLOSURE_IS_RAY_PORTAL(bsdf.type)) {
    result.failure_code = MPG_FAILURE_UNSUPPORTED;
    return result;
  }

  const bool gate_active = (opt.gate_w > 0.0f) || (opt.gate_kappa > 0.0f);
  const bool has_direction_relaxed = (guide.rbar > 1.0e-4f);
  const bool has_direction_strict = (guide.rbar > 1.0e-3f);
  const bool strict_gate = has_direction_strict && (guide.peak_weight >= opt.gate_w) &&
                           (guide.kappa >= opt.gate_kappa);
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
  /* When the gate is disabled we still attempt a bootstrap seed even if the guide has no
   * dominant direction. This mirrors the Mitsuba reference fallback behaviour. */

  MpgSeedRay seed;
  MpgFailureCode seed_failure = MPG_FAILURE_NONE;
  if (!mpg_generate_seed(
          kg, sd, bsdf, guide, opt, path_flag, bounce, rng_state, seed, seed_failure))
  {
    result.failure_code = (seed_failure != MPG_FAILURE_NONE) ? seed_failure : MPG_FAILURE_SEED;
    result.attempt_count = 0;
    return result;
  }
  result.seed_pdf_raw = seed.seed_pdf_raw;
  result.seed_trial_count = seed.trial_count;
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
  if (!(isfinite_safe(pdf_selection) && pdf_selection > 0.0f)) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
    result.light_pdf = 0.0f;
    return result;
  }

  /* Validate the stored selection probability before reusing it. The Mitsuba
   * reference keeps the receiver-side probability, so only perform light-link
   * compatibility checks here to preserve existing failure codes. */
  /* Light linking is evaluated at the diffuse receiver. Keep the original object id
   * from the shading point instead of the specular vertex to avoid rejecting valid
   * connections when intermediate specular surfaces belong to a different object. */
  const int receiver_object = light_link_receiver_nee(kg, &sd);
  const int emitter_object = light_sample.object;

#ifdef __LIGHT_TREE__
  if (kernel_data.integrator.use_light_tree) {
    if (light_sample.emitter_id < 0) {
      result.attempt_count = attempt_count;
      result.failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
      result.light_pdf = 0.0f;
      return result;
    }
  }
  else
#endif
  {
    if (!light_link_object_match(kg, receiver_object, emitter_object)) {
      result.attempt_count = attempt_count;
      result.failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
      result.light_pdf = 0.0f;
      return result;
    }
  }

  /* `light_sample_update` expects `ls->pdf` without the selection term. */
  light_sample.pdf /= pdf_selection;

  uint32_t updated_path_flag = path_flag;
  if (solution.is_refraction) {
    updated_path_flag |= PATH_RAY_MIS_HAD_TRANSMISSION;
  }

  light_sample.pdf_selection = 1.0f;
  light_sample_update(kg, &light_sample, exit_vertex.position, exit_vertex.normal, updated_path_flag);
  /* Restore the receiver-side selection probability rather than recomputing
   * it at the specular vertex. This matches the Mitsuba reference measure and
   * keeps the technique PDF consistent with the stored seed. */
  light_sample.pdf *= pdf_selection;
  light_sample.pdf_selection = pdf_selection;

  LightSample tmp = seed.light_sample;
  /* Recompute the pre-MPG NEE pdf at the receiver. The light update routine expects
   * `ls->pdf` without the light-selection probability and multiplies it back in, so
   * temporarily strip it to avoid squaring the factor. */
  tmp.pdf /= pdf_selection;
  tmp.pdf_selection = 1.0f;
  light_sample_update(kg, &tmp, sd.P, sd.N, path_flag);
  tmp.pdf *= pdf_selection;
  tmp.pdf_selection = pdf_selection;
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
  nee_pdf_sa = fmaxf(nee_pdf_sa, 0.0f);
  tmp.pdf = nee_pdf_sa;
  result.nee_pdf = nee_pdf_sa;

  float p_light = mpg_light_sample_pdf_solid(kg, sd, light_sample);
  if (!isfinite_safe(p_light) || p_light <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
    result.light_pdf = p_light;
    return result;
  }

  /* Prevent vanishing light pdfs from creating arbitrarily large MIS weights. */
  p_light = fmaxf(p_light, 1.0e-16f);

  light_sample.pdf = p_light;

  result.visibility = compute_visibility_after_update(kg, sd, light_sample, result);

  const float p_seed = (isfinite_safe(seed.seed_pdf)) ? fmaxf(seed.seed_pdf, 1.0e-16f) : 0.0f;
#ifdef WITH_CYCLES_DEBUG
  if (seed.accepted_trial_count > 0) {
    const float mitsuba_seed_pdf = seed.seed_pdf_raw * float(seed.accepted_trial_count);
    if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
      LOG_DEBUG << "MPG seed pdf parity (trials=" << seed.accepted_trial_count
                << "): cycles=" << p_seed << ", Mitsuba=" << mitsuba_seed_pdf
                << ", raw=" << seed.seed_pdf_raw;
    }
    if (isfinite_safe(p_seed) && isfinite_safe(mitsuba_seed_pdf)) {
      const float tolerance = fmaxf(fabsf(mitsuba_seed_pdf), 1.0e-16f) * 1.0e-4f;
      DCHECK(fabsf(p_seed - mitsuba_seed_pdf) <= tolerance);
    }
  }
#endif
  if (p_seed <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    result.seed_pdf = p_seed;
    return result;
  }

  result.seed_pdf = p_seed;

  const float J_total = (isfinite_safe(solution.jacobian_total)) ?
                            fmaxf(fabsf(solution.jacobian_total), 1.0e-16f) :
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
  result.seed_pdf_raw = seed.seed_pdf_raw;
  result.light_pdf = p_light;
  result.light = light_sample;
  result.attempt_count = attempt_count;
  result.seed_trial_count = seed.trial_count;
  result.seed_accepted_trial_count = seed.accepted_trial_count;

  return result;
}

CCL_NAMESPACE_END