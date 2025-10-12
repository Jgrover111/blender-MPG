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

constexpr int MPG_BOOTSTRAP_RNG_OFFSET = 128;

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

  const bool strict_gate_pass = (gate_mask & MPG_GATE_MASK_STRICT_PASS) != 0;
  const bool relaxed_gate_pass = (gate_mask & MPG_GATE_MASK_RELAX_PASS) != 0;
  const bool bootstrap_gate_pass = (gate_mask & MPG_GATE_MASK_BOOTSTRAP_PASS) != 0;
  const bool gate_permits_solver = strict_gate_pass || (opt.relax_gate &&
                                                        (relaxed_gate_pass || bootstrap_gate_pass));

  if (!gate_permits_solver) {
    result.failure_code = MPG_FAILURE_GATE;
    result.attempt_count = 0;
    return result;
  }
  /* When the gate is disabled we still attempt a bootstrap seed even if the guide has no
   * dominant direction. This mirrors the Mitsuba reference fallback behaviour. */

  MpgSeedRay seed;
  MpgFailureCode seed_failure = MPG_FAILURE_NONE;
  bool seed_success = mpg_generate_seed(
      kg, sd, bsdf, guide, opt, path_flag, bounce, rng_state, seed, seed_failure);

  bool attempted_bootstrap_fallback = false;
  bool used_bootstrap_seed = false;

  if (!seed_success && bootstrap_gate_pass) {
    attempted_bootstrap_fallback = true;

    GuideSummary bootstrap_summary = guide;
    bootstrap_summary.mean_dir = zero_float3();
    bootstrap_summary.peak_weight = 0.0f;
    bootstrap_summary.kappa = 0.0f;
    bootstrap_summary.rbar = 0.0f;

    MpgFailureCode bootstrap_failure = MPG_FAILURE_NONE;
    seed_success = mpg_generate_seed(kg,
                                     sd,
                                     bsdf,
                                     bootstrap_summary,
                                     opt,
                                     path_flag,
                                     bounce,
                                     rng_state,
                                     seed,
                                     bootstrap_failure,
                                     MPG_BOOTSTRAP_RNG_OFFSET);

    if (!seed_success) {
      if (bootstrap_failure != MPG_FAILURE_NONE) {
        seed_failure = bootstrap_failure;
      }
    }
    else {
      seed_failure = MPG_FAILURE_NONE;
      used_bootstrap_seed = true;
#ifdef WITH_CYCLES_DEBUG
      if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
        LOG_DEBUG << "MPG bootstrap gating fallback seed succeeded";
      }
#endif
    }
  }

  if (!seed_success) {
#ifdef WITH_CYCLES_DEBUG
    if (bootstrap_gate_pass && LOG_IS_ON(LOG_LEVEL_DEBUG)) {
      LOG_DEBUG << "MPG bootstrap gating exhausted seeds (fallback="
                << (attempted_bootstrap_fallback ? "yes" : "no")
                << ", failure=" << static_cast<int>(seed_failure) << ")";
    }
#endif
    result.failure_code = (seed_failure != MPG_FAILURE_NONE) ? seed_failure : MPG_FAILURE_SEED;
    result.attempt_count = 0;
    result.seed_trial_count = seed.trial_count;
    result.seed_accepted_trial_count = seed.accepted_trial_count;
    result.seed_guided_trial_count = seed.guided_trial_count;
    result.seed_fallback_trial_count = seed.fallback_trial_count;
    result.seed_pdf_raw = seed.seed_pdf_raw;
    result.seed_pdf = seed.seed_pdf;
    result.seed_resample_factor = seed.seed_resample_factor;
    result.seed_branch_pdf = seed.seed_branch_pdf;
    result.seed_direction_pdf = seed.seed_direction_pdf;
    result.seed_scatter_pdf = seed.seed_scatter_pdf;
    result.seed_direction_normalized = seed.direction_normalized;
    result.seed_tau_bits = seed.tau_bits;
    result.seed_tau_count = seed.tau_count;
    result.seed_branch = seed.branch;
    result.seed_scatter = seed.scatter;
    result.bounce_pdf_raw = seed.bounce_pdf_raw;
    result.bounce_pdf = seed.bounce_pdf;
    result.bounce_count = seed.bounce_count;
    return result;
  }
  result.seed_pdf_raw = seed.seed_pdf_raw;
  result.seed_trial_count = seed.trial_count;
  result.seed_accepted_trial_count = seed.accepted_trial_count;
  result.seed_guided_trial_count = seed.guided_trial_count;
  result.seed_fallback_trial_count = seed.fallback_trial_count;
  result.light = seed.light_sample;
  result.seed_pdf = seed.seed_pdf;
  result.seed_resample_factor = seed.seed_resample_factor;
  result.seed_branch_pdf = seed.seed_branch_pdf;
  result.seed_direction_pdf = seed.seed_direction_pdf;
  result.seed_scatter_pdf = seed.seed_scatter_pdf;
  result.seed_direction_normalized = seed.direction_normalized;
  result.seed_tau_bits = seed.tau_bits;
  result.seed_tau_count = seed.tau_count;
  result.seed_branch = seed.branch;
  result.seed_scatter = seed.scatter;
  result.bounce_pdf_raw = seed.bounce_pdf_raw;
  result.bounce_pdf = seed.bounce_pdf;
  result.bounce_count = seed.bounce_count;
  if (!is_zero(seed.direction)) {
    result.wi = normalize(seed.direction);
  }

  if (!gate_permits_solver) {
    result.failure_code = MPG_FAILURE_GATE;
    result.attempt_count = 0;
    return result;
  }

  MpgSolverOutput solution;
  bool solved = false;
  int attempt_count = 0;
  MpgFailureCode solver_failure = MPG_FAILURE_NONE;

  solution.seed_resample_factor = seed.seed_resample_factor;
  solution.seed_branch = seed.branch;
  solution.seed_branch_pdf = seed.seed_branch_pdf;
  solution.seed_direction_pdf = seed.seed_direction_pdf;
  solution.seed_scatter = seed.scatter;
  solution.seed_scatter_pdf = seed.seed_scatter_pdf;
  solution.seed_guided_trial_count = seed.guided_trial_count;
  solution.seed_fallback_trial_count = seed.fallback_trial_count;

  if (seed.bounce_count == 2) {
    ++attempt_count;
    solved = mpg_solve_double_bounce(kg, sd, bsdf, seed, opt, rng_state, solution, solver_failure);
    if (!solved && solver_failure != MPG_FAILURE_NONE) {
      result.failure_code = solver_failure;
    }
  }
  else {
    ++attempt_count;
    solved = mpg_solve_single_bounce(kg, sd, bsdf, seed, opt, rng_state, solution, solver_failure);
    if (!solved && solver_failure != MPG_FAILURE_NONE) {
      result.failure_code = solver_failure;
    }
  }

  result.attempt_count = attempt_count;

  if (!solved) {
    if (result.failure_code == MPG_FAILURE_NONE) {
      result.failure_code = (solver_failure != MPG_FAILURE_NONE) ? solver_failure : MPG_FAILURE_NO_SPECULAR;
    }
    result.attempt_count = attempt_count;
#ifdef WITH_CYCLES_DEBUG
    if (bootstrap_gate_pass && LOG_IS_ON(LOG_LEVEL_DEBUG)) {
      LOG_DEBUG << "MPG bootstrap gate solver attempts=" << attempt_count
                << " (used_bootstrap_seed=" << (used_bootstrap_seed ? "yes" : "no")
                << ", failure=" << static_cast<int>(result.failure_code) << ")";
    }
#endif
    return result;
  }

  if (solution.specular_vertex_count <= 0) {
    result.failure_code = MPG_FAILURE_NO_SPECULAR;
    result.attempt_count = attempt_count;
    return result;
  }

  result.visibility = solution.visibility;
  result.spec_weight = solution.specular_throughput;
  result.specular_vertex_count = solution.specular_vertex_count;
  result.bounce_count = solution.specular_vertex_count;
  for (int i = 0; i < solution.specular_vertex_count; ++i) {
    result.specular_vertices[i] = solution.specular_vertices[i];
  }

  const MpgSpecularVertex &exit_vertex =
      solution.specular_vertices[solution.specular_vertex_count - 1];
  const bool exit_is_refraction = exit_vertex.is_refraction;
#ifdef WITH_CYCLES_DEBUG
  DCHECK(solution.is_refraction == exit_is_refraction);
#endif

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
  if (exit_is_refraction) {
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
  /* Recompute the pre-MPG NEE pdf at the receiver. Strip the selection probability while the
   * light sample is updated so it is re-applied exactly once. */
  tmp.pdf /= pdf_selection;
  tmp.pdf_selection = 1.0f;
  light_sample_update(kg, &tmp, sd.P, sd.N, path_flag);
  float nee_pdf_sa = mpg_light_sample_pdf_solid(kg, sd, tmp);
  nee_pdf_sa *= pdf_selection;
  if (!isfinite_safe(nee_pdf_sa)) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_NEE_PDF;
    return result;
  }
  if (nee_pdf_sa <= 0.0f) {
    nee_pdf_sa = 0.0f;
  }
  result.nee_pdf = nee_pdf_sa;
  tmp.pdf = nee_pdf_sa;

  float p_light = mpg_light_sample_pdf_solid(kg, sd, light_sample);
  result.light_pdf = p_light;
  if (!isfinite_safe(p_light) || p_light <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
    return result;
  }
  light_sample.pdf = p_light;

  result.visibility = compute_visibility_after_update(kg, sd, light_sample, result);

  const float p_seed = fmaxf(seed.seed_pdf, 1.0e-16f);
  result.seed_pdf = p_seed;
#ifdef WITH_CYCLES_DEBUG
  const int total_trials = seed.trial_count;
  const float expected_trials =
      (isfinite_safe(seed.seed_resample_factor) && seed.seed_resample_factor > 0.0f) ?
          seed.seed_resample_factor :
          0.0f;
  const float acceptance_probability = (expected_trials > 0.0f) ?
                                           (1.0f / expected_trials) :
                                           0.0f;
  const float mitsuba_seed_pdf = mpg_rebuild_seed_pdf(seed);
  if (total_trials > 0 && mitsuba_seed_pdf > 0.0f) {
    if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
      DCHECK(seed.scatter != MPG_SEED_SCATTER_NONE);
      DCHECK(seed.seed_scatter_pdf > 0.0f);
      LOG_DEBUG << "MPG seed pdf parity (trials=" << total_trials
                << "): cycles=" << p_seed << ", Mitsuba=" << mitsuba_seed_pdf
                << ", raw=" << seed.seed_pdf_raw
                << ", expected_trials=" << expected_trials
                << ", accept_p=" << acceptance_probability
                << ", branch=" << seed.seed_branch_pdf
                << ", dir=" << seed.seed_direction_pdf
                << ", scatter=" << seed.seed_scatter_pdf
                << ", scatter_branch=" << static_cast<int>(seed.scatter)
                << ", bounce=" << seed.bounce_pdf << " (folded into seed)"
                << ", bounce_raw=" << seed.bounce_pdf_raw;
    }
    if (isfinite_safe(p_seed) && isfinite_safe(mitsuba_seed_pdf)) {
      const float tolerance = fmaxf(fabsf(mitsuba_seed_pdf), 1.0e-16f) * 1.0e-4f;
      DCHECK(fabsf(p_seed - mitsuba_seed_pdf) <= tolerance);
    }
  }
#endif
  if (!isfinite_safe(p_seed) || p_seed <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_SEED_PDF;
    return result;
  }

  const float raw_jacobian = solution.jacobian_total;
  const float J_total = fabsf(raw_jacobian);
  if (!isfinite_safe(J_total) || J_total <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_JACOBIAN_ZERO;
    result.jacobian_total = raw_jacobian;
    return result;
  }
  result.jacobian_total = J_total;
  result.light_pdf *= result.jacobian_total;

  if (!isfinite_safe(seed.bounce_pdf) || seed.bounce_pdf <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_PDF;
    return result;
  }
  result.bounce_pdf = seed.bounce_pdf;
  result.bounce_pdf_raw = seed.bounce_pdf_raw;

  const float pdf_product = p_seed * result.light_pdf;
  result.pdf = pdf_product;
  if (!isfinite_safe(pdf_product) || pdf_product <= 0.0f) {
    result.attempt_count = attempt_count;
    result.failure_code = MPG_FAILURE_INVALID_PDF;
    return result;
  }

#ifdef WITH_CYCLES_DEBUG
  {
    float eval_pdf = 0.0f;
    const bool eval_success = mpg_evaluate_pdf(kg, sd, bsdf, guide, seed, solution, eval_pdf);
    DCHECK(eval_success);
    if (eval_success) {
      const float tolerance = fmaxf(fabsf(result.pdf), 1.0e-16f) * 1.0e-4f;
      DCHECK(fabsf(eval_pdf - result.pdf) <= tolerance);
    }
  }
#endif

  result.success = true;
  result.failure_code = MPG_FAILURE_NONE;
  result.wi = solution.wi;
  result.seed_pdf = p_seed;
  result.seed_pdf_raw = seed.seed_pdf_raw;
  result.bounce_pdf = seed.bounce_pdf;
  result.bounce_pdf_raw = seed.bounce_pdf_raw;
  result.light = light_sample;
  result.attempt_count = attempt_count;
  result.seed_trial_count = seed.trial_count;
  result.seed_accepted_trial_count = seed.accepted_trial_count;

#ifdef WITH_CYCLES_DEBUG
  if (bootstrap_gate_pass && LOG_IS_ON(LOG_LEVEL_DEBUG)) {
    LOG_DEBUG << "MPG bootstrap gate solver attempts=" << attempt_count
              << " (used_bootstrap_seed=" << (used_bootstrap_seed ? "yes" : "no")
              << ", success)";
  }
  if ((result.gate_mask & MPG_GATE_MASK_STRICT_PASS) != 0) {
    DCHECK(isfinite_safe(result.seed_pdf) && result.seed_pdf > 0.0f);
    DCHECK(isfinite_safe(result.light_pdf) && result.light_pdf > 0.0f);
    DCHECK(isfinite_safe(result.bounce_pdf) && result.bounce_pdf > 0.0f);
    DCHECK(isfinite_safe(result.jacobian_total) && result.jacobian_total > 0.0f);
    DCHECK(isfinite_safe(result.pdf) && result.pdf > 0.0f);
    if (LOG_IS_ON(LOG_LEVEL_DEBUG)) {
      LOG_DEBUG << "MPG strict gate factors: seed(with bounce)=" << result.seed_pdf
                << ", bounce=" << result.bounce_pdf
                << ", light_receiver=" << result.light_pdf
                << " (light_spec=" << p_light << ")"
                << ", J=" << result.jacobian_total << ", pdf=" << result.pdf;
    }
  }
#endif

  return result;
}

CCL_NAMESPACE_END