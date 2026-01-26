/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/mnee.h"

/*
 * Specular Manifold Sampling (SMS)
 *
 * This code implements Specular Manifold Sampling for rendering caustic light paths through
 * specular surfaces. SMS extends MNEE's manifold walking infrastructure to support:
 * - Both reflective and refractive caustics
 * - Glossy caustics (not just perfect specular)
 * - Full S-D-S (Specular-Diffuse-Specular) paths
 * - Unbiased sampling via stochastic manifold exploration
 *
 * Unlike MNEE which uses deterministic initialization for shadow caustics, SMS uses:
 * - Stochastic initialization with random microfacet normals
 * - Bernoulli trials for unbiased probability estimation
 * - Support for both reflection and refraction BSDFs
 *
 * The algorithm (Unbiased SMS - Algorithm 2 from paper):
 * 1. Discover caustic caster chain via seed ray tracing
 * 2. Sample random microfacet normal offsets for stochastic initialization
 * 3. Solve for reference manifold solution using Newton solver
 * 4. Run Bernoulli trials with new random offsets until matching solution found
 * 5. Estimate probability as geometric series: p = 1 / (number of trials)
 * 6. Return contribution weighted by inverse probability: contribution / p
 *
 * References:
 * [1] "Specular Manifold Sampling for Rendering High-Frequency Caustics and Glints"
 *     Tizian Zeltner, Iliyan Georgiev, Wenzel Jakob (2020)
 * [2] "Manifold Next Event Estimation"
 *     Johannes Hanika, Marc Droske, Luca Fascione (2015)
 */

// NOLINTBEGIN

/* Unbiased SMS: Maximum number of Bernoulli trials for probability estimation.
 * Paper uses 64 trials. Higher values give better probability estimates but cost more. */
#define SMS_MAX_TRIALS 64

/* Tolerance for comparing vertex positions between trials.
 * Two solutions are considered matching if positions differ by less than this. */
#define SMS_POSITION_EPSILON 1e-5f

// NOLINTEND

CCL_NAMESPACE_BEGIN

/* Check if two manifold solutions match (same vertex positions).
 * Used to detect when a Bernoulli trial finds the reference solution. */
ccl_device_inline bool sms_solutions_match(
    const int vertex_count,
    ccl_private const ManifoldVertex *solution_a,
    ccl_private const ManifoldVertex *solution_b)
{
  for (int i = 0; i < vertex_count; i++) {
    const float dist = len(solution_a[i].p - solution_b[i].p);
    if (dist > SMS_POSITION_EPSILON) {
      return false;
    }
  }
  return true;
}

/* Sample random microfacet normal offsets for stochastic initialization.
 * This is the key difference from MNEE which uses deterministic (zero) offsets.
 * Returns true if offsets were sampled, false if roughness is zero (perfect specular). */
ccl_device_inline bool sms_sample_microfacet_offsets(
    KernelGlobals kg,
    ccl_private const RNGState *rng_state,
    const int vertex_count,
    ccl_private ShaderClosure **compatible_bsdfs,
    ccl_private float2 *h_offsets)
{
  bool has_roughness = false;

  for (int v_idx = 0; v_idx < vertex_count; v_idx++) {
    ccl_private ShaderClosure *bsdf = compatible_bsdfs[v_idx];
    h_offsets[v_idx] = zero_float2();

    if (!bsdf || !CLOSURE_IS_BSDF_MICROFACET(bsdf->type)) {
      continue;
    }

    ccl_private MicrofacetBsdf *microfacet_bsdf = (ccl_private MicrofacetBsdf *)bsdf;

    /* Only sample offset if material has roughness */
    if (microfacet_bsdf->alpha_x > 0.0f && microfacet_bsdf->alpha_y > 0.0f) {
      const float2 bsdf_uv = path_state_rng_2D(kg, rng_state, PRNG_SURFACE_BSDF);
      h_offsets[v_idx] = mnee_sample_bsdf_dh(
          bsdf->type,
          microfacet_bsdf->alpha_x,
          microfacet_bsdf->alpha_y,
          bsdf_uv.x,
          bsdf_uv.y);
      has_roughness = true;
    }
  }

  return has_roughness;
}

/* Unbiased SMS Sampling (Algorithm 2 from paper)
 *
 * Attempts to solve for a caustic path using stochastic manifold sampling with
 * unbiased probability estimation via Bernoulli trials.
 *
 * Parameters:
 *   kg: Kernel globals
 *   state: Current integrator state
 *   sd: Shader data at current vertex (diffuse receiver surface)
 *   sd_sms: Shader data for SMS vertices (updated during solve)
 *   rng_state: Random number generator state
 *   ls: Light sample (pre-sampled, may be updated by solver)
 *   throughput: BSDF evaluation to accumulate contribution into
 *
 * Returns:
 *   Number of manifold vertices in successful SMS path (> 0 on success)
 *   0 if SMS failed or is not applicable
 */
ccl_device_forceinline int kernel_path_sms_sample(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    ccl_private ShaderData *sd_sms,
    ccl_private const RNGState *rng_state,
    ccl_private LightSample *ls,
    ccl_private BsdfEval *throughput)
{
  /* ============================================================================
   * DEBUG CODE - TEMPORARY - REMOVE BEFORE PRODUCTION
   * Return negative values to encode failure reasons:
   * >= 1 = success (vertex count)
   * -1 = caustics mode not FULL
   * -2 = no vertices found during seed ray discovery
   * -3 = Newton solver failed to converge
   * -4 = path contribution failed
   * -5 = probability estimation failed
   * ============================================================================ */

  /* Check if SMS is enabled globally */
  if (kernel_data.integrator.caustics_mode != CAUSTICS_FULL) {
    return -1;  /* SMS not enabled - DEBUG: mode check failed */
  }

  /* Step 1: Discover caustic caster chain by tracing seed ray from receiver to light */
  Ray seed_ray;
  seed_ray.self.object = sd->object;
  seed_ray.self.prim = sd->prim;
  seed_ray.self.light_object = ls->object;
  seed_ray.self.light_prim = ls->prim;
  seed_ray.P = sd->P;
  seed_ray.tmin = 0.0f;
  seed_ray.time = sd->time;
  seed_ray.dP = differential_make_compact(sd->dP);
  seed_ray.dD = differential_zero_compact();

  /* Direction and distance depend on light type */
  bool light_fixed_direction = (ls->t == FLT_MAX);
  if (light_fixed_direction) {
    /* Distant or environment light */
    seed_ray.D = ls->D;
    seed_ray.tmax = ls->t;
  }
  else {
    /* Point/spot/area light - compute direction and distance */
    seed_ray.D = normalize_len(ls->P - seed_ray.P, &seed_ray.tmax);
  }

  /* Check for area light with zero spread (also fixed direction) */
  if (ls->type == LIGHT_AREA) {
    const ccl_global KernelLight *klight = &kernel_data_fetch(lights, ls->prim);
    if (klight->area.tan_half_spread == 0.0f) {
      /* Area light with zero spread also has fixed direction */
      light_fixed_direction = true;
    }
  }

  /* Storage for discovered intersections and BSDFs */
  Intersection isects[MNEE_MAX_CAUSTIC_CASTERS];
  ccl_private ShaderClosure *compatible_bsdfs[MNEE_MAX_CAUSTIC_CASTERS];
  float etas[MNEE_MAX_CAUSTIC_CASTERS];
  int vertex_count = 0;
  bool hit_receiver = false;

  /* Trace seed ray and discover caustic casters */
  for (int isect_count = 0; isect_count < MNEE_MAX_INTERSECTION_COUNT; isect_count++) {
    Intersection probe_isect ccl_optional_struct_init;
    probe_isect.object = OBJECT_NONE;
    probe_isect.prim = PRIM_NONE;

    /* Intersect scene looking for caustic casters
     * Allow both reflection and transmission to discover both mirror and glass caustics */
    if (!scene_intersect(kg, &seed_ray, PATH_RAY_REFLECT | PATH_RAY_TRANSMIT, &probe_isect)) {
      break;  /* No intersection - seed ray reached light or empty space */
    }

    /* Get object and check if it's a caustic caster */
    const int probe_object = probe_isect.object;
    const int probe_object_flag = kernel_data_fetch(object_flag, probe_object);

    /* Skip receiver self-intersection */
    if (probe_isect.object == sd->object && probe_isect.prim == sd->prim) {
      hit_receiver = true;
      /* Update ray to continue past receiver surface
       * Use temporary ShaderData to avoid corrupting sd_sms */
      ShaderData temp_sd;
      shader_setup_from_ray(kg, &temp_sd, &seed_ray, &probe_isect);
      seed_ray.P = ray_offset(seed_ray.P + seed_ray.D * probe_isect.t, temp_sd.Ng);
      seed_ray.tmin = 0.0f;
      seed_ray.tmax -= probe_isect.t;
      if (seed_ray.tmax < 0.0f) {
        break;
      }
      continue;
    }

    /* Skip non-caustic surfaces - continue tracing to find casters */
    if (!(probe_object_flag & SD_OBJECT_CAUSTICS_CASTER)) {
      /* Update ray to continue past this surface
       * Use temporary ShaderData to avoid corrupting sd_sms */
      ShaderData temp_sd;
      shader_setup_from_ray(kg, &temp_sd, &seed_ray, &probe_isect);
      seed_ray.P = ray_offset(seed_ray.P + seed_ray.D * probe_isect.t, temp_sd.Ng);
      seed_ray.tmin = 0.0f;
      seed_ray.tmax -= probe_isect.t;
      if (seed_ray.tmax < 0.0f) {
        break;
      }
      continue;
    }

    /* Check if we have room for more vertices */
    if (vertex_count >= MNEE_MAX_CAUSTIC_CASTERS) {
      break;  /* Chain too long */
    }

    /* Validate this is a triangle primitive */
    if (!(probe_isect.type & PRIMITIVE_TRIANGLE)) {
      return -2;  /* DEBUG: Non-triangle geometry */
    }

    /* Setup shader data for this intersection */
    shader_setup_from_ray(kg, sd_sms, &seed_ray, &probe_isect);

    /* Check for smooth normals (required for dn_du, dn_dv computation) */
    if (!(sd_sms->shader & SHADER_SMOOTH_NORMAL)) {
      return -2;  /* DEBUG: Flat normals, no derivatives */
    }

    /* Evaluate shader to get BSDF
     * Use combined ray type to evaluate both reflective and refractive BSDFs */
    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW>(
        kg, state, sd_sms, nullptr, PATH_RAY_REFLECT | PATH_RAY_TRANSMIT, true);

    /* Find a compatible specular BSDF at this vertex
     * SMS supports both reflection and refraction (unlike MNEE which is refraction-only) */
    ccl_private ShaderClosure *specular_bsdf = nullptr;
    for (int ci = 0; ci < sd_sms->num_closure; ci++) {
      ccl_private ShaderClosure *sc = &sd_sms->closure[ci];
      if (CLOSURE_IS_BSDF_MICROFACET(sc->type)) {
        specular_bsdf = sc;
        break;
      }
    }

    if (!specular_bsdf) {
      return -2;  /* DEBUG: No microfacet BSDF found at caster */
    }

    /* Store intersection data */
    isects[vertex_count] = probe_isect;
    compatible_bsdfs[vertex_count] = specular_bsdf;

    /* Extract IOR */
    if (CLOSURE_IS_BSDF_MICROFACET(specular_bsdf->type)) {
      ccl_private const MicrofacetBsdf *microfacet_bsdf =
          (ccl_private const MicrofacetBsdf *)specular_bsdf;
      etas[vertex_count] = microfacet_bsdf->ior;
    }
    else {
      etas[vertex_count] = 1.0f;
    }

    vertex_count++;

    /* Continue seed ray from this point */
    seed_ray.P = sd_sms->P;
    seed_ray.tmin = intersection_t_offset(probe_isect.t);
    seed_ray.tmax -= probe_isect.t;

    if (seed_ray.tmax < 0.0f) {
      break;
    }
  }

  /* Validate we found at least one caustic caster */
  if (vertex_count == 0) {
    return -2;  /* DEBUG: No caustic casters found in seed ray trace */
  }

  /* Check bounce limits before attempting to solve */
  const int transmission_bounce = INTEGRATOR_STATE(state, path, transmission_bounce);
  const int diffuse_bounce = INTEGRATOR_STATE(state, path, diffuse_bounce);
  const int total_bounce = INTEGRATOR_STATE(state, path, bounce);

  if ((transmission_bounce + vertex_count - 1) >=
      kernel_data.integrator.max_transmission_bounce)
  {
    return -2;  /* DEBUG: Transmission bounce limit */
  }

  if ((diffuse_bounce + 1) >= kernel_data.integrator.max_diffuse_bounce) {
    return -2;  /* DEBUG: Diffuse bounce limit */
  }

  if ((total_bounce + vertex_count) >= kernel_data.integrator.max_bounce) {
    return -2;  /* DEBUG: Total bounce limit */
  }

  /* Step 2: Stochastic initialization - sample random microfacet normal offsets
   * For glossy surfaces: use random offsets for stochastic manifold exploration
   * For perfect specular: use zero offsets (deterministic, like MNEE) */
  float2 h_offsets_ref[MNEE_MAX_CAUSTIC_CASTERS];
  const bool has_roughness = sms_sample_microfacet_offsets(
      kg, rng_state, vertex_count, compatible_bsdfs, h_offsets_ref);

  /* Step 3: Setup and solve for reference manifold solution */
  ManifoldVertex vertices_ref[MNEE_MAX_CAUSTIC_CASTERS];

  for (int v_idx = 0; v_idx < vertex_count; v_idx++) {
    mnee_setup_manifold_vertex(kg,
                                &vertices_ref[v_idx],
                                compatible_bsdfs[v_idx],
                                etas[v_idx],
                                h_offsets_ref[v_idx],
                                &seed_ray,
                                &isects[v_idx],
                                sd_sms);
  }

  /* Solve using MNEE's full block Newton solver */
  if (!mnee_newton_solver(
          kg, sd, sd_sms, ls, light_fixed_direction, vertex_count, vertices_ref))
  {
    return -3;  /* DEBUG: Manifold solver failed to converge */
  }

  /* Store reference solution for comparison */
  ManifoldVertex solution_ref[MNEE_MAX_CAUSTIC_CASTERS];
  for (int i = 0; i < vertex_count; i++) {
    solution_ref[i] = vertices_ref[i];
  }

  /* Step 4: Bernoulli trials for unbiased probability estimation
   * For glossy surfaces: run trials with new random offsets until matching solution found
   * For perfect specular: skip trials (deterministic, probability = 1.0) */
  float inv_probability = 1.0f;
  int trial_count = 1;  /* Already found reference solution */

  if (has_roughness) {
    for (int trial = 1; trial < SMS_MAX_TRIALS; trial++) {
    /* Sample new random microfacet offsets for this trial */
    float2 h_offsets_trial[MNEE_MAX_CAUSTIC_CASTERS];
    sms_sample_microfacet_offsets(
        kg, rng_state, vertex_count, compatible_bsdfs, h_offsets_trial);

    /* Setup vertices with new offsets */
    ManifoldVertex vertices_trial[MNEE_MAX_CAUSTIC_CASTERS];
    for (int v_idx = 0; v_idx < vertex_count; v_idx++) {
      mnee_setup_manifold_vertex(kg,
                                  &vertices_trial[v_idx],
                                  compatible_bsdfs[v_idx],
                                  etas[v_idx],
                                  h_offsets_trial[v_idx],
                                  &seed_ray,
                                  &isects[v_idx],
                                  sd_sms);
    }

    /* Solve manifold with new initialization */
    if (!mnee_newton_solver(
            kg, sd, sd_sms, ls, light_fixed_direction, vertex_count, vertices_trial))
    {
      /* Solver failed - count as non-matching trial */
      trial_count++;
      inv_probability += 1.0f;
      continue;
    }

    /* Check if trial solution matches reference solution */
    if (sms_solutions_match(vertex_count, vertices_trial, solution_ref)) {
      /* Found matching solution - stop trials */
      break;
    }

    /* Non-matching solution - continue trials */
    trial_count++;
    inv_probability += 1.0f;
    }

    /* If we exhausted all trials without finding match, probability estimate is invalid */
    if (trial_count >= SMS_MAX_TRIALS &&
        !sms_solutions_match(vertex_count, vertices_ref, solution_ref))
    {
      return -5;  /* DEBUG: Probability estimation failed (Bernoulli trials) */
    }
  }
  /* For perfect specular, inv_probability stays 1.0 (deterministic) */

  /* Step 5: Compute path contribution using MNEE's evaluation infrastructure */
  if (!mnee_path_contribution(kg,
                              state,
                              sd,
                              sd_sms,
                              ls,
                              light_fixed_direction,
                              vertex_count,
                              solution_ref,
                              throughput))
  {
    return -4;  /* DEBUG: Path contribution (mnee_path_contribution) failed */
  }

  /* Step 6: Weight contribution by inverse probability for unbiased estimator */
  bsdf_eval_mul(throughput, inv_probability);

  /* Return vertex count to signal success */
  return vertex_count;
}

CCL_NAMESPACE_END
