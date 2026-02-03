/* SPDX-FileCopyrightText: 2024 Blender Authors
 * Based on Specular Manifold Sampling by Zeltner et al. [2020]
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/mnee.h"

/*
 * Specular Manifold Sampling (SMS)
 *
 * This code extends MNEE to support Specular Manifold Sampling, which generalizes
 * the technique to handle both reflection and refraction, introduces stochastic
 * initialization and probability estimation to handle complex caustics with multiple
 * solutions. This implementation is based on the algorithm described in paper [1].
 *
 * [1] Specular Manifold Sampling for Rendering High-Frequency Caustics and Glints
 * Tizian Zeltner, Iliyan Georgiev, and Wenzel Jakob. 2020. ACM Trans. Graph. 39, 4,
 * Article 149 (July 2020), 15 pages.
 * https://rgl.epfl.ch/publications/Zeltner2020Specular
 */

// NOLINTBEGIN
#define SMS_MAX_TRIALS 64   /* Maximum iterations for unbiased SMS probability estimation. */
#define SMS_BIASED_BUDGET 2 /* Budget for unique solutions in biased SMS. */
#define SMS_EPSILON 1e-5f   /* Small epsilon for float comparisons in solution checking. */
// NOLINTEND

CCL_NAMESPACE_BEGIN

/* Structure to store unique solutions found during biased SMS integration.
 * Uniqueness is determined based on the state of the first vertex in the chain. */
struct SMSUniqueSolution {
  float3 pos; /* Position on the first caustic caster vertex. */
  float3 dir; /* Direction from the first caster vertex towards the receiver. */
  uint hash;  /* Hash of pos and dir for quick comparison. */
  bool valid; /* Flag indicating if this slot contains a valid solution. */
};

/* Helper function to generate a simple hash for float3. */
ccl_device_inline uint hash_float3(const float3 v)
{
  /* Simple XOR hash of float bits. */
  uint hx = __float_as_uint(v.x);
  uint hy = __float_as_uint(v.y);
  uint hz = __float_as_uint(v.z);
  return hx ^ (hy << 11 | hy >> 21) ^ (hz << 22 | hz >> 10);
}

/* Helper function to check if a solution (defined by the first vertex's state)
 * is already present in the unique solutions list. Uses a hash for quick check
 * followed by distance check for collision resolution. */
ccl_device_inline bool sms_is_duplicate_solution(const float3 pos,
                                                 const float3 dir,
                                                 const uint hash,
                                                 const ccl_private SMSUniqueSolution *solutions,
                                                 const int count)
{
  for (int i = 0; i < count; ++i) {
    if (solutions[i].valid && solutions[i].hash == hash) {
      /* Hash collision check: verify actual values using squared distance. */
      if (len_squared(solutions[i].pos - pos) < SMS_EPSILON * SMS_EPSILON &&
          len_squared(solutions[i].dir - dir) < SMS_EPSILON * SMS_EPSILON)
      {
        return true;
      }
    }
  }
  return false;
}

/* Helper function to find a chain of potential SMS caster intersections along a ray.
 * Iterates along the probe ray, identifies potential caster surfaces, checks their
 * properties (triangle, not receiver, smooth normals, compatible BSDF), and stores
 * relevant data (intersection, BSDF closure, eta). Stops if max intersections or
 * max caster vertices are reached. Also checks path length limits against integrator settings.
 * Returns the number of vertices found in the chain, or 0 if no valid chain is found
 * or path limits are exceeded. */
ccl_device_forceinline int sms_find_caster_chain(
    KernelGlobals kg,
    IntegratorState state,
    const ccl_private ShaderData *sd,                 /* Receiver surface point. */
    const ccl_private LightSample *ls,                /* Light sample information. */
    ccl_private ShaderData *sd_scratch,               /* Scratch ShaderData for evaluations. */
    ccl_private Ray *probe_ray_out,                   /* Output: Initial probe ray properties. */
    ccl_private Intersection *caster_isects_out,      /* Output: Array for caster intersections. */
    ccl_private ShaderClosure **compatible_bsdfs_out, /* Output: Array for compatible BSDFs. */
    ccl_private float *compatible_etas_out)           /* Output: Array for compatible etas. */
{
  /* Setup probe ray from receiver towards light sample. */
  Ray probe_ray;
  probe_ray.self.object = sd->object;
  probe_ray.self.prim = sd->prim;
  probe_ray.self.light_object = ls->object;
  probe_ray.self.light_prim = ls->prim;
  probe_ray.P = sd->P;
  probe_ray.tmin = 0.0f;
  float light_dist = ls->t;
  if (ls->t == FLT_MAX) {
    /* Distant / env light. */
    probe_ray.D = ls->D;
    probe_ray.tmax = ls->t;
  }
  else {
    /* Other lights, avoid self-intersection. */
    probe_ray.D = normalize_len(ls->P - probe_ray.P, &light_dist);
    probe_ray.tmax = light_dist;
  }
  probe_ray.dP = differential_make_compact(sd->dP);
  probe_ray.dD = differential_zero_compact();
  probe_ray.time = sd->time;
  *probe_ray_out = probe_ray; /* Store initial ray state for later use in vertex setup. */

  Intersection probe_isect;
  int vertex_count = 0;

  /* Iterate along the probe ray to find potential caster intersections. */
  for (int isect_count = 0; isect_count < MNEE_MAX_INTERSECTION_COUNT; isect_count++) {
    const bool hit = scene_intersect(kg, &probe_ray, PATH_RAY_TRANSMIT, &probe_isect);
    if (!hit) {
      break; /* Reached light or end of ray. */
    }

    const bool hit_receiver = (probe_isect.prim == sd->prim && probe_isect.object == sd->object);
    const int object_flags = intersection_get_object_flags(kg, &probe_isect);

    /* Check if the hit object is marked as a potential caustic caster. */
    if (object_flags & SD_OBJECT_CAUSTICS_CASTER) {
      /* Ensure it's not the receiver itself. */
      if (!hit_receiver) {

        /* Check if the maximum number of caster vertices is reached. */
        if (vertex_count >= MNEE_MAX_CAUSTIC_CASTERS) {
          /* Stop searching if limit reached, proceed with current chain. */
          break;
        }

        /* Reject caster if it is not a triangles mesh. */
        if (!(probe_isect.type & PRIMITIVE_TRIANGLE)) {
          return 0;
        }

        /* Check surface properties: Requires smooth normals for the solver.
         * Use sd_scratch (sd_mnee) for shader evaluation like MNEE does, so BSDF
         * pointers remain valid (pointing into sd_scratch->closure[]). */
        shader_setup_from_ray(kg, sd_scratch, &probe_ray, &probe_isect);

        /* The MNEE solver requires smooth normals to compute derivatives (dn_du, dn_dv).
         * Flat shaded surfaces lack these derivatives, preventing the solver from working
         * correctly. */
        if (sd_scratch->shader & SHADER_SMOOTH_NORMAL) {

          /* Evaluate the surface shader to find compatible BSDF closures. */
          surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW>(
              kg, state, sd_scratch, nullptr, PATH_RAY_DIFFUSE, true);

          ccl_private ShaderClosure *found_bsdf = nullptr;
          float found_eta = 1.0f; /* Default eta for reflection. */

          /* Iterate through closures to find the first compatible one. */
          for (int ci = 0; ci < sd_scratch->num_closure; ++ci) {
            ccl_private ShaderClosure *sc = &sd_scratch->closure[ci];

            // Reflection is not supported for now.
            // if (CLOSURE_IS_SMS_COMPATIBLE(sc->type)) {
            //   found_bsdf = sc;
            //   ccl_private MicrofacetBsdf *microfacet_bsdf = (ccl_private MicrofacetBsdf *)
            //       found_bsdf;

            //   /* Calculate eta for refraction/glass based on facing direction. */
            //   if (!CLOSURE_IS_REFLECTION(found_bsdf->type)) {
            //     found_eta = (sd_scratch.flag & SD_BACKFACING) ? 1.0f / microfacet_bsdf->ior
            //     :
            //                                                          microfacet_bsdf->ior;
            //   }

            //   break; /* Use the first compatible closure found. */
            // }
            if (CLOSURE_IS_REFRACTION(sc->type) || CLOSURE_IS_GLASS(sc->type)) {
              /* Note that Glass closures are treated as refractive further below. */
              found_bsdf = sc;
              ccl_private MicrofacetBsdf *microfacet_bsdf = (ccl_private MicrofacetBsdf *)
                  found_bsdf;

              /* Calculate eta for refraction/glass based on facing direction. */
              found_eta = (sd_scratch->flag & SD_BACKFACING) ? 1.0f / microfacet_bsdf->ior :
                                                               microfacet_bsdf->ior;

              break; /* Use the first compatible closure found. */
            }
          }

          /* If a compatible BSDF was found, store the intersection data. */
          if (found_bsdf) {
            caster_isects_out[vertex_count] = probe_isect;
            compatible_bsdfs_out[vertex_count] = found_bsdf;
            compatible_etas_out[vertex_count] = found_eta;
            vertex_count++; /* Increment the number of found caster vertices. */
            /* Continue searching for subsequent vertices in the chain. */
          }
        }
        /* Ignore intersection if not smooth or no compatible BSDF found. */
      }
    }

    /* Advance the probe ray beyond the current intersection. */
    probe_ray.self.object = probe_isect.object;
    probe_ray.self.prim = probe_isect.prim;
    probe_ray.tmin = intersection_t_offset(probe_isect.t);
    /* Stop if the ray origin goes beyond the light source distance. */
    if (probe_ray.tmin >= probe_ray.tmax) {
      break;
    }
  }

  /* Return 0 if no valid caster vertices were found. */
  if (vertex_count == 0) {
    return 0;
  }

  /* Check path length limits based on the found number of specular vertices. */
  if ((INTEGRATOR_STATE(state, path, transmission_bounce) + vertex_count - 1) >=
          kernel_data.integrator.max_transmission_bounce ||
      (INTEGRATOR_STATE(state, path, diffuse_bounce) + 1) >=
          kernel_data.integrator.max_diffuse_bounce ||
      (INTEGRATOR_STATE(state, path, bounce) + vertex_count) >= kernel_data.integrator.max_bounce)
  {
    return 0; /* Path limits exceeded. */
  }

  /* Return the number of vertices in the valid caster chain. */
  return vertex_count;
}

// NOLINTBEGIN
// Algorithm 2: Unbiased specular manifold sampling (Multi-Scatter Extension)
// Input: shading point x₁, emitter position xₙ (density p(xₙ))
// Output: unbiased estimate of radiance from xₙ to x₁ via a specular chain x₂...x_{k+1}
// ```
// 1  (I₂..I_{k+1}) ← find potential caster chain intersections I from x₁ towards xₙ
// 2  (h₂..h_{k+1}) ← sample microfacet offsets (initial guess for reference path)
// 3  (x₂*..x_{k+1}*) ← manifold_walk(x₁, (I₂..I_{k+1}), xₙ, (h₂..h_{k+1}))
// 4  if no solution found then return 0
// 5  inv_p ← 1                          ⟵ ⟨1 / p_k⟩ (geometric-series estimator)
// 6  while true do                      ⟵ Bernoulli trials
// 7      (h₂'..h_{k+1}') ← sample microfacet offsets (new trial)
// 8      (x₂'..x_{k+1}') ← manifold_walk(x₁, (I₂..I_{k+1}), xₙ, (h₂'..h_{k+1}'))
// 9      if converged and ‖(x₂'..x_{k+1}') – (x₂*..x_{k+1}*)‖ < ε then break
// 10     inv_p ← inv_p + 1
// 11 contribution ← f_s(x₂*..x_{k+1}*) · G(x₁ ↔ x₂*..x_{k+1}* ↔ xₙ) · L_e(xₙ) / p(xₙ)
// 12 return contribution · inv_p
// ```
// NOLINTEND
/* Unbiased SMS integration, extended for multi-scatter paths. Follows Algorithm 2 from the paper,
 * adapted for chains of vertices. Finds one reference solution path and uses Bernoulli trials
 * to estimate its probability, then computes the contribution. */
ccl_device_forceinline Spectrum
integrate_sms_unbiased(KernelGlobals kg,
                       IntegratorState state,
                       ccl_private ShaderData *sd,      /* Receiver surface point. */
                       ccl_private ShaderData *sd_mnee, /* Scratch ShaderData for evaluations. */
                       const ccl_private RNGState *rng_state, /* RNG state for sampling. */
                       ccl_private LightSample *ls,           /* Light sample information. */
                       ccl_private BsdfEval *out_bsdf_eval)   /* Output BSDF evaluation. */
{
  bsdf_eval_init(out_bsdf_eval, zero_spectrum());

  /* 1. Find the caster chain and check path limits. */
  Ray probe_ray;
  Intersection caster_isects[MNEE_MAX_CAUSTIC_CASTERS];
  ccl_private ShaderClosure *compatible_bsdfs[MNEE_MAX_CAUSTIC_CASTERS];
  float compatible_etas[MNEE_MAX_CAUSTIC_CASTERS];

  int vertex_count = sms_find_caster_chain(
      kg, state, sd, ls, sd_mnee, &probe_ray, caster_isects, compatible_bsdfs, compatible_etas);

  if (vertex_count == 0) {
    return zero_spectrum(); /* No valid chain found or limits exceeded. */
  }

  /* Determine if the light source provides a fixed direction (e.g., distant light). */
  bool light_fixed_direction = (ls->t == FLT_MAX);
  if (!light_fixed_direction && ls->type == LIGHT_AREA) {
    const ccl_global KernelLight *klight = &kernel_data_fetch(lights, ls->prim);
    if (klight->area.tan_half_spread == 0.0f) {
      light_fixed_direction = true; /* Area light with zero spread acts like distant light. */
    }
  }

  /* 2. Sample the initial reference path (multi-scatter equivalent of x₂*..x_{k+1}*). */
  ManifoldVertex vertices_ref[MNEE_MAX_CAUSTIC_CASTERS];
  float2 h_offsets_ref[MNEE_MAX_CAUSTIC_CASTERS]; /* Microfacet offsets (hx, hy) for the reference
                                                     path. */

  const int caustics_constraint_derivatives =
      kernel_data.integrator.caustics_constraint_derivatives;

  /* Initialize reference vertices with one set of sampled offsets. */
  for (int v_idx = 0; v_idx < vertex_count; ++v_idx) {
    h_offsets_ref[v_idx] = zero_float2();
    ccl_private MicrofacetBsdf *microfacet_bsdf = (ccl_private MicrofacetBsdf *)
        compatible_bsdfs[v_idx];

    /* Sample offset only if BSDF is rough. */
    if (microfacet_bsdf->alpha_x > 0.0f && microfacet_bsdf->alpha_y > 0.0f) {
      const float2 bsdf_uv = path_state_rng_2D(kg, rng_state, PRNG_SURFACE_BSDF);
      h_offsets_ref[v_idx] = mnee_sample_bsdf_dh(compatible_bsdfs[v_idx]->type,
                                                 microfacet_bsdf->alpha_x,
                                                 microfacet_bsdf->alpha_y,
                                                 bsdf_uv.x,
                                                 bsdf_uv.y);
    }

    /* Setup the manifold vertex. */
    mnee_setup_manifold_vertex(kg,
                               &vertices_ref[v_idx],
                               compatible_bsdfs[v_idx],
                               compatible_etas[v_idx],
                               h_offsets_ref[v_idx],
                               &probe_ray,            /* Original probe ray context. */
                               &caster_isects[v_idx], /* Intersection data for this vertex. */
                               sd_mnee,               /* Scratch ShaderData. */
                               rng_state);            /* Sample random barycentric coordinates. */
  }

  /* Run the Newton solver to find the reference solution path.
   * Use reflection = true if implemented. */
  bool converged_ref = mnee_newton_solver(kg,
                                          sd,
                                          sd_mnee,
                                          ls,
                                          light_fixed_direction,
                                          vertex_count,
                                          vertices_ref,
                                          false, /* Use sms_flag = false. */
                                          caustics_constraint_derivatives);

  if (!converged_ref) {
    /* Initial manifold walk failed to find a reference solution. */
    return zero_spectrum();
  }

  /* Store the reference solution vertex positions for comparison during Bernoulli trials. */
  float3 solution_p_ref[MNEE_MAX_CAUSTIC_CASTERS];
  for (int v_idx = 0; v_idx < vertex_count; ++v_idx) {
    solution_p_ref[v_idx] = vertices_ref[v_idx].p;
  }

  /* 3. Estimate inverse probability using Bernoulli trials (Geometric series estimator). */
  float inv_prob_estimate = 1.0f; /* Initialize estimate for 1/p_k. */
  int iterations = 0;             /* Iteration counter for trial limit. */

  while (true) {
    iterations++;
    /* Check if the maximum number of trials is exceeded. */
    if (iterations > SMS_MAX_TRIALS) {
      inv_prob_estimate = 0.0f; /* Estimation failed if max trials reached without finding ref. */
      break;
    }

    /* Sample a new trial path (multi-scatter equivalent of x₂'..x_{k+1}'). */
    ManifoldVertex vertices_trial[MNEE_MAX_CAUSTIC_CASTERS];
    float2 h_offsets_trial[MNEE_MAX_CAUSTIC_CASTERS]; /* Microfacet offsets for this trial. */

    /* Initialize trial vertices with newly sampled offsets. */
    for (int v_idx = 0; v_idx < vertex_count; ++v_idx) {
      h_offsets_trial[v_idx] = zero_float2();
      ccl_private MicrofacetBsdf *microfacet_bsdf = (ccl_private MicrofacetBsdf *)
          compatible_bsdfs[v_idx];

      /* Sample offset only if BSDF is rough. */
      if (microfacet_bsdf->alpha_x > 0.0f && microfacet_bsdf->alpha_y > 0.0f) {
        const float2 bsdf_uv = path_state_rng_2D(kg, rng_state, PRNG_SURFACE_BSDF);
        h_offsets_trial[v_idx] = mnee_sample_bsdf_dh(compatible_bsdfs[v_idx]->type,
                                                     microfacet_bsdf->alpha_x,
                                                     microfacet_bsdf->alpha_y,
                                                     bsdf_uv.x,
                                                     bsdf_uv.y);
      }

      /* Setup trial vertex. */
      mnee_setup_manifold_vertex(kg,
                                 &vertices_trial[v_idx],
                                 compatible_bsdfs[v_idx],
                                 compatible_etas[v_idx],
                                 h_offsets_trial[v_idx],
                                 &probe_ray,
                                 &caster_isects[v_idx],
                                 sd_mnee,
                                 rng_state);
    }

    /* Run solver for the trial path. */
    bool converged_trial = mnee_newton_solver(kg,
                                              sd,
                                              sd_mnee,
                                              ls,
                                              light_fixed_direction,
                                              vertex_count,
                                              vertices_trial,
                                              false, /* Use reflection = false. */
                                              caustics_constraint_derivatives);

    if (converged_trial) {
      /* Check if the trial solution path matches the reference solution path.
       * Compare positions of all vertices in the chain. */
      bool match = true;
      for (int v_idx = 0; v_idx < vertex_count; ++v_idx) {
        if (len_squared(vertices_trial[v_idx].p - solution_p_ref[v_idx]) >=
            SMS_EPSILON * SMS_EPSILON)
        {
          match = false; /* Mismatch found. */
          break;
        }
      }

      if (match) {
        /* Found the same solution path, terminate Bernoulli trials. */
        break;
      }
    }

    /* Trial failed (didn't converge or didn't match reference path). Increment estimator. */
    inv_prob_estimate += 1.0f;
  }

  /* 4. Calculate final contribution. */
  if (inv_prob_estimate <= 0.0f) {
    /* Probability estimation failed (e.g., exceeded max trials). */
    return zero_spectrum();
  }

  /* Calculate the contribution of the *reference path*. */
  bool contribution_success = mnee_path_contribution(
      kg,
      state,
      sd,
      sd_mnee, /* Pass scratch, evaluated inside. */
      ls,
      light_fixed_direction,
      vertex_count, /* Use actual vertex count. */
      vertices_ref, /* Use the reference vertex chain. */
      out_bsdf_eval,
      false,
      true); /* Use reflection = false. */

  if (!contribution_success) {
    return zero_spectrum(); /* Contribution calculation failed. */
  }

  /* Return final unbiased estimate: Contribution(reference_path) * inv_probability. */
  return bsdf_eval_sum(out_bsdf_eval) * inv_prob_estimate;
}

// NOLINTBEGIN
// Algorithm 3: Biased specular manifold sampling (Multi-Scatter Extension)
// Input: shading point x₁, emitter position xₙ (density p(xₙ)), trial budget M
// Output: biased—but consistent—estimate of radiance from xₙ to x₁ via chain x₂...x_{k+1}
// ```
// 1  (I₂..I_{k+1}) ← find potential caster chain intersections I from x₁ towards xₙ
// 2  S ← ∅                                 ⟵ set of unique solution paths (keyed by x₂* state)
// 3  for i = 1 … M do                      ⟵ fixed trial budget
// 4      (h₂..h_{k+1}) ← sample microfacet offsets (new trial)
// 5      (x₂*..x_{k+1}*) ← manifold_walk(x₁, (I₂..I_{k+1}), xₙ, (h₂..h_{k+1}))
// 6      if converged and x₂* state is unique then
// 7          contribution ← f_s(x₂*..x_{k+1}*) · G(x₁ ↔ x₂*..x_{k+1}* ↔ xₙ) · L_e(xₙ) / p(xₙ)
// 8          S ← S ∪ { (x₂* state, contribution) }
// 9  result ← 0
// 10 for each (state, contribution) ∈ S do
// 11     result += contribution
// 12 return result
// NOLINTEND
/* Biased SMS integration, extended for multi-scatter paths. Follows Algorithm 3 from the paper,
 * adapted for chains of vertices. Runs a fixed number of trials (SMS_BIASED_BUDGET),
 * finds solutions, stores unique ones based on the first vertex state, and sums their
 * contributions. */
ccl_device_forceinline Spectrum
integrate_sms_biased(KernelGlobals kg,
                     IntegratorState state,
                     ccl_private ShaderData *sd,      /* Receiver surface point. */
                     ccl_private ShaderData *sd_mnee, /* Scratch ShaderData for evaluations. */
                     const ccl_private RNGState *rng_state, /* RNG state for sampling. */
                     ccl_private LightSample *ls,           /* Light sample information. */
                     ccl_private BsdfEval *out_bsdf_eval)   /* Output BSDF evaluation. */
{
  bsdf_eval_init(out_bsdf_eval, zero_spectrum());

  /* 1. Find the caster chain and check path limits. */
  Ray probe_ray;
  Intersection caster_isects[MNEE_MAX_CAUSTIC_CASTERS];
  ccl_private ShaderClosure *compatible_bsdfs_storage[MNEE_MAX_CAUSTIC_CASTERS];
  ccl_private ShaderClosure **compatible_bsdfs = compatible_bsdfs_storage;
  float compatible_etas[MNEE_MAX_CAUSTIC_CASTERS];

  int vertex_count = sms_find_caster_chain(
      kg, state, sd, ls, sd_mnee, &probe_ray, caster_isects, compatible_bsdfs, compatible_etas);

  if (vertex_count == 0) {
    return zero_spectrum(); /* No valid chain found or limits exceeded. */
  }

  /* 2. Storage for unique solutions (based on first vertex state). */
  SMSUniqueSolution unique_solutions[SMS_BIASED_BUDGET];
  Spectrum solution_throughputs[SMS_BIASED_BUDGET]; /* Store contribution for each unique
  solution.
                                                     */
  int num_unique_solutions = 0;
  for (int i = 0; i < SMS_BIASED_BUDGET; ++i) {
    unique_solutions[i].valid = false; /* Initialize solution slots as invalid. */
  }

  /* Determine if the light source provides a fixed direction. */
  bool light_fixed_direction = (ls->t == FLT_MAX);
  if (!light_fixed_direction && ls->type == LIGHT_AREA) {
    const ccl_global KernelLight *klight = &kernel_data_fetch(lights, ls->prim);
    if (klight->area.tan_half_spread == 0.0f) {
      light_fixed_direction = true;
    }
  }

  const int caustics_constraint_derivatives =
      kernel_data.integrator.caustics_constraint_derivatives;

  /* 3. Run M trials (manifold walks from stochastic microfacet normals). M = SMS_BIASED_BUDGET.
   */
  for (int i = 0; i < SMS_BIASED_BUDGET; ++i) {

    ManifoldVertex vertices_trial[MNEE_MAX_CAUSTIC_CASTERS];

    /* Initialize the ManifoldVertices for this trial using the stored intersections. */
    for (int v_idx = 0; v_idx < vertex_count; ++v_idx) {
      /* Sample microfacet normal offset h_xy for this vertex based on its compatible BSDF. */
      float2 h_offset = zero_float2();
      ccl_private MicrofacetBsdf *microfacet_bsdf = (ccl_private MicrofacetBsdf *)
          compatible_bsdfs[v_idx];
      if (microfacet_bsdf->alpha_x > 0.0f && microfacet_bsdf->alpha_y > 0.0f) {
        const float2 bsdf_uv = path_state_rng_2D(kg, rng_state, PRNG_SURFACE_BSDF);
        h_offset = mnee_sample_bsdf_dh(compatible_bsdfs[v_idx]->type,
                                       microfacet_bsdf->alpha_x,
                                       microfacet_bsdf->alpha_y,
                                       bsdf_uv.x,
                                       bsdf_uv.y);
      }

      /* Setup trial vertex. */
      mnee_setup_manifold_vertex(kg,
                                 &vertices_trial[v_idx],
                                 compatible_bsdfs[v_idx],
                                 compatible_etas[v_idx],
                                 h_offset,
                                 &probe_ray,            /* Original probe ray context. */
                                 &caster_isects[v_idx], /* Intersection data for this vertex. */
                                 sd_mnee,               /* Scratch ShaderData. */
                                 rng_state);
    }

    /* Run the Newton solver on the whole chain. */
    bool converged = mnee_newton_solver(kg,
                                        sd,
                                        sd_mnee,
                                        ls,
                                        light_fixed_direction,
                                        vertex_count,
                                        vertices_trial,
                                        false, /* Use sms_flag = false. */
                                        caustics_constraint_derivatives);

    if (converged) {
      /* 4. Check for duplicates (based on first vertex) and store unique solutions. */
      float3 sol_p = vertices_trial[0].p; /* Position of the first vertex in the solved chain.
                                           */
      /* Direction from the first vertex towards the receiver. */
      float3 sol_dir = safe_normalize(sd->P - sol_p);
      /* Hash based on the first vertex state for quick uniqueness check. */
      uint sol_hash = hash_float3(sol_p) ^ hash_float3(sol_dir);

      /* Check if this first-vertex state is already stored. */
      if (!sms_is_duplicate_solution(
              sol_p, sol_dir, sol_hash, unique_solutions, num_unique_solutions))
      {
        /* Check if there is space left in the unique solution budget. */
        if (num_unique_solutions < SMS_BIASED_BUDGET) {
          /* Calculate contribution of the *entire path*. */
          if (mnee_path_contribution(kg,
                                     state,
                                     sd,
                                     sd_mnee, /* Pass scratch, evaluated inside. */
                                     ls,
                                     light_fixed_direction,
                                     vertex_count,   /* Pass the actual number of vertices. */
                                     vertices_trial, /* Pass the solved vertex chain. */
                                     out_bsdf_eval,
                                     false,
                                     true)) /* Use sms_flag = false. */
          {
            Spectrum f_trial = bsdf_eval_sum(out_bsdf_eval);
            /* Only store if the contribution is non-zero. */
            if (!is_zero(f_trial)) {
              /* Store the unique solution (based on first vertex) and its total contribution.
               */
              unique_solutions[num_unique_solutions].pos = sol_p;
              unique_solutions[num_unique_solutions].dir = sol_dir;
              unique_solutions[num_unique_solutions].hash = sol_hash;
              unique_solutions[num_unique_solutions].valid = true;
              solution_throughputs[num_unique_solutions] = f_trial;
              num_unique_solutions++;
            }
          }
        }
        /* If budget is full, discard new unique solutions. */
      }
    }
  }

  /* 5. Sum contributions of unique solutions found. */
  Spectrum total_throughput = zero_spectrum();
  for (int l = 0; l < num_unique_solutions; ++l) {
    total_throughput += solution_throughputs[l];
  }

  /* The biased estimator is sum(f(x_l)), no division by M or probabilities needed. */
  return total_throughput;
}

CCL_NAMESPACE_END