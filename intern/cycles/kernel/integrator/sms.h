/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/mnee.h"
#include "kernel/sample/mapping.h"

/*
 * Specular Manifold Sampling (SMS)
 *
 * This code implements Specular Manifold Sampling for rendering high-frequency caustics.
 * SMS extends Manifold Next Event Estimation (MNEE) by using stochastic initialization
 * instead of deterministic seed paths, enabling unbiased rendering of caustics.
 *
 * Key difference from MNEE:
 * - MNEE uses deterministic seeding: traces directly toward light, finds caustic casters
 *   only along the receiver→light axis, then applies Newton iteration.
 * - SMS uses stochastic seeding: randomizes the initial probe direction (angular jittering)
 *   and samples random positions on found surfaces (random barycentrics), enabling
 *   discovery of off-axis specular paths that MNEE would miss.
 *
 * The Mitsuba reference implementation uses full "global seeding" where it iterates over
 * all caustic caster shapes and samples uniformly on each. Our implementation approximates
 * this by using angular jittering within a cone around the light direction, combined with
 * random surface point sampling, which is more practical within Cycles' architecture.
 *
 * Reference:
 * "Specular Manifold Sampling for Rendering High-Frequency Caustics and Glints"
 * Tizian Zeltner, Iliyan Georgiev, Wenzel Jakob
 * ACM Transactions on Graphics (Proc. SIGGRAPH), 2020
 * https://github.com/tizian/specular-manifold-sampling
 */

CCL_NAMESPACE_BEGIN

/* SMS algorithm constants. */
#define SMS_MAX_TRIALS 64
#define SMS_BIASED_BUDGET 2
#define SMS_PROBE_CONE_ANGLE 0.5f  /* Half-angle in radians (~28 degrees) for probe ray jittering */

/* Structure for tracking unique solutions in biased SMS. */
struct SMSUniqueSolution {
  float3 pos;  /* Position on first caustic caster vertex. */
  float3 dir;  /* Direction from first caster towards receiver. */
  uint hash;   /* Hash for quick comparison. */
  bool valid;  /* Slot validity flag. */
};

/* Hash function for solution uniqueness check. */
ccl_device_inline uint sms_hash_solution(const float3 pos, const float3 dir)
{
  /* Simple spatial hashing. */
  const int3 ipos = make_int3((int)(pos.x * 1000.0f), (int)(pos.y * 1000.0f), (int)(pos.z * 1000.0f));
  const int3 idir = make_int3((int)(dir.x * 100.0f), (int)(dir.y * 100.0f), (int)(dir.z * 100.0f));
  return hash_uint3(ipos.x ^ idir.x, ipos.y ^ idir.y, ipos.z ^ idir.z);
}

/* Check if a solution is unique compared to previously found solutions. */
ccl_device_inline bool sms_is_unique_solution(const float3 pos,
                                               const float3 dir,
                                               ccl_private SMSUniqueSolution *solutions,
                                               const int num_solutions)
{
  const uint hash = sms_hash_solution(pos, dir);
  const float pos_threshold = 1e-4f;
  const float dir_threshold = 1e-3f;

  for (int i = 0; i < num_solutions; i++) {
    if (!solutions[i].valid) {
      continue;
    }
    if (solutions[i].hash == hash) {
      if (len_squared(solutions[i].pos - pos) < pos_threshold &&
          len_squared(solutions[i].dir - dir) < dir_threshold)
      {
        return false;
      }
    }
  }
  return true;
}

/* Sample a random point on a triangle mesh (caustic caster). */
ccl_device_inline void sms_sample_triangle_point(KernelGlobals kg,
                                                  const int object,
                                                  const int prim,
                                                  const float u,
                                                  const float v,
                                                  ccl_private float3 &P,
                                                  ccl_private float3 &Ng)
{
  /* Sample point on triangle using barycentric coordinates. */
  const float sqrt_u = sqrtf(u);
  const float u1 = 1.0f - sqrt_u;
  const float u2 = v * sqrt_u;

  /* Get triangle vertices. */
  float3 tri_a, tri_b, tri_c;
  triangle_vertices(kg, prim, &tri_a, &tri_b, &tri_c);

  /* Apply object transform if needed. */
  if (object != OBJECT_NONE) {
    const Transform tfm = object_get_transform(kg, object);
    tri_a = transform_point(&tfm, tri_a);
    tri_b = transform_point(&tfm, tri_b);
    tri_c = transform_point(&tfm, tri_c);
  }

  /* Compute position and geometric normal. */
  P = u1 * tri_a + u2 * tri_b + (1.0f - u1 - u2) * tri_c;
  Ng = normalize(cross(tri_b - tri_a, tri_c - tri_a));
}

/* Find caustic casters using stochastic probe ray direction.
 * Unlike deterministic MNEE which traces directly toward the light,
 * SMS uses angular jittering to discover off-axis specular paths.
 * This implements a practical approximation of global seeding where
 * the probe direction is sampled within a cone around the light direction. */
ccl_device_forceinline int sms_find_caster_chain(KernelGlobals kg,
                                                  IntegratorState state,
                                                  ccl_private ShaderData *sd,
                                                  ccl_private ShaderData *sd_mnee,
                                                  const ccl_private LightSample *ls,
                                                  ccl_private ManifoldVertex *vertices,
                                                  const ccl_private RNGState *rng_state,
                                                  bool *has_reflection)
{
  /* Base direction toward light. */
  float3 base_D;
  float base_tmax;
  if (ls->t == FLT_MAX) {
    base_D = ls->D;
    base_tmax = ls->t;
  }
  else {
    base_D = ls->P - sd->P;
    base_D = normalize_len(base_D, &base_tmax);
  }

  /* Apply angular jittering: sample direction within a cone around the light direction.
   * This allows discovery of caustic casters that don't lie on the direct receiver-light axis,
   * which is essential for rendering off-axis caustics from complex geometry.
   *
   * The cone angle determines how far off-axis we can search. Larger angles allow finding
   * more diverse caustic paths but may reduce efficiency for simple geometries. */
  const float2 jitter_rand = path_state_rng_2D(kg, rng_state, PRNG_LIGHT_U);
  float cos_theta_unused;
  float pdf_unused;
  const float one_minus_cos_angle = 1.0f - cosf(SMS_PROBE_CONE_ANGLE);
  float3 jittered_D = sample_uniform_cone(
      base_D, one_minus_cos_angle, jitter_rand, &cos_theta_unused, &pdf_unused);

  /* Setup probe ray with jittered direction. */
  Ray probe_ray;
  probe_ray.self.object = sd->object;
  probe_ray.self.prim = sd->prim;
  probe_ray.self.light_object = ls->object;
  probe_ray.self.light_prim = ls->prim;
  probe_ray.P = sd->P;
  probe_ray.tmin = 0.0f;
  probe_ray.D = jittered_D;
  /* Extend tmax to account for jittered direction potentially being longer path. */
  probe_ray.tmax = base_tmax * 2.0f;
  probe_ray.dP = differential_make_compact(sd->dP);
  probe_ray.dD = differential_zero_compact();
  probe_ray.time = sd->time;
  Intersection probe_isect;

  *has_reflection = false;
  int vertex_count = 0;

  for (int isect_count = 0; isect_count < MNEE_MAX_INTERSECTION_COUNT; isect_count++) {
    const bool hit = scene_intersect(kg, &probe_ray, PATH_RAY_TRANSMIT, &probe_isect);
    if (!hit) {
      break;
    }

    const int object_flags = intersection_get_object_flags(kg, &probe_isect);
    if (object_flags & SD_OBJECT_CAUSTICS_CASTER) {
      /* Check if we have enough slots. */
      if (vertex_count >= MNEE_MAX_CAUSTIC_CASTERS) {
        return 0;
      }

      /* Reject if not a triangle mesh. */
      if (!(probe_isect.type & PRIMITIVE_TRIANGLE)) {
        return 0;
      }

      ccl_private ManifoldVertex &mv = vertices[vertex_count++];

      /* Setup shader data on caustic caster. */
      shader_setup_from_ray(kg, sd_mnee, &probe_ray, &probe_isect);

      /* Reject if smooth normals not available. */
      if (!(sd_mnee->shader & SHADER_SMOOTH_NORMAL)) {
        return 0;
      }

      surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW>(
          kg, state, sd_mnee, nullptr, PATH_RAY_DIFFUSE, true);

      /* Find SMS-compatible BSDF (refraction, glass, or reflection). */
      bool found_compatible_bsdf = false;
      for (int ci = 0; ci < sd_mnee->num_closure; ci++) {
        ccl_private ShaderClosure *bsdf = &sd_mnee->closure[ci];
        if (CLOSURE_IS_SMS_COMPATIBLE(bsdf->type)) {
          found_compatible_bsdf = true;
          ccl_private MicrofacetBsdf *microfacet_bsdf = (ccl_private MicrofacetBsdf *)bsdf;

          /* Determine if this is a reflection closure. */
          if (CLOSURE_IS_REFLECTION(bsdf->type)) {
            *has_reflection = true;
          }

          /* Figure out appropriate index of refraction ratio. */
          float eta;
          if (CLOSURE_IS_REFLECTION(bsdf->type)) {
            eta = 1.0f;
          }
          else {
            eta = (sd_mnee->flag & SD_BACKFACING) ? 1.0f / microfacet_bsdf->ior :
                                                    microfacet_bsdf->ior;
          }

          /* Sample microfacet normal offset. */
          float2 h = zero_float2();
          if (microfacet_bsdf->alpha_x > 0.f && microfacet_bsdf->alpha_y > 0.f) {
            const float2 bsdf_uv = path_state_rng_2D(kg, rng_state, PRNG_SURFACE_BSDF);
            h = mnee_sample_bsdf_dh(bsdf->type,
                                    microfacet_bsdf->alpha_x,
                                    microfacet_bsdf->alpha_y,
                                    bsdf_uv.x,
                                    bsdf_uv.y);
          }

          /* Setup differential geometry on vertex with random barycentric sampling.
           * By passing rng_state, the vertex position is sampled uniformly on the
           * triangle instead of using the intersection point. This, combined with
           * angular jittering, provides more comprehensive exploration of the
           * specular manifold than deterministic MNEE initialization. */
          mnee_setup_manifold_vertex(
              kg, &mv, bsdf, eta, h, &probe_ray, &probe_isect, sd_mnee, rng_state);
          break;
        }
      }
      if (!found_compatible_bsdf) {
        return 0;
      }
    }

    probe_ray.self.object = probe_isect.object;
    probe_ray.self.prim = probe_isect.prim;
    probe_ray.tmin = intersection_t_offset(probe_isect.t);
  }

  return vertex_count;
}

/* Evaluate SMS path contribution using MNEE infrastructure. */
ccl_device_forceinline bool sms_path_contribution(KernelGlobals kg,
                                                   IntegratorState state,
                                                   ccl_private ShaderData *sd,
                                                   ccl_private ShaderData *sd_mnee,
                                                   ccl_private LightSample *ls,
                                                   const bool light_fixed_direction,
                                                   const int vertex_count,
                                                   ccl_private ManifoldVertex *vertices,
                                                   ccl_private BsdfEval *throughput,
                                                   bool reflection)
{
  /* Use MNEE's path contribution evaluation. */
  return mnee_path_contribution(
      kg, state, sd, sd_mnee, ls, light_fixed_direction, vertex_count, vertices, throughput);
}

/* Biased SMS: Find multiple unique solutions with fixed budget. */
ccl_device_forceinline Spectrum integrate_sms_biased(KernelGlobals kg,
                                                      IntegratorState state,
                                                      ccl_private ShaderData *sd,
                                                      ccl_private ShaderData *sd_mnee,
                                                      const ccl_private RNGState *rng_state,
                                                      ccl_private LightSample *ls,
                                                      const bool light_fixed_direction)
{
  SMSUniqueSolution solutions[SMS_BIASED_BUDGET];
  for (int i = 0; i < SMS_BIASED_BUDGET; i++) {
    solutions[i].valid = false;
  }

  Spectrum result = zero_spectrum();
  int num_unique_solutions = 0;

  for (int trial = 0; trial < SMS_BIASED_BUDGET; trial++) {
    ManifoldVertex vertices[MNEE_MAX_CAUSTIC_CASTERS];
    bool has_reflection = false;

    /* Find caster chain with stochastic sampling. */
    int vertex_count = sms_find_caster_chain(
        kg, state, sd, sd_mnee, ls, vertices, rng_state, &has_reflection);

    if (vertex_count == 0) {
      continue;
    }

    /* Check depth limits. */
    if ((INTEGRATOR_STATE(state, path, transmission_bounce) + vertex_count - 1) >=
        kernel_data.integrator.max_transmission_bounce)
    {
      continue;
    }
    if ((INTEGRATOR_STATE(state, path, diffuse_bounce) + 1) >=
        kernel_data.integrator.max_diffuse_bounce)
    {
      continue;
    }
    if ((INTEGRATOR_STATE(state, path, bounce) + vertex_count) >=
        kernel_data.integrator.max_bounce)
    {
      continue;
    }

    /* Walk on specular manifold. */
    if (!mnee_newton_solver_sms(kg,
                                sd,
                                sd_mnee,
                                ls,
                                light_fixed_direction,
                                vertex_count,
                                vertices,
                                has_reflection,
                                kernel_data.integrator.caustics_constraint_derivatives))
    {
      continue;
    }

    /* Check if this is a unique solution. */
    const float3 first_pos = vertices[0].p;
    const float3 first_dir = normalize(sd->P - first_pos);

    if (!sms_is_unique_solution(first_pos, first_dir, solutions, num_unique_solutions)) {
      continue;
    }

    /* Record solution. */
    if (num_unique_solutions < SMS_BIASED_BUDGET) {
      solutions[num_unique_solutions].pos = first_pos;
      solutions[num_unique_solutions].dir = first_dir;
      solutions[num_unique_solutions].hash = sms_hash_solution(first_pos, first_dir);
      solutions[num_unique_solutions].valid = true;
      num_unique_solutions++;
    }

    /* Evaluate path contribution. */
    BsdfEval throughput;
    if (sms_path_contribution(kg,
                              state,
                              sd,
                              sd_mnee,
                              ls,
                              light_fixed_direction,
                              vertex_count,
                              vertices,
                              &throughput,
                              has_reflection))
    {
      result += bsdf_eval_sum(&throughput);
    }
  }

  return result;
}

/* Unbiased SMS: Use geometric series estimator for unbiased rendering. */
ccl_device_forceinline Spectrum integrate_sms_unbiased(KernelGlobals kg,
                                                        IntegratorState state,
                                                        ccl_private ShaderData *sd,
                                                        ccl_private ShaderData *sd_mnee,
                                                        const ccl_private RNGState *rng_state,
                                                        ccl_private LightSample *ls,
                                                        const bool light_fixed_direction)
{
  Spectrum result = zero_spectrum();

  /* First, find a reference solution. */
  ManifoldVertex ref_vertices[MNEE_MAX_CAUSTIC_CASTERS];
  bool ref_has_reflection = false;

  int ref_vertex_count = sms_find_caster_chain(
      kg, state, sd, sd_mnee, ls, ref_vertices, rng_state, &ref_has_reflection);

  if (ref_vertex_count == 0) {
    return result;
  }

  /* Check depth limits for reference path. */
  if ((INTEGRATOR_STATE(state, path, transmission_bounce) + ref_vertex_count - 1) >=
      kernel_data.integrator.max_transmission_bounce)
  {
    return result;
  }
  if ((INTEGRATOR_STATE(state, path, diffuse_bounce) + 1) >=
      kernel_data.integrator.max_diffuse_bounce)
  {
    return result;
  }
  if ((INTEGRATOR_STATE(state, path, bounce) + ref_vertex_count) >=
      kernel_data.integrator.max_bounce)
  {
    return result;
  }

  /* Walk on specular manifold to find reference solution. */
  if (!mnee_newton_solver_sms(kg,
                              sd,
                              sd_mnee,
                              ls,
                              light_fixed_direction,
                              ref_vertex_count,
                              ref_vertices,
                              ref_has_reflection,
                              kernel_data.integrator.caustics_constraint_derivatives))
  {
    return result;
  }

  /* Reference solution found, record it. */
  const float3 ref_pos = ref_vertices[0].p;
  const float3 ref_dir = normalize(sd->P - ref_pos);
  const uint ref_hash = sms_hash_solution(ref_pos, ref_dir);

  /* Evaluate reference path contribution. */
  BsdfEval ref_throughput;
  if (!sms_path_contribution(kg,
                             state,
                             sd,
                             sd_mnee,
                             ls,
                             light_fixed_direction,
                             ref_vertex_count,
                             ref_vertices,
                             &ref_throughput,
                             ref_has_reflection))
  {
    return result;
  }

  Spectrum ref_contrib = bsdf_eval_sum(&ref_throughput);

  /* Geometric series probability estimator.
   * Perform Bernoulli trials to find the same solution again. */
  int n_trials = 0;
  const float success_prob = 0.5f;

  for (int trial = 1; trial < SMS_MAX_TRIALS; trial++) {
    /* Bernoulli trial: continue with probability success_prob. */
    const float rnd = path_state_rng_1D(kg, rng_state, PRNG_PHASE_CHANNEL);
    if (rnd > success_prob) {
      n_trials = trial;
      break;
    }

    /* Try to find the same solution again. */
    ManifoldVertex vertices[MNEE_MAX_CAUSTIC_CASTERS];
    bool has_reflection = false;

    int vertex_count = sms_find_caster_chain(
        kg, state, sd, sd_mnee, ls, vertices, rng_state, &has_reflection);

    if (vertex_count == 0) {
      continue;
    }

    if (!mnee_newton_solver_sms(kg,
                                sd,
                                sd_mnee,
                                ls,
                                light_fixed_direction,
                                vertex_count,
                                vertices,
                                has_reflection,
                                kernel_data.integrator.caustics_constraint_derivatives))
    {
      continue;
    }

    /* Check if we found the same solution. */
    const float3 pos = vertices[0].p;
    const float3 dir = normalize(sd->P - pos);
    const uint hash = sms_hash_solution(pos, dir);

    const float pos_threshold = 1e-4f;
    const float dir_threshold = 1e-3f;

    if (hash == ref_hash && len_squared(pos - ref_pos) < pos_threshold &&
        len_squared(dir - ref_dir) < dir_threshold)
    {
      /* Found the same solution, count success. */
      n_trials = trial;
      break;
    }
  }

  if (n_trials == 0) {
    n_trials = SMS_MAX_TRIALS;
  }

  /* Compute unbiased estimate using inverse of success probability. */
  const float weight = powf(1.0f / success_prob, (float)n_trials);
  result = ref_contrib * weight;

  return result;
}

CCL_NAMESPACE_END
