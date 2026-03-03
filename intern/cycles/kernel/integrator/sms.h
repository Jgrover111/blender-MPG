/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/mnee.h"

/*
 * Specular Manifold Sampling (SMS)
 *
 * This code implements Specular Manifold Sampling for rendering high-frequency caustics,
 * following the architecture of the Mitsuba reference implementation.
 *
 * Key design: Global seeding (matching Mitsuba reference)
 * - Iterates over ALL caustic caster shapes in the scene.
 * - For each shape, samples a uniformly random point on its surface.
 * - Traces from receiver toward that sampled point to initialize the manifold vertex.
 * - Runs Newton iteration (with angle-difference constraints) to find a valid specular path.
 * - Uses Bernoulli trials to estimate the inverse probability of finding each solution.
 * - Accumulates contributions from all shapes.
 *
 * This differs from MNEE's topology-local approach which traces a single deterministic
 * ray from receiver toward light and only finds caustic casters along that fixed axis.
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
#define SMS_UNIQUENESS_THRESHOLD 1e-4f

/* Sample a uniformly random point on a caustic caster shape.
 * Selects a random triangle from the shape's mesh, then samples
 * uniform barycentrics on that triangle.
 *
 * This is the kernel-side equivalent of Mitsuba's shape->sample_position(). */
ccl_device_forceinline void sms_sample_surface_point(KernelGlobals kg,
                                                     const int caster_idx,
                                                     const float rand_tri,
                                                     const float2 rand_bary,
                                                     ccl_private float3 &P,
                                                     ccl_private float3 &Ng,
                                                     ccl_private int &out_object,
                                                     ccl_private int &out_prim)
{
  const int object = kernel_data_fetch(caustic_caster_object_index, caster_idx);
  const int prim_offset = kernel_data_fetch(caustic_caster_prim_offset, caster_idx);
  const int num_prims = kernel_data_fetch(caustic_caster_num_prims, caster_idx);

  /* Select a random triangle weighted by surface area, matching Mitsuba's
   * shape->sample_position() behavior. */
  int prim = prim_offset;
  float total_area = 0.0f;

  for (int i = 0; i < num_prims; i++) {
    const int area_prim = prim_offset + i;
    const packed_uint3 area_vindex = kernel_data_fetch(tri_vindex, area_prim);
    const float3 area_a = kernel_data_fetch(tri_verts, area_vindex.x);
    const float3 area_b = kernel_data_fetch(tri_verts, area_vindex.y);
    const float3 area_c = kernel_data_fetch(tri_verts, area_vindex.z);
    total_area += 0.5f * len(cross(area_b - area_a, area_c - area_a));
  }

  if (total_area > 0.0f) {
    const float target_area = rand_tri * total_area;
    float cumulative_area = 0.0f;

    prim = prim_offset + num_prims - 1;
    for (int i = 0; i < num_prims; i++) {
      const int area_prim = prim_offset + i;
      const packed_uint3 area_vindex = kernel_data_fetch(tri_vindex, area_prim);
      const float3 area_a = kernel_data_fetch(tri_verts, area_vindex.x);
      const float3 area_b = kernel_data_fetch(tri_verts, area_vindex.y);
      const float3 area_c = kernel_data_fetch(tri_verts, area_vindex.z);
      cumulative_area += 0.5f * len(cross(area_b - area_a, area_c - area_a));
      if (target_area <= cumulative_area) {
        prim = area_prim;
        break;
      }
    }
  }
  else {
    /* Fallback for degenerate meshes: uniform by triangle index. */
    const int tri_idx = min((int)(rand_tri * (float)num_prims), num_prims - 1);
    prim = prim_offset + tri_idx;
  }

  /* Sample uniform barycentrics on the selected triangle. */
  const float sqrt_u = sqrtf(rand_bary.x);
  const float u = 1.0f - sqrt_u;
  const float v = rand_bary.y * sqrt_u;
  const float w = 1.0f - u - v;

  /* Get triangle vertices. */
  const packed_uint3 vindex = kernel_data_fetch(tri_vindex, prim);
  const float3 tri_a = kernel_data_fetch(tri_verts, vindex.x);
  const float3 tri_b = kernel_data_fetch(tri_verts, vindex.y);
  const float3 tri_c = kernel_data_fetch(tri_verts, vindex.z);

  /* Compute position and geometric normal in object space. */
  float3 pos = u * tri_a + v * tri_b + w * tri_c;
  float3 ng = normalize(cross(tri_b - tri_a, tri_c - tri_a));

  /* Apply object transform if needed. */
  const int ob_flag = kernel_data_fetch(object_flag, object);
  if (!(ob_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    Transform tfm = object_fetch_transform(kg, object, OBJECT_TRANSFORM);
    pos = transform_point(&tfm, pos);
    ng = normalize(transform_direction(&tfm, ng));
  }

  if (ob_flag & SD_OBJECT_NEGATIVE_SCALE) {
    ng = -ng;
  }

  P = pos;
  Ng = ng;
  out_object = object;
  out_prim = prim;
}

ccl_device_forceinline float sms_bsdf_offset_normal_pdf(const ClosureType type,
                                                        const float alpha_x,
                                                        const float alpha_y,
                                                        const float2 h)
{
  if (alpha_x <= 0.0f || alpha_y <= 0.0f) {
    return 1.0f;
  }

  const float h2 = dot(h, h);
  if (h2 >= 1.0f) {
    return 0.0f;
  }

  const float hz = safe_sqrtf(1.0f - h2);
  if (hz <= 0.0f) {
    return 0.0f;
  }

  const float inv_alpha_x2 = 1.0f / (alpha_x * alpha_x);
  const float inv_alpha_y2 = 1.0f / (alpha_y * alpha_y);
  const float slope2 = (h.x * h.x * inv_alpha_x2 + h.y * h.y * inv_alpha_y2) / (hz * hz);

  float d = 0.0f;
  switch (type) {
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
      d = expf(-slope2) / (M_PI_F * alpha_x * alpha_y * sqr(hz * hz));
      break;
    default:
      d = 1.0f / (M_PI_F * alpha_x * alpha_y * sqr(hz * hz) * sqr(1.0f + slope2));
      break;
  }

  return d * hz;
}

/* Sample a seed path through specular surfaces, matching Mitsuba's sample_seed_path().
 *
 * Samples a random point on the designated caster shape to get an initial direction,
 * then traces through all specular surfaces encountered, building the manifold vertex
 * chain. Does NOT require reaching a specific target - accepts whatever specular chain
 * the ray produces.
 *
 * For Bernoulli trials (seed_vertex_count > 0), verifies that trial paths hit the same
 * sequence of shapes as the reference path, which is required for offset normals to
 * remain valid across trials. */
ccl_device_forceinline bool sms_sample_path(KernelGlobals kg,
                                            IntegratorState state,
                                            ccl_private ShaderData *sd,
                                            ccl_private ShaderData *sd_mnee,
                                            const ccl_private LightSample *ls,
                                            const int caster_idx,
                                            const float2 roughness_offset,
                                            const int branch,
                                            const int num_branches,
                                            const ccl_private RNGState *rng_state,
                                            ccl_private ManifoldVertex *vertices,
                                            ccl_private int *out_vertex_count,
                                            ccl_private bool *out_has_reflection,
                                            ccl_private float *out_offset_pdf,
                                            ccl_private int *vertex_objects,
                                            const int seed_vertex_count,
                                            const ccl_private int *seed_objects)
{
  *out_vertex_count = 0;
  *out_has_reflection = false;
  *out_offset_pdf = 1.0f;

  /* Sample a random point on the designated caustic caster surface.
   * This provides the initial tracing direction, matching Mitsuba's
   * shape->sample_position(). */
  float3 sampled_P, sampled_Ng;
  int sampled_object, sampled_prim;

  /* SMS consumes RNG dimensions as follows:
   * - PRNG_SURFACE_BSDF: 1D for caster triangle selection, and 2D per rough vertex for dh.
   * - PRNG_LIGHT: 2D (x/y) for caster triangle barycentrics.
   *
   * Trials branch through sample index (sample * num_branches + branch), keeping dimensions stable
   * with the rest of the integrator while still producing per-trial decorrelated draws. */
  const float rand_tri = path_branched_rng_1D(
      kg, rng_state, branch, num_branches, PRNG_SURFACE_BSDF);
  const float3 rand_light = path_branched_rng_3D(kg, rng_state, branch, num_branches, PRNG_LIGHT);
  const float2 rand_bary = make_float2(rand_light.x, rand_light.y);

  sms_sample_surface_point(
      kg, caster_idx, rand_tri, rand_bary, sampled_P, sampled_Ng, sampled_object, sampled_prim);

  /* Trace from receiver toward the sampled point on the caster surface.
   * The sampled point provides the initial direction; we accept whatever
   * specular surfaces the ray encounters (matching Mitsuba). */
  float3 wo = normalize(sampled_P - sd->P);

  Ray probe_ray;
  probe_ray.self.object = sd->object;
  probe_ray.self.prim = sd->prim;
  probe_ray.self.light_object = ls->object;
  probe_ray.self.light_prim = ls->prim;
  probe_ray.P = sd->P;
  probe_ray.D = wo;
  probe_ray.tmin = 0.0f;
  probe_ray.tmax = FLT_MAX;
  probe_ray.dP = differential_make_compact(sd->dP);
  probe_ray.dD = differential_zero_compact();
  probe_ray.time = sd->time;

  Intersection probe_isect;
  int vertex_count = 0;

  /* Target bounce count: for the first path (seed), trace until we run out of
   * specular surfaces. For Bernoulli trials, match the seed path's vertex count. */
  const int max_bounces = (seed_vertex_count > 0) ? seed_vertex_count :
                                                     MNEE_MAX_CAUSTIC_CASTERS;

  /* Build seed path by tracing through specular surfaces.
   * Matches Mitsuba's sample_seed_path: trace from receiver, accumulate
   * specular bounces, scatter (reflect/refract) at each vertex, continue. */
  for (int bounce = 0; bounce < max_bounces; bounce++) {
    if (!scene_intersect(kg, &probe_ray, PATH_RAY_TRANSMIT, &probe_isect)) {
      break;
    }

    const int object_flags = intersection_get_object_flags(kg, &probe_isect);
    if (!(object_flags & SD_OBJECT_CAUSTICS_CASTER)) {
      break;
    }

    if (!(probe_isect.type & PRIMITIVE_TRIANGLE)) {
      return false;
    }

    const int hit_object = (probe_isect.object == OBJECT_NONE) ?
                               kernel_data_fetch(prim_object, probe_isect.prim) :
                               probe_isect.object;

    /* For Bernoulli trials, verify shape consistency with reference path.
     * Matching Mitsuba: "only allow paths that intersect the same shapes again"
     * so that the sampled offset normals remain valid. */
    if (seed_vertex_count > 0 && seed_objects != nullptr) {
      if (hit_object != seed_objects[bounce]) {
        return false;
      }
    }

    shader_setup_from_ray(kg, sd_mnee, &probe_ray, &probe_isect);
    if (!(sd_mnee->shader & SHADER_SMOOTH_NORMAL)) {
      return false;
    }

    surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW>(
        kg, state, sd_mnee, nullptr, PATH_RAY_DIFFUSE, true);

    ccl_private ShaderClosure *selected_bsdf = nullptr;
    for (int ci = 0; ci < sd_mnee->num_closure; ci++) {
      ccl_private ShaderClosure *bsdf = &sd_mnee->closure[ci];
      if (!CLOSURE_IS_SMS_COMPATIBLE(bsdf->type)) {
        continue;
      }
      selected_bsdf = bsdf;
      break;
    }

    if (selected_bsdf == nullptr) {
      return false;
    }

    if (vertex_count >= MNEE_MAX_CAUSTIC_CASTERS) {
      return false;
    }

    ccl_private MicrofacetBsdf *microfacet_bsdf = (ccl_private MicrofacetBsdf *)selected_bsdf;
    ccl_private ManifoldVertex &mv = vertices[vertex_count];

    if (CLOSURE_IS_REFLECTION(selected_bsdf->type) || CLOSURE_IS_GLASS(selected_bsdf->type)) {
      *out_has_reflection = true;
    }

    float eta = 1.0f;
    if (!CLOSURE_IS_REFLECTION(selected_bsdf->type)) {
      eta = (sd_mnee->flag & SD_BACKFACING) ? 1.0f / microfacet_bsdf->ior : microfacet_bsdf->ior;
    }

    /* Tie roughness offsets to the actual visited specular surface. */
    const uint rough_seed = hash_uint2((uint)hit_object, (uint)probe_isect.prim);
    const float2 per_vertex_offset = make_float2(
        fractf(roughness_offset.x + hash_uint2_to_float(rough_seed, 0x12a35d91u)),
        fractf(roughness_offset.y + hash_uint2_to_float(rough_seed, 0x7f4a7c15u)));

    float2 h = zero_float2();
    if (microfacet_bsdf->alpha_x > 0.f && microfacet_bsdf->alpha_y > 0.f) {
      h = mnee_sample_bsdf_dh(selected_bsdf->type,
                              microfacet_bsdf->alpha_x,
                              microfacet_bsdf->alpha_y,
                              per_vertex_offset.x,
                              per_vertex_offset.y);
      *out_offset_pdf *= sms_bsdf_offset_normal_pdf(
          selected_bsdf->type, microfacet_bsdf->alpha_x, microfacet_bsdf->alpha_y, h);
    }

    mnee_setup_manifold_vertex(kg, &mv, selected_bsdf, eta, h, &probe_ray, &probe_isect, sd_mnee);

    /* Store hit object for Bernoulli trial shape consistency checking. */
    vertex_objects[vertex_count] = hit_object;
    vertex_count++;

    /* If this is the last bounce we need, don't scatter - just connect to light.
     * Matching Mitsuba: "connect to the light source now by terminating". */
    if (bounce == max_bounces - 1) {
      break;
    }

    /* Build outgoing direction by scattering at this vertex.
     * Matching Mitsuba: reflect if eta==1, refract otherwise. */
    const float3 wi = -probe_ray.D;
    const float3 n_offset = safe_normalize(mv.n + h.x * mv.dp_du + h.y * mv.dp_dv);
    float3 outgoing = zero_float3();
    bool valid_outgoing = false;

    if (CLOSURE_IS_REFLECTION(selected_bsdf->type)) {
      outgoing = sms_ad_reflect(wi, n_offset);
      valid_outgoing = true;
    }
    else if (CLOSURE_IS_GLASS(selected_bsdf->type)) {
      /* Glass can both refract and reflect. Try refraction first;
       * fall back to reflection on total internal reflection. */
      valid_outgoing = sms_ad_refract(wi, n_offset, eta, outgoing);
      if (!valid_outgoing) {
        outgoing = sms_ad_reflect(wi, n_offset);
        valid_outgoing = true;
      }
    }
    else {
      valid_outgoing = sms_ad_refract(wi, n_offset, eta, outgoing);
    }

    if (!valid_outgoing) {
      return false;
    }
    outgoing = safe_normalize(outgoing);

    probe_ray.self.object = probe_isect.object;
    probe_ray.self.prim = probe_isect.prim;
    probe_ray.P = sd_mnee->P;
    probe_ray.D = outgoing;
    probe_ray.tmin = MNEE_MIN_DISTANCE;
    probe_ray.tmax = FLT_MAX;
  }

  if (vertex_count == 0) {
    return false;
  }

  /* For Bernoulli trials, require matching vertex count. */
  if (seed_vertex_count > 0 && vertex_count != seed_vertex_count) {
    return false;
  }

  *out_vertex_count = vertex_count;
  return true;
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
  /* Validate visibility from the solved manifold endpoint to the sampled light.
   * This mirrors direct-light shadow-ray setup conventions while still allowing
   * the intended light geometry hit at the endpoint for finite emitters. */
  Ray probe_ray;
  probe_ray.self.object = vertices[vertex_count - 1].object;
  probe_ray.self.prim = vertices[vertex_count - 1].prim;
  probe_ray.self.light_object = ls->object;
  probe_ray.self.light_prim = ls->prim;
  probe_ray.P = vertices[vertex_count - 1].p;
  probe_ray.tmin = 0.0f;
  if (light_fixed_direction) {
    probe_ray.D = ls->D;
    probe_ray.tmax = ls->t;
  }
  else {
    probe_ray.D = ls->P - probe_ray.P;
    probe_ray.D = safe_normalize_len(probe_ray.D, &probe_ray.tmax);
  }
  probe_ray.dP = differential_make_compact(sd->dP);
  probe_ray.dD = differential_zero_compact();
  probe_ray.time = sd->time;

  Intersection probe_isect;
  if (scene_intersect(kg, &probe_ray, PATH_RAY_TRANSMIT, &probe_isect)) {
    if (light_fixed_direction) {
      return false;
    }

    const int hit_object = (probe_isect.object == OBJECT_NONE) ?
                               kernel_data_fetch(prim_object, probe_isect.prim) :
                               probe_isect.object;
    const bool hit_intended_light = (hit_object == ls->object) &&
                                    (ls->prim == PRIM_NONE || probe_isect.prim == ls->prim) &&
                                    (fabsf(probe_ray.tmax - probe_isect.t) <= MNEE_MIN_DISTANCE);
    if (!hit_intended_light) {
      return false;
    }
  }

  /* Recompute HV constraint derivatives for the transfer matrix / Jacobian.
   * The Newton solver used AD constraints, but mnee_compute_transfer_matrix()
   * expects HV constraint derivatives in the vertices' .a, .b, .c fields. */
  const float3 light_sample = light_fixed_direction ? ls->D : ls->P;
  if (!mnee_compute_hv_constraint_derivatives(
          vertex_count, vertices, sd->P, light_fixed_direction, light_sample, reflection))
  {
    return false;
  }

  /* Use MNEE's path contribution evaluation. */
  return mnee_path_contribution(
      kg, state, sd, sd_mnee, ls, light_fixed_direction, vertex_count, vertices, throughput,
      reflection);
}

/* Check depth limits for a given vertex count. */
ccl_device_forceinline bool sms_check_depth_limits(KernelGlobals kg,
                                                   IntegratorState state,
                                                   const int vertex_count)
{
  if ((INTEGRATOR_STATE(state, path, transmission_bounce) + vertex_count - 1) >=
      kernel_data.integrator.max_transmission_bounce)
  {
    return false;
  }
  if ((INTEGRATOR_STATE(state, path, diffuse_bounce) + 1) >=
      kernel_data.integrator.max_diffuse_bounce)
  {
    return false;
  }
  if ((INTEGRATOR_STATE(state, path, bounce) + vertex_count) >= kernel_data.integrator.max_bounce)
  {
    return false;
  }
  return true;
}

/* SMS: For each caustic caster shape, find one solution then use
 * Bernoulli trials to estimate the inverse probability of finding that solution.
 *
 * Matches Mitsuba's unbiased mode:
 *   for each shape:
 *     sample_path -> newton_solver -> evaluate_contribution
 *     estimate inverse probability via repeated sample_path trials */
ccl_device_forceinline Spectrum integrate_sms(KernelGlobals kg,
                                                       IntegratorState state,
                                                       ccl_private ShaderData *sd,
                                                       ccl_private ShaderData *sd_mnee,
                                                       const ccl_private RNGState *rng_state,
                                                       ccl_private LightSample *ls,
                                                       const bool light_fixed_direction)
{
  const int num_casters = kernel_data.integrator.caustics_num_casters;
  if (num_casters == 0) {
    return zero_spectrum();
  }

  Spectrum result = zero_spectrum();

  /* Iterate over all caustic caster shapes in the scene. */
  for (int caster_idx = 0; caster_idx < num_casters; caster_idx++) {
    const int sms_num_branches = num_casters * SMS_MAX_TRIALS;
    /* Sample one roughness offset for this shape attempt and reuse it for all Bernoulli retries.
     */
    RNGState offset_rng_state = *rng_state;
    path_state_rng_scramble(&offset_rng_state, (int)hash_uint2(caster_idx, 0x6e624eb7));
    const float2 roughness_offset = path_state_rng_2D(kg, &offset_rng_state, PRNG_SURFACE_BSDF);
    /* Sample initial seed path via this caster shape. */
    ManifoldVertex ref_vertices[MNEE_MAX_CAUSTIC_CASTERS];
    int ref_objects[MNEE_MAX_CAUSTIC_CASTERS];
    bool ref_has_reflection = false;
    int ref_vertex_count = 0;
    float ref_offset_pdf = 1.0f;

    if (!sms_sample_path(kg,
                         state,
                         sd,
                         sd_mnee,
                         ls,
                         caster_idx,
                         roughness_offset,
                         caster_idx * SMS_MAX_TRIALS,
                         sms_num_branches,
                         rng_state,
                         ref_vertices,
                         &ref_vertex_count,
                         &ref_has_reflection,
                         &ref_offset_pdf,
                         ref_objects,
                         0,
                         nullptr))
    {
      continue;
    }

    /* Check depth limits. */
    if (!sms_check_depth_limits(kg, state, ref_vertex_count)) {
      continue;
    }

    /* Newton solver to find reference solution. */
    if (!mnee_newton_solver_sms(kg,
                                sd,
                                sd_mnee,
                                ls,
                                light_fixed_direction,
                                ref_vertex_count,
                                ref_vertices,
                                ref_has_reflection))
    {
      continue;
    }

    /* Evaluate reference path contribution.
     * Save ls before calling sms_path_contribution because mnee_path_contribution
     * mutates it via light_sample_update(). We need the original ls for subsequent
     * caster iterations and Bernoulli trials. */
    LightSample ls_backup = *ls;
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
      *ls = ls_backup;
      continue;
    }
    *ls = ls_backup;

    Spectrum ref_contrib = bsdf_eval_sum(&ref_throughput);

    /* Reference direction for uniqueness comparison (matching Mitsuba).
     * We only compare final direction, not intermediate topology. */
    const float3 ref_direction = normalize(ref_vertices[0].p - sd->P);

    /* Estimate inverse probability via Bernoulli trials.
     * Keep sampling paths until we find the same solution again.
     * The number of trials estimates 1/p where p is the probability
     * of finding this particular solution. */
    float inv_prob_estimate = 1.0f;
    int iterations = 1;

    for (int trial = 1; trial < SMS_MAX_TRIALS; trial++) {
      ManifoldVertex trial_vertices[MNEE_MAX_CAUSTIC_CASTERS];
      int trial_objects[MNEE_MAX_CAUSTIC_CASTERS];
      bool trial_has_reflection = false;
      int trial_vertex_count = 0;
      float trial_offset_pdf = 1.0f;

      if (!sms_sample_path(kg,
                           state,
                           sd,
                           sd_mnee,
                           ls,
                           caster_idx,
                           roughness_offset,
                           trial + caster_idx * SMS_MAX_TRIALS,
                           sms_num_branches,
                           rng_state,
                           trial_vertices,
                           &trial_vertex_count,
                           &trial_has_reflection,
                           &trial_offset_pdf,
                           trial_objects,
                           ref_vertex_count,
                           ref_objects) ||
          trial_offset_pdf <= 0.0f)
      {
        inv_prob_estimate += 1.0f;
        iterations++;
        continue;
      }

      /* Run Newton solver on trial path. */
      if (!mnee_newton_solver_sms(kg,
                                  sd,
                                  sd_mnee,
                                  ls,
                                  light_fixed_direction,
                                  trial_vertex_count,
                                  trial_vertices,
                                  trial_has_reflection))
      {
        inv_prob_estimate += 1.0f;
        iterations++;
        continue;
      }

      /* Check if we found the same solution by comparing directions. */
      const float3 trial_direction = normalize(trial_vertices[0].p - sd->P);
      if (fabsf(dot(ref_direction, trial_direction) - 1.0f) < SMS_UNIQUENESS_THRESHOLD) {
        /* Found the same solution - stop. */
        break;
      }

      inv_prob_estimate += 1.0f;
      iterations++;
    }

    /* If we hit max trials without finding the same solution,
     * set contribution to zero to avoid bias. */
    if (iterations >= SMS_MAX_TRIALS) {
      inv_prob_estimate = 0.0f;
    }

    result += ref_contrib * inv_prob_estimate / max(ref_offset_pdf, 1e-20f);
  }

  return result;
}

CCL_NAMESPACE_END
