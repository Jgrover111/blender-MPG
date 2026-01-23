/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/shader_data.h"
#include "kernel/geom/triangle.h"

#include "kernel/light/sample.h"

/*
 * Specular Manifold Sampling (SMS)
 *
 * This code implements Specular Manifold Sampling for rendering caustic light paths through
 * specular surfaces. Unlike MNEE which focuses on shadow caustics (direct lighting through
 * refractive interfaces), SMS handles complete specular chains including glossy caustics and
 * arbitrary S-D-S (Specular-Diffuse-Specular) paths.
 *
 * The algorithm works by:
 * 1. Detecting when a path contains specular vertices that could form a manifold
 * 2. Building a manifold constraint system for the complete specular chain
 * 3. Using a Newton solver to find the manifold solution satisfying Fermat's principle
 * 4. Computing the path contribution with proper importance sampling weights
 *
 * Key differences from MNEE:
 * - MNEE: Receiver → Specular(s) → Light (shadow caustics only, invoked during NEE)
 * - SMS: Any S-D-S pattern in the path (full caustics, invoked during path extension)
 *
 * References:
 * [1] "Specular Manifold Sampling for Rendering High-Frequency Caustics and Glints"
 *     Tizian Zeltner, Iliyan Georgiev, Wenzel Jakob (2020)
 * [2] "Manifold Next Event Estimation"
 *     Johannes Hanika, Marc Droske, Luca Fascione (2015)
 * [3] "Manifold Exploration: A Markov Chain Monte Carlo Technique for Rendering Scenes
 *     with Difficult Specular Transport"
 *     Wenzel Jakob, Steve Marschner (2012)
 */

// NOLINTBEGIN

/* Maximum number of specular vertices in an SMS chain.
 * This limits memory usage and solver complexity. Typical scenes rarely exceed 4-6. */
#define SMS_MAX_VERTICES 8

/* Newton solver parameters */
#define SMS_MAX_ITERATIONS 64
#define SMS_SOLVER_THRESHOLD 0.0001f
#define SMS_MINIMUM_STEP_SIZE 0.00001f

/* Projection and validation thresholds */
#define SMS_MIN_PROGRESS_DISTANCE 0.0001f
#define SMS_MIN_DETERMINANT 0.0001f

// NOLINTEND

CCL_NAMESPACE_BEGIN

/* SMS Manifold Vertex
 *
 * Stores the local differential geometry and constraint information for a single
 * vertex in the specular manifold chain. Unlike MNEE's ManifoldVertex which focuses
 * on refractive interfaces, this supports both reflection and refraction with
 * glossy BSDFs. */
struct SMSManifoldVertex {
  /* Position and partials for manifold parameterization */
  float3 p;       /* Vertex position */
  float3 dp_du;   /* Position partial derivative w.r.t. u */
  float3 dp_dv;   /* Position partial derivative w.r.t. v */

  /* Normal and partials */
  float3 n;       /* Shading normal */
  float3 ng;      /* Geometric normal */
  float3 dn_du;   /* Normal partial derivative w.r.t. u */
  float3 dn_dv;   /* Normal partial derivative w.r.t. v */

  /* Surface parameterization */
  float2 uv;      /* Surface UV coordinates */
  int object;     /* Object index */
  int prim;       /* Primitive index */
  int shader;     /* Shader index */

  /* BSDF information */
  float eta;      /* Index of refraction (for refractive surfaces) */
  float roughness; /* Surface roughness (for glossy surfaces) */
  int bsdf_type;  /* BSDF type flags (reflection/transmission/glossy) */

  /* Constraint matrices (2x2 encoded as float4 in row-major order)
   * These encode the manifold constraint: C(x) = 0 where x is the vertex position.
   * The constraint enforces that the path satisfies Fermat's principle. */
  float2 constraint;  /* Current constraint value C(x) */
  float4 dC_dx;       /* Jacobian: derivative of constraint w.r.t. this vertex position */
  float4 dC_dprev;    /* Jacobian: derivative of constraint w.r.t. previous vertex */
  float4 dC_dnext;    /* Jacobian: derivative of constraint w.r.t. next vertex */
};

/* SMS Configuration
 *
 * Stores the configuration and current state of an SMS solve attempt.
 * This is passed through the solver to track progress and parameters. */
struct SMSConfig {
  int num_vertices;           /* Number of specular vertices in chain */
  int diffuse_vertex_index;   /* Index of the diffuse vertex (-1 if light connection) */
  int max_iterations;         /* Maximum Newton iterations */
  float step_size;            /* Current step size (adaptive) */
  bool use_mis;               /* Use multiple importance sampling */
};

/* Utility Functions */

/* 2x2 matrix-vector multiplication (matrix encoded as row-major float4) */
ccl_device_inline float2 sms_mat22_mult_vec(const float4 mat, const float2 vec)
{
  return make_float2(mat.x * vec.x + mat.y * vec.y,
                     mat.z * vec.x + mat.w * vec.y);
}

/* 2x2 matrix-matrix multiplication (both encoded as row-major float4) */
ccl_device_inline float4 sms_mat22_mult_mat(const float4 a, const float4 b)
{
  return make_float4(
      a.x * b.x + a.y * b.z,  /* (0,0) */
      a.x * b.y + a.y * b.w,  /* (0,1) */
      a.z * b.x + a.w * b.z,  /* (1,0) */
      a.z * b.y + a.w * b.w); /* (1,1) */
}

/* Compute 2x2 matrix determinant */
ccl_device_inline float sms_mat22_determinant(const float4 mat)
{
  return mat.x * mat.w - mat.y * mat.z;
}

/* Compute 2x2 matrix inverse */
ccl_device_inline float4 sms_mat22_inverse(const float4 mat)
{
  const float det = sms_mat22_determinant(mat);
  if (fabsf(det) < SMS_MIN_DETERMINANT) {
    /* Singular matrix, return identity */
    return make_float4(1.0f, 0.0f, 0.0f, 1.0f);
  }

  const float inv_det = 1.0f / det;
  return make_float4(mat.w * inv_det, -mat.y * inv_det,
                     -mat.z * inv_det, mat.x * inv_det);
}

/* Initialize an SMS manifold vertex from intersection data */
ccl_device_inline void sms_vertex_init(
    KernelGlobals kg,
    ccl_private SMSManifoldVertex *vertex,
    ccl_private const ShaderData *sd,
    ccl_private const ShaderClosure *bsdf)
{
  /* Store position and geometric data */
  vertex->p = sd->P;
  vertex->ng = sd->Ng;
  vertex->n = sd->N;

  /* Surface parameterization */
  vertex->uv = make_float2(sd->u, sd->v);
  vertex->object = sd->object;
  vertex->prim = sd->prim;
  vertex->shader = sd->shader;

  /* BSDF properties */
  if (bsdf && CLOSURE_IS_BSDF_MICROFACET(bsdf->type)) {
    /* Extract IOR and roughness from microfacet BSDF */
    ccl_private const MicrofacetBsdf *microfacet_bsdf = (ccl_private const MicrofacetBsdf *)bsdf;
    vertex->eta = microfacet_bsdf->ior;
    vertex->roughness = 0.5f * (microfacet_bsdf->alpha_x + microfacet_bsdf->alpha_y);
    vertex->bsdf_type = bsdf->type;
  }
  else {
    vertex->eta = 1.0f;
    vertex->roughness = 0.0f;
    vertex->bsdf_type = bsdf ? bsdf->type : 0;
  }

  /* Compute surface partials (tangent frame) */
  make_orthonormals(vertex->ng, &vertex->dp_du, &vertex->dp_dv);

  /* Normal partials - approximation for now */
  vertex->dn_du = make_float3(0.0f, 0.0f, 0.0f);
  vertex->dn_dv = make_float3(0.0f, 0.0f, 0.0f);

  /* Constraint matrices initialized to zero */
  vertex->constraint = make_float2(0.0f, 0.0f);
  vertex->dC_dx = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  vertex->dC_dprev = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  vertex->dC_dnext = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

/* Compute the specular half-vector for a vertex */
ccl_device_inline float3 sms_compute_half_vector(const float3 wi, const float3 wo, float eta)
{
  /* For reflection: h = normalize(wi + wo)
   * For refraction: h = normalize(wi * eta_i + wo * eta_o)
   * For now, use simple reflection half-vector */
  return normalize(wi + wo);
}

/* Compute manifold constraint for a specular vertex
 *
 * The constraint enforces that the generalized half-vector lies in the
 * tangent plane of the surface. For a vertex with incoming direction wi
 * and outgoing direction wo:
 *
 * C(p) = [dot(h, dp_du), dot(h, dp_dv)]^T
 *
 * where h is the half-vector. When C(p) = 0, the vertex satisfies the
 * specular constraint (Fermat's principle). */
ccl_device_inline void sms_compute_constraint(
    ccl_private SMSManifoldVertex *vertex,
    const float3 p_prev,  /* Previous vertex position */
    const float3 p_next)  /* Next vertex position */
{
  /* Compute incoming and outgoing directions */
  const float3 wi = normalize(vertex->p - p_prev);
  const float3 wo = normalize(p_next - vertex->p);

  /* Compute half-vector */
  const float3 h = sms_compute_half_vector(wi, wo, vertex->eta);

  /* Constraint: half-vector must lie in tangent plane */
  vertex->constraint.x = dot(h, vertex->dp_du);
  vertex->constraint.y = dot(h, vertex->dp_dv);
}

/* Compute constraint Jacobian (derivative of constraint w.r.t. vertex positions)
 *
 * This computes dC/dx where C is the constraint and x represents the vertex
 * positions. The Jacobian is needed for the Newton solver.
 *
 * For a chain of vertices [prev, vertex, next], we need:
 * - dC/d(vertex.p): derivative w.r.t. current vertex
 * - dC/d(prev.p): derivative w.r.t. previous vertex
 * - dC/d(next.p): derivative w.r.t. next vertex */
ccl_device_inline void sms_compute_constraint_jacobian(
    ccl_private SMSManifoldVertex *vertex,
    const float3 p_prev,
    const float3 p_next)
{
  /* Compute directions and distances */
  const float3 dir_prev = vertex->p - p_prev;
  const float3 dir_next = p_next - vertex->p;
  const float dist_prev = len(dir_prev);
  const float dist_next = len(dir_next);

  if (dist_prev < SMS_MIN_PROGRESS_DISTANCE || dist_next < SMS_MIN_PROGRESS_DISTANCE) {
    /* Degenerate configuration, use identity */
    vertex->dC_dx = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    vertex->dC_dprev = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    vertex->dC_dnext = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }

  const float3 wi = dir_prev / dist_prev;
  const float3 wo = dir_next / dist_next;

  /* Compute half-vector and its derivatives
   * This is a simplified computation; full implementation would include
   * proper derivatives for refractive interfaces with IOR changes. */

  const float3 h = normalize(wi + wo);

  /* Derivative of half-vector w.r.t. vertex position */
  const float3 dh_dp = (1.0f / dist_prev) * (make_float3(1.0f, 1.0f, 1.0f) - wi * wi) +
                       (1.0f / dist_next) * (make_float3(1.0f, 1.0f, 1.0f) - wo * wo);

  /* Constraint derivatives (simplified - projects derivatives onto tangent frame) */
  /* dC/dx (current vertex) */
  vertex->dC_dx.x = dot(dh_dp, vertex->dp_du);  /* d(C.x)/d(p.u) */
  vertex->dC_dx.y = 0.0f;                        /* d(C.x)/d(p.v) */
  vertex->dC_dx.z = 0.0f;                        /* d(C.y)/d(p.u) */
  vertex->dC_dx.w = dot(dh_dp, vertex->dp_dv);  /* d(C.y)/d(p.v) */

  /* dC/dprev and dC/dnext (neighboring vertices) - approximated */
  vertex->dC_dprev = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  vertex->dC_dnext = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

/* Newton Solver for SMS
 *
 * Solves the manifold constraint system using Newton-Raphson iteration.
 * For a single specular vertex, this finds the position that satisfies:
 * C(p) = 0, where C is the manifold constraint.
 *
 * Returns true if a solution was found within max_iterations. */
ccl_device_inline bool sms_newton_solver(
    KernelGlobals kg,
    ccl_private SMSManifoldVertex *vertex,
    const float3 p_prev,
    const float3 p_next,
    const int max_iterations)
{
  float step_size = 1.0f;  /* Adaptive step size */

  for (int iter = 0; iter < max_iterations; iter++) {
    /* Compute current constraint value */
    sms_compute_constraint(vertex, p_prev, p_next);

    /* Check for convergence */
    const float constraint_norm = len(vertex->constraint);
    if (constraint_norm < SMS_SOLVER_THRESHOLD) {
      return true;  /* Solution found! */
    }

    /* Compute Jacobian */
    sms_compute_constraint_jacobian(vertex, p_prev, p_next);

    /* Solve for Newton step: delta = -J^{-1} * C */
    const float4 J_inv = sms_mat22_inverse(vertex->dC_dx);
    const float2 delta = sms_mat22_mult_vec(J_inv, -vertex->constraint);

    /* Adaptive step size: reduce if step is too large */
    const float delta_norm = len(delta);
    if (delta_norm < SMS_MINIMUM_STEP_SIZE) {
      /* Step too small, likely converged or stuck */
      return constraint_norm < (SMS_SOLVER_THRESHOLD * 10.0f);
    }

    if (delta_norm > 0.1f) {
      step_size *= 0.5f;  /* Reduce step size for stability */
    }
    else if (delta_norm < 0.01f && step_size < 1.0f) {
      step_size *= 2.0f;  /* Increase step size if making good progress */
      step_size = min(step_size, 1.0f);
    }

    /* Update vertex position in UV space, then project to surface
     * For now, simplified: directly update position
     * TODO: Proper surface projection using ray casting */
    vertex->p = vertex->p + step_size * (delta.x * vertex->dp_du + delta.y * vertex->dp_dv);
  }

  /* Failed to converge */
  return false;
}

/* Solve manifold for a complete specular chain
 *
 * Given a chain of vertices [p0, v1, v2, ..., vN, p_end] where v_i are specular
 * vertices and p0, p_end are fixed endpoints (diffuse vertex and light),
 * solve for the positions of all v_i that satisfy the manifold constraints.
 *
 * This is the core SMS algorithm. For Phase 3.1, we implement a simplified
 * version that solves vertices sequentially. Phase 3.2 will add the full
 * block Newton solver for simultaneous solving. */
ccl_device_inline bool sms_solve_manifold(
    KernelGlobals kg,
    ccl_private SMSManifoldVertex *vertices,
    const int num_vertices,
    const float3 p_start,  /* Starting position (e.g., camera or diffuse vertex) */
    const float3 p_end,    /* Ending position (e.g., light sample point) */
    const int max_iterations)
{
  /* Simplified solver: solve each vertex independently (Gauss-Seidel style)
   * TODO Phase 3.2: Implement full block Newton solver */

  bool all_converged = true;

  /* Iterate multiple times to allow vertices to influence each other */
  const int outer_iterations = 3;
  for (int outer = 0; outer < outer_iterations; outer++) {
    /* Solve each vertex */
    for (int i = 0; i < num_vertices; i++) {
      const float3 prev_p = (i == 0) ? p_start : vertices[i - 1].p;
      const float3 next_p = (i == num_vertices - 1) ? p_end : vertices[i + 1].p;

      const bool converged = sms_newton_solver(
          kg, &vertices[i], prev_p, next_p, max_iterations / outer_iterations);

      all_converged = all_converged && converged;
    }

    if (all_converged) {
      return true;  /* All vertices converged */
    }
  }

  return all_converged;
}

/* Main SMS Sampling Function
 *
 * Attempts to solve for a caustic path using Specular Manifold Sampling.
 * Called when the path tracer detects a potential S-D-S configuration.
 *
 * Parameters:
 *   kg: Kernel globals
 *   state: Current integrator state
 *   sd: Shader data at current vertex (diffuse surface)
 *   rng_state: Random number generator state
 *
 * Returns:
 *   true if SMS successfully found and sampled a caustic path
 *   false if SMS failed or is not applicable
 *
 * Phase 3.1 Implementation Status:
 *   - Framework in place with detection logic
 *   - Basic vertex chain construction
 *   - TODO Phase 3.2: Complete path construction and contribution calculation
 *   - TODO Phase 3.3: Integration with light sampling and MIS
 */
ccl_device_forceinline bool kernel_path_sms_sample(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    ccl_private const RNGState *rng_state)
{
  /* Check if SMS is enabled globally */
  if (kernel_data.integrator.caustics_mode != CAUSTICS_FULL) {
    return false;  /* SMS not enabled, use standard path tracing */
  }

  /* Check if we have specular vertices in the chain */
  const uint8_t vertex_count = INTEGRATOR_STATE(state, sms, vertex_count);
  if (vertex_count == 0 || vertex_count > SMS_MAX_VERTICES) {
    return false;  /* No specular chain or chain too long */
  }

  /* Check if current surface is diffuse (required for S-D-S pattern) */
  const int shader_flags = kernel_data_fetch(shaders, sd->shader).flags;
  if (!(shader_flags & SD_HAS_ONLY_VOLUME)) {
    /* Has surface interaction - check if it's diffuse */
    bool has_diffuse = false;
    for (int i = 0; i < sd->num_closure; i++) {
      const ccl_private ShaderClosure *sc = &sd->closure[i];
      if (CLOSURE_IS_BSDF_DIFFUSE(sc->type)) {
        has_diffuse = true;
        break;
      }
    }

    if (!has_diffuse) {
      return false;  /* Not a diffuse vertex, can't be the D in S-D-S */
    }
  }

  /* TODO Phase 3.2: Implement full SMS sampling */
  /* Steps needed:
   * 1. Sample a light source
   * 2. Build the complete vertex chain from path history
   * 3. Invoke sms_solve_manifold() to find the caustic path
   * 4. Evaluate path contribution (throughput * BSDF * geometry terms)
   * 5. Apply MIS weight combining SMS with standard path tracing
   * 6. Update integrator state with the sampled path
   */

  /* For Phase 3.1, just mark that SMS was attempted but didn't produce a sample */
  INTEGRATOR_STATE_WRITE(state, path, caustics) |= PATH_SMS_VALID;

  return false;  /* No sample yet - full implementation in Phase 3.2 */
}

CCL_NAMESPACE_END
