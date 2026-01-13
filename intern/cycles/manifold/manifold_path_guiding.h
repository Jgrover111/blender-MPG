/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Manifold Path Guiding (MPG) - Cycles implementation
 *
 * This file matches the structure of the Mitsuba MPG reference implementation:
 * https://github.com/mollnn/manifold-path-guiding/tree/main/mitsuba/src/integrators/MPG
 *
 * File structure corresponds to: manifold_path_guiding.h in the Mitsuba reference
 *
 * Organization (matching Mitsuba):
 * 1. ManifoldPathGuidingConfig - Configuration structure
 * 2. ManifoldVertex - Vertex on specular manifold
 * 3. EmitterInteraction - Light source interaction
 * 4. SpecularManifold - Static utility class for geometric operations
 * 5. Manifold_Walk - Newton solver class
 *
 * Key differences from Mitsuba:
 * - Uses Cycles types (float3, Spectrum) instead of Mitsuba templates
 * - Integrates with OpenPGL for spatial/directional distributions
 * - CPU-only implementation (no GPU kernel variants)
 */

#pragma once

#include "integrator/guiding.h"

#include "kernel/device/cpu/globals.h"
#include "kernel/integrator/path_state.h"
#include "kernel/light/common.h"
#include "kernel/light/sample.h"
#include "kernel/types.h"

#include "util/math.h"

#include <cstdint>

CCL_NAMESPACE_BEGIN

/* ========================================================================
 * Configuration Structure
 * Corresponds to: ManifoldPathGuidingConfig in Mitsuba
 * ======================================================================== */

struct ManifoldPathGuidingConfig {
  /* Solver parameters */
  int max_bounces = 2;           /* Maximum specular bounces (sms_max_depth in Mitsuba) */
  int max_iterations = 20;       /* Newton solver iterations */
  float solver_threshold = 1e-4f;
  float step_scale = 1.0f;       /* Newton step scaling factor */
  bool halfvector_constraints = false;

  /* Guiding parameters */
  bool guided = true;
  float gate_w = 0.35f;          /* Gate weight threshold */
  float gate_kappa = 40.0f;      /* Gate concentration threshold */
  bool relax_gate = true;        /* Allow bootstrap when guide not ready */

  /* Sampling parameters */
  int max_seed_repeat_trials = 64; /* Seed generation retry budget */
  float prob_uniform = 0.1f;        /* Uniform sampling probability */

  ManifoldPathGuidingConfig() {}

  std::string to_string() const;
};

/* Convenience alias matching common usage */
using MpgConfig = ManifoldPathGuidingConfig;

/* ========================================================================
 * ManifoldVertex Structure
 * Corresponds to: ManifoldVertex<Float_, Spectrum_> in Mitsuba
 * ======================================================================== */

struct ManifoldVertex {
  /* Surface point and derivatives (corresponds to p, dp_du, dp_dv) */
  float3 p = zero_float3();
  float3 dp_du = zero_float3();
  float3 dp_dv = zero_float3();

  /* Surface normals and derivatives (corresponds to n, gn, dn_du, dn_dv) */
  float3 n = zero_float3();   /* Shading normal */
  float3 gn = zero_float3();  /* Geometric normal */
  float3 dn_du = zero_float3();
  float3 dn_dv = zero_float3();

  /* Scattering type */
  bool is_refraction = false;

  /* Tangent frame and derivatives (corresponds to s, t, ds_du, dt_du, etc) */
  float3 s = zero_float3();
  float3 t = zero_float3();
  float3 ds_du = zero_float3();
  float3 ds_dv = zero_float3();
  float3 dt_du = zero_float3();
  float3 dt_dv = zero_float3();

  /* Material properties */
  float eta = 1.0f;
  float2 uv = zero_float2();

  /* Shape reference */
  int object = -1;
  int prim = -1;

  /* Newton solver data (corresponds to C, dC_dx_prev, dC_dx_cur, dC_dx_next, dx) */
  float2 C = zero_float2();
  float dC_dx_prev[2][2] = {{0, 0}, {0, 0}};
  float dC_dx_cur[2][2] = {{0, 0}, {0, 0}};
  float dC_dx_next[2][2] = {{0, 0}, {0, 0}};
  float2 dx = zero_float2();

  /* Cached path contribution data */
  float3 wi = zero_float3();
  float3 wo = zero_float3();
  float dist_in = 0.0f;
  float dist_out = 0.0f;
  float cos_theta_in = 0.0f;
  float cos_theta_out = 0.0f;
  Spectrum throughput = zero_spectrum();

  /* Constructors */
  ManifoldVertex() {}
  ManifoldVertex(const float3 &position) : p(position) {}

  /* Initialize from surface interaction */
  void from_surface_interaction(const ShaderData &sd);

  /* Make tangent frame orthonormal (corresponds to make_orthonormal()) */
  void make_orthonormal();
};

/* ========================================================================
 * EmitterInteraction Structure
 * Corresponds to: EmitterInteraction<Float_, Spectrum_> in Mitsuba
 * ======================================================================== */

struct EmitterInteraction {
  /* Light source position and normal */
  float3 p = zero_float3();
  float3 n = zero_float3();

  /* Direction from surface to light */
  float3 d = zero_float3();

  /* UV coordinates and derivatives */
  float2 uv = zero_float2();
  float3 dp_du = zero_float3();
  float3 dp_dv = zero_float3();

  /* Sampling weight and PDF */
  Spectrum weight = zero_spectrum();
  float pdf = 0.0f;

  /* Light sample reference */
  LightSample light_sample = {};

  /* Type queries (corresponds to is_point(), is_directional(), is_area(), is_delta()) */
  bool is_point() const;
  bool is_directional() const;
  bool is_area() const;
  bool is_delta() const;
};

/* ========================================================================
 * SpecularManifold - Static Utility Class
 * Corresponds to: SpecularManifold<Float_, Spectrum_> in Mitsuba
 * All methods are static geometric/sampling utilities
 * ======================================================================== */

struct SpecularManifold {
  /* Emitter sampling and interaction methods */
  static EmitterInteraction sample_emitter_interaction(KernelGlobals kg,
                                                       const ShaderData &sd,
                                                       const uint32_t path_flag,
                                                       RNGState &rng_state);

  static bool emitter_interaction_to_vertex(KernelGlobals kg,
                                           const EmitterInteraction &ei,
                                           const float3 &source_p,
                                           float time,
                                           ManifoldVertex &vertex);

  static EmitterInteraction emitter_interaction(KernelGlobals kg,
                                               const ShaderData &sd,
                                               const ShaderData &light_sd);

  /* Reflection and refraction with derivatives */
  static bool reflect(const float3 &wi, const float3 &n, float3 &wo);

  static void d_reflect(const float3 &wi,
                       const float3 &d_wi_du,
                       const float3 &d_wi_dv,
                       const float3 &n,
                       const float3 &dn_du,
                       const float3 &dn_dv,
                       float3 &d_wo_du,
                       float3 &d_wo_dv);

  static bool refract(const float3 &wi, const float3 &n, float eta, float3 &wo);

  static void d_refract(const float3 &wi,
                       const float3 &d_wi_du,
                       const float3 &d_wi_dv,
                       const float3 &n,
                       const float3 &dn_du,
                       const float3 &dn_dv,
                       float eta,
                       float3 &d_wo_du,
                       float3 &d_wo_dv);

  /* Spherical coordinate transformations with derivatives */
  static void sphcoords(const float3 &v, float &theta, float &phi);

  static void d_sphcoords(const float3 &v,
                         const float3 &dv_du,
                         const float3 &dv_dv,
                         float &dtheta_du,
                         float &dtheta_dv,
                         float &dphi_du,
                         float &dphi_dv);

  /* Path contribution evaluation */
  static Spectrum specular_reflectance(KernelGlobals kg,
                                      const ShaderData &sd,
                                      const EmitterInteraction &ei,
                                      const std::vector<ManifoldVertex> &vertices);

  static float geometric_term(const ManifoldVertex &v_prev,
                             const ManifoldVertex &v_cur,
                             std::vector<ManifoldVertex> &vertices);

  static float invert_tridiagonal_geo(std::vector<ManifoldVertex> &vertices);

  static Spectrum evaluate_path_contribution(KernelGlobals kg,
                                            const std::vector<ManifoldVertex> &vertices,
                                            const ShaderData &sd,
                                            const EmitterInteraction &ei);
};

/* ========================================================================
 * Manifold_Walk - Newton Solver Class
 * Corresponds to: Manifold_Walk<Float, Spectrum> in Mitsuba
 * ======================================================================== */

class ManifoldWalk {
public:
  /* Configuration and scene reference */
  ManifoldPathGuidingConfig config;
  KernelGlobals kg = nullptr;

  /* Constructors */
  ManifoldWalk() {}
  ManifoldWalk(KernelGlobals kg_, ManifoldPathGuidingConfig config_) : config(config_), kg(kg_) {}

  /* Main Newton solver
   * Corresponds to: newton_solver() in Mitsuba */
  bool newton_solver(const ShaderData &sd,
                    const EmitterInteraction &ei,
                    std::vector<ManifoldVertex> &vertices,
                    const std::vector<float3> &offset_normals);

  /* Constraint computation methods
   * Corresponds to: compute_step_halfvector() and compute_step_anglediff() */
  bool compute_step_halfvector(const ShaderData &sd,
                               const EmitterInteraction &ei,
                               std::vector<ManifoldVertex> &vertices,
                               const std::vector<float3> &offset_normals);

  bool compute_step_anglediff(const ShaderData &sd,
                             const EmitterInteraction &ei,
                             std::vector<ManifoldVertex> &vertices,
                             const std::vector<float3> &offset_normals);

  /* Surface reprojection
   * Corresponds to: reproject() in Mitsuba */
  bool reproject(const ShaderData &sd,
                const std::vector<float3> &target_positions,
                const std::vector<ManifoldVertex> &seed_vertices,
                const std::vector<float3> &offset_normals,
                std::vector<ManifoldVertex> &result_vertices);

  /* Tridiagonal matrix inversion for Newton step
   * Corresponds to: invert_tridiagonal_step() in Mitsuba */
  bool invert_tridiagonal_step(std::vector<ManifoldVertex> &vertices);
};

/* ========================================================================
 * Helper Structures (Cycles-specific, not in Mitsuba)
 * ======================================================================== */

/* Failure codes and status enums */
enum MpgFailureCode : int {
  MPG_FAILURE_NONE = 0,
  MPG_FAILURE_SEED = 1,
  MPG_FAILURE_TOTAL_INTERNAL_REFLECTION = 2,
  MPG_FAILURE_DEGENERATE_NORMALS = 3,
  MPG_FAILURE_NEWTON_DIVERGED = 4,
  MPG_FAILURE_JACOBIAN_ZERO = 5,
  MPG_FAILURE_GATE = 6,
  MPG_FAILURE_GEOMETRY = 7,
  MPG_FAILURE_ZERO_THROUGHPUT = 8,
  MPG_FAILURE_INVALID_LIGHT_PDF = 9,
  MPG_FAILURE_INVALID_SEED_PDF = 10,
  MPG_FAILURE_INVALID_NEE_PDF = 11,
  MPG_FAILURE_INVALID_PDF = 12,
  MPG_FAILURE_NO_SPECULAR = 13,
  MPG_FAILURE_UNSUPPORTED = 14,
  MPG_FAILURE_UNKNOWN = 15
};

enum MpgGateMask : uint32_t {
  MPG_GATE_MASK_NONE = 0,
  MPG_GATE_MASK_ACTIVE = 1u << 0,
  MPG_GATE_MASK_HAS_DIRECTION_RELAXED = 1u << 1,
  MPG_GATE_MASK_HAS_DIRECTION_STRICT = 1u << 2,
  MPG_GATE_MASK_STRICT_PASS = 1u << 3,
  MPG_GATE_MASK_RELAX_PASS = 1u << 4,
  MPG_GATE_MASK_BOOTSTRAP_PASS = 1u << 5
};

enum MpgSeedBranch : int {
  MPG_SEED_BRANCH_NONE = 0,
  MPG_SEED_BRANCH_GUIDED = 1,
  MPG_SEED_BRANCH_FALLBACK = 2,
};

enum MpgSeedScatter : int {
  MPG_SEED_SCATTER_NONE = 0,
  MPG_SEED_SCATTER_REFLECTION = 1,
  MPG_SEED_SCATTER_REFRACTION = 2,
};

/* Seed ray proposal structure */
struct MpgSeedRay {
  float3 direction = zero_float3();
  float3 direction_normalized = zero_float3();
  float seed_pdf = 0.0f;
  float seed_pdf_raw = 0.0f;
  float seed_branch_pdf = 0.0f;
  float seed_direction_pdf = 0.0f;
  float seed_scatter_pdf = 0.0f;
  float bounce_pdf = 1.0f;
  float bounce_pdf_raw = 1.0f;
  LightSample light_sample = {};
  uint32_t path_flag = 0;
  int object = -1;
  int prim = -1;
  float bary_u = 0.0f;
  float bary_v = 0.0f;
  int trial_count = 0;
  int accepted_trial_count = 0;
  int guided_trial_count = 0;
  int fallback_trial_count = 0;
  float seed_resample_factor = 0.0f;
  MpgSeedBranch branch = MPG_SEED_BRANCH_NONE;
  MpgSeedScatter scatter = MPG_SEED_SCATTER_NONE;
  int bounce_count = 1;
  uint8_t tau_bits = 0;
  uint8_t tau_count = 0;
  bool use_smooth_normals = false;
};

/* Solver output structure */
struct MpgSolverOutput {
  bool success = false;
  int specular_vertex_count = 0;
  float3 wi = zero_float3();
  float visibility = 1.0f;
  float jacobian_total = 0.0f;
  Spectrum specular_throughput = zero_spectrum();
  float seed_resample_factor = 0.0f;
  MpgSeedBranch seed_branch = MPG_SEED_BRANCH_NONE;
  MpgSeedScatter seed_scatter = MPG_SEED_SCATTER_NONE;
  float seed_branch_pdf = 0.0f;
  float seed_direction_pdf = 0.0f;
  float seed_scatter_pdf = 0.0f;
  int seed_guided_trial_count = 0;
  int seed_fallback_trial_count = 0;
  std::vector<ManifoldVertex> specular_vertices;
};

/* Main result structure */
struct MpgResult {
  bool success = false;
  float3 wi = zero_float3();
  float pdf = 0.0f;
  float nee_pdf = 0.0f;
  float visibility = 0.0f;
  float jacobian_total = 0.0f;
  float seed_pdf = 0.0f;
  float seed_pdf_raw = 0.0f;
  float seed_resample_factor = 0.0f;
  float seed_branch_pdf = 0.0f;
  float seed_direction_pdf = 0.0f;
  float seed_scatter_pdf = 0.0f;
  float3 seed_direction_normalized = zero_float3();
  uint8_t seed_tau_bits = 0;
  uint8_t seed_tau_count = 0;
  float bounce_pdf = 1.0f;
  float bounce_pdf_raw = 1.0f;
  float light_pdf = 0.0f;
  Spectrum spec_weight = zero_spectrum();
  LightSample light = {};
  std::vector<ManifoldVertex> specular_vertices;
  int attempt_count = 0;
  int seed_trial_count = 0;
  int seed_accepted_trial_count = 0;
  int seed_guided_trial_count = 0;
  int seed_fallback_trial_count = 0;
  MpgSeedBranch seed_branch = MPG_SEED_BRANCH_NONE;
  MpgSeedScatter seed_scatter = MPG_SEED_SCATTER_NONE;
  int bounce_count = 1;
  uint32_t gate_mask = MPG_GATE_MASK_NONE;
  MpgFailureCode failure_code = MPG_FAILURE_NONE;
};

/* ========================================================================
 * Top-Level Functions (Cycles integration layer)
 * ======================================================================== */

/* Main entry point for Manifold Path Guiding */
MpgResult mpg_try_connect(KernelGlobals kg,
                          const ShaderData &sd,
                          const ShaderClosure &bsdf,
                          const GuideSummary &g,
                          const ManifoldPathGuidingConfig &config,
                          const uint32_t path_flag,
                          const int bounce,
                          RNGState &rng_state);

/* Seed generation */
bool mpg_generate_seed(KernelGlobals kg,
                       const ShaderData &sd,
                       const ShaderClosure &bsdf,
                       const GuideSummary &guide,
                       const ManifoldPathGuidingConfig &config,
                       const uint32_t path_flag,
                       const int bounce,
                       const RNGState &rng_state,
                       MpgSeedRay &seed,
                       MpgFailureCode &failure_code,
                       const int rng_branch_offset = 0);

/* Manifold walk solvers */
bool mpg_solve_single_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
                             const GuideSummary &guide,
                             const ManifoldPathGuidingConfig &config,
                             RNGState &rng_state,
                             MpgSolverOutput &result,
                             MpgFailureCode &failure_code);

bool mpg_solve_double_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
                             const GuideSummary &guide,
                             const ManifoldPathGuidingConfig &config,
                             RNGState &rng_state,
                             MpgSolverOutput &result,
                             MpgFailureCode &failure_code);

/* PDF evaluation and utilities */
bool mpg_evaluate_pdf(KernelGlobals kg,
                      const ShaderData &sd,
                      const ShaderClosure &bsdf,
                      const GuideSummary &guide,
                      const MpgSeedRay &seed,
                      const MpgSolverOutput &solution,
                      float &pdf);

float mpg_light_sample_pdf_solid(KernelGlobals kg,
                                 const ShaderData &sd,
                                 const LightSample &light_sample);

CCL_NAMESPACE_END
