# AGENTS.md — Cycles CPU Manifold Path Guiding (MPG)

Purpose: This document tells any code-generation agent exactly what to implement in Blender Cycles (CPU-only) to add Manifold Path Guiding (MPG) and hook it into existing OpenPGL path guiding, without leaving placeholders or confusing scaffolding.

AGENTS

0) Non-negotiables

CPU only. Do not touch CUDA/HIP/Metal codepaths in this task.

No volumes v1. Skip when inside or crossing volumetric media.

No placeholders / TODO stubs. Every added function must be fully implemented or the feature gated off. Do not add “WIP” paths, dead flags, or empty classes.

Unbiasedness preserved. MPG is an extra sampling proposal; combine via MIS with existing BSDF/NEE/OpenPGL proposals using PDFs in the same measure (solid angle at the shading point).

Minimal blast radius. New code must live under a new folder (intern/cycles/manifold/) and integrate through well-scoped calls. Do not refactor unrelated systems.

Do not rely on OpenPGL internals. The seeding/gating “guide summary” must be estimated from the public OpenPGL sampler/pdf; do not include or depend on private headers or internal lobes.

1) Goal (single sentence)

Implement MPG for 1–2 specular bounces (reflection/refraction) at CPU surface hits in Cycles and MIS-combine its contribution with BSDF + NEE + OpenPGL, with an estimated OpenPGL-driven gate to avoid wasted attempts.

2) Scope (v1)

Specular chain length: up to 2 (D→S→L, D→S→S→L).

Materials supported: perfect reflection & refraction on triangle meshes with smooth normals (ignore bump/normal maps inside the solver).

Lights supported: Sun, area lights, and environment maps with small solid angle (treat env sun as small disc).

Where it runs: CPU surface shading path only.

What it uses from OpenPGL: a GuideSummary estimated from samples at the hit (dominant direction, “peak mass” in a cone, concentration κ estimate) for gating + seeding. (No internal getters.)

What it records back: optional training sample from successful MPG direction (pos, dir, weight) via existing guiding buffer API.

Out of scope v1: volumes, microfacet roughness chains, GPU backends, long multi-bounce chains, SMS/VCM/VM, UI for per-material overrides.

3) New module (files and responsibilities)

Create intern/cycles/manifold/ with these files:

mpg.h / mpg.cpp — Public API for a single attempt at a manifold connection from shading point D.

mpg_solve.h / mpg_solve.cpp — Newton/trust-region solver for 1–2 specular bounces (Snell/reflect constraints, residuals, Jacobians, line search).

mpg_pdf.h — Computes the proposal PDF w.r.t. solid angle at D (includes change-of-variables Jacobian).

mpg_seed.h / mpg_seed.cpp — Seeding logic using GuideSummary (endpoint choice + initial direction); early reject if no candidate specular is hit.

NEW: mpg_pgl_summary.h / mpg_pgl_summary.cpp — Sampling-based estimator that computes GuideSummary from the public OpenPGL distribution at the surface hit.

Public API (exact signature shape; adapt types to the tree)
namespace ccl {

struct MpgOptions {
int   max_bounces = 1;        // 1 or 2
int   max_iters   = 6;        // solver steps
float gate_w      = 0.35f;    // peak mass threshold in a cone
float gate_kappa  = 40.0f;    // sharpness threshold
float angular_jitter = 0.02f; // radians for seed jitter
};

struct GuideSummary {
float3 mean_dir;   // world-space dominant (mean resultant) direction
float  peak_weight; // mass within a small cone about mean_dir (fraction 0..1)
float  kappa;       // vMF concentration estimate from Rbar
float  rbar;        // mean resultant length (0..1) for diagnostics
};

struct MpgResult {
bool   success = false;
float3 wi;         // direction to sample at D (world)
float  pdf;        // solid-angle pdf at D for wi
float  visibility; // 0..1; shadow term checked during solve
};

MpgResult mpg_try_connect(const ShadingPoint& D,
const ClosureBSDF& bsdf,
const Lights& lights,
const GuideSummary& g,
const MpgOptions& opt,
RNG& rng);

/* Estimate GuideSummary from OpenPGL public sampler/pdf (no private headers). */
bool pgl_estimate_summary(const OpenPGLSurfaceDistribution& dist_world,
const float3& Ng_world, /* geometric normal for hemisphere */
RNG& rng,
GuideSummary& out,
int n_samples = 128,
float cone_half_angle_rad = radians(10.0f));

} // namespace ccl


Estimator details (must implement):

Draw n_samples directions {ω_i} from dist_world (respect hemisphere via Ng_world if needed).

Compute mean resultant m = (1/N) * Σ ω_i, out.mean_dir = normalize(m), and Rbar = ||m||.

Estimate κ from Rbar using a closed-form approximation (e.g., κ ≈ Rbar*(3 - Rbar*Rbar)/(1 - Rbar*Rbar)), then clamp to [0, 1e4]. Optionally do one Newton step on A(κ)=Rbar for refinement.

Compute peak mass as the fraction of samples with angle(ω_i, out.mean_dir) ≤ cone_half_angle_rad. Set out.peak_weight = count/N.

If Rbar < 1e-3 (nearly uniform), return false (no reliable guidance).

4) Integrator hook (CPU)

At the CPU surface shading site where BSDF and OpenPGL guided proposals are handled:

Estimate GuideSummary via pgl_estimate_summary(...) from the public OpenPGL distribution at the hit. Do not call or add internal getters.

Gate:

Skip MPG unless peak_weight >= gate_w and kappa >= gate_kappa and BSDF is non-delta.

Attempt mpg_try_connect. On success:

Evaluate f = bsdf_eval(D, r.wi), cosNI = max(dot(N, r.wi), 0).

Contribution: throughput * f * cosNI * r.visibility / r.pdf.

Compute MIS with all active proposals at this bounce (same measure, solid angle at D):

p_bsdf, p_guided (OpenPGL), p_nee (if attempted), p_mpg = r.pdf.

Weight: w_mpg = p_mpg / (p_bsdf + p_guided + p_nee + p_mpg).

Accumulate w_mpg * contribution.

(Optional) Call guiding training hook to add (D.pos, r.wi, weight=luminance(contribution)).

MPG is additive and gated; do not alter other proposals.

5) Scene flags & UI

Add scene options (defaults shown):

manifold_guiding_enable = false

manifold_max_bounces = 1

manifold_iters = 6

manifold_gate_weight = 0.35

manifold_gate_kappa = 40.0

(internal/advanced) manifold_summary_samples = 128, manifold_summary_cone_deg = 10.0 (optional; if not exposed, keep as constants in the estimator)

Expose under Sampling ▸ Guiding as “Manifold Path Guiding (CPU, Experimental)”. When disabled, no codepath changes must execute.

6) Build / CMake

Add intern/cycles/manifold/*.cpp to the Cycles CPU target only.

Introduce WITH_CYCLES_MANIFOLD CMake option and guard code/UI with it.

Do not alter GPU builds. Do not add global compiler flags that affect other 3rd-party libs.

7) Acceptance criteria (Definition of Done)

Functional

Cornell + glass slab + Sun: at 64 spp, MPG ON shows a clear caustic vs OFF. At 16k spp, MPG matches reference within 1% RMSE (no bias).

Env “sun pixel” + pool bottom: MPG ON reduces salt-and-pepper speckle at equal spp.

Non-caustic interior: with MPG enabled but gate unmet, results equal (within noise) and render time within ±5% of baseline.

Engineering

Builds pass on Windows/MSVC and Linux/Clang/GCC for CPU device.

Zero new warnings in touched files at default warning level.

No TODO/FIXME placeholders. No unused functions. No dead flags.

Code style follows Cycles conventions; functions documented with brief doxygen comments.

Safety

If solver fails or returns pdf<=0, MPG contribution is skipped without affecting other proposals.

All PDFs measured in solid angle at D; MIS uses balance heuristic consistently.

8) Guardrails

Do not refactor existing integrator architecture beyond inserting the MPG callsite and wiring options.

Do not modify or include OpenPGL internals; use only public sample/pdf calls for the summary.

Do not add experimental features (volumes, rough microfacets, long chains). Defer with a clear comment in the README, not as code stubs.

If an exact type or file path differs, search by symbol (e.g., “openpgl”, “guiding”, “shade_surface”, “bsdf_eval”) and adapt—do not leave partial edits.

9) Test plan (manual)

Build Blender with WITH_CYCLES_MANIFOLD=ON (CPU device).

Render the three acceptance scenes (provided or simple to construct).

Compare images (RMSE or eyeball) at low spp (variance) and high spp (bias check).

Toggle MPG on/off and move thresholds to validate gating behavior.

10) README note for the module

Create intern/cycles/manifold/README.md summarizing:

What MPG does, v1 limitations, PDFs/MIS measure, gating thresholds.

Sampling-based OpenPGL summary approach (no internal getters).

How to run the acceptance scenes.