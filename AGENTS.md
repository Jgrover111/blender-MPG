# AGENTS.md — Cycles CPU Manifold Path Guiding (MPG)

**Purpose:** This document tells any code-generation agent exactly what to implement in Blender **Cycles (CPU-only)** to add **Manifold Path Guiding (MPG)** and hook it into existing **OpenPGL path guiding**, without leaving placeholders or confusing scaffolding.

---

## 0) Non‑negotiables

- **CPU only**. Do **not** touch CUDA/HIP/Metal codepaths in this task.
- **No volumes v1**. Skip when inside or crossing volumetric media.
- **No placeholders / TODO stubs.** Every added function must be fully implemented or the feature gated off. Do not add “WIP” paths, dead flags, or empty classes.
- **Unbiasedness preserved.** MPG is an extra **sampling proposal**; combine via MIS with existing BSDF/NEE/OpenPGL proposals using PDFs in the same measure (solid angle at the shading point).
- **Minimal blast radius.** New code must live under a new folder (`intern/cycles/manifold/`) and integrate through well‑scoped calls. Do not refactor unrelated systems.

---

## 1) Goal (single sentence)

Implement **MPG for 1–2 specular bounces (reflection/refraction)** at CPU surface hits in Cycles and **MIS-combine** its contribution with BSDF + NEE + OpenPGL, with an **OpenPGL‑driven gate** to avoid wasted attempts.

---

## 2) Scope (v1)

- **Specular chain length:** up to **2** (D→S→L, D→S→S→L).
- **Materials supported:** perfect reflection & refraction on triangle meshes with **smooth normals** (ignore bump/normal maps inside the solver).
- **Lights supported:** Sun, area lights, and environment maps with small solid angle (treat env sun as small disc).
- **Where it runs:** CPU surface shading path only.
- **What it uses from OpenPGL:** a **GuideSummary** at the hit (top‑mode direction, peak weight, concentration κ) for gating + seeding.
- **What it records back:** optional training sample from successful MPG direction (pos, dir, weight) via existing guiding buffer API.

**Out of scope v1:** volumes, microfacet roughness chains, GPU backends, long multi‑bounce chains, SMS/VCM/VM, UI for per‑material overrides.

---

## 3) New module (files and responsibilities)

Create `intern/cycles/manifold/` with these files:

- `mpg.h / mpg.cpp` — **Public API** for a single attempt at a manifold connection from shading point D.
- `mpg_solve.h / mpg_solve.cpp` — **Newton/trust‑region** solver for 1–2 specular bounces (Snell/reflect constraints, residuals, Jacobians, line search).
- `mpg_pdf.h` — Computes the **proposal PDF** w.r.t. **solid angle at D** (includes change‑of‑variables Jacobian).
- `mpg_seed.h / mpg_seed.cpp` — Seeding logic using OpenPGL **GuideSummary** (endpoint choice + initial direction); early reject if no candidate specular is hit.

### Public API (exact signature shape; adapt types to the tree)

```cpp
namespace ccl {
struct MpgOptions {
  int   max_bounces = 1;        // 1 or 2
  int   max_iters   = 6;        // solver steps
  float gate_w      = 0.35f;    // OpenPGL peak weight threshold
  float gate_kappa  = 40.0f;    // OpenPGL sharpness threshold
  float angular_jitter = 0.02f; // radians for seed jitter
};

struct GuideSummary {
  float3 mean_dir;   // world-space top mode
  float  peak_weight;
  float  kappa;
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
} // namespace ccl
```

**Implementation rules:**
- `mpg_try_connect` **must not** throw or crash; on failure return `success=false`.
- `mpg_solve` must check **TIR**, invalid UVs, and terminate after `max_iters` with early out if residual increases twice.
- `mpg_pdf` must return **exact** solid‑angle pdf at D for the final path (seed density × |det ∂seed/∂ω_D|). If you can’t derive part of it, **do not guess**; return `success=false` instead of an invalid PDF.

---

## 4) Integrator hook (CPU)

At the CPU surface shading site where **BSDF** and **OpenPGL guided** proposals are handled:

1. Query **OpenPGL GuideSummary** (`mean_dir`, `peak_weight`, `kappa`) at D.
2. **Gate**:
   - Skip MPG unless `peak_weight >= gate_w` and `kappa >= gate_kappa` and BSDF is **non‑delta**.
3. **Attempt** `mpg_try_connect`. On success:
   - Evaluate `f = bsdf_eval(D, r.wi)`, `cosNI = max(dot(N, r.wi), 0)`.
   - Contribution: `throughput * f * cosNI * r.visibility / r.pdf`.
   - Compute **MIS** with all active proposals at this bounce (**same measure**, solid angle at D):
     - `p_bsdf`, `p_guided` (OpenPGL), `p_nee` (if attempted), `p_mpg = r.pdf`.
     - Weight: `w_mpg = p_mpg / (p_bsdf + p_guided + p_nee + p_mpg)`.
   - Accumulate `w_mpg * contribution` to the path radiance.
   - (Optional) Call guiding training hook to add `(D.pos, r.wi, weight=luminance(contribution))`.

**Do not** change the behavior of existing proposals; MPG is additive and gated.

---

## 5) Scene flags & UI

Add scene options (defaults shown):

- `manifold_guiding_enable = false`
- `manifold_max_bounces = 1`
- `manifold_iters = 6`
- `manifold_gate_weight = 0.35`
- `manifold_gate_kappa = 40.0`

Expose in the Cycles CPU panel under **Sampling ▸ Guiding** as “**Manifold Path Guiding (CPU, Experimental)**” with sub‑settings above. When disabled, no codepath changes must execute.

---

## 6) Build / CMake

- Add `intern/cycles/manifold/*.cpp` to the **Cycles CPU** target only.
- Introduce `WITH_CYCLES_MANIFOLD` CMake option and guard code/UI with it.
- Do **not** alter GPU builds. Do **not** add global compiler flags that affect other 3rd‑party libs.

---

## 7) Acceptance criteria (Definition of Done)

**Functional**
- Cornell + glass slab + Sun: at **64 spp**, MPG ON shows a clear caustic compared to OFF. At **16k spp**, MPG matches reference within 1% RMSE (no bias).
- Env “sun pixel” + pool bottom: MPG ON eliminates salt‑and‑pepper speckle vs OFF at equal spp.
- Non‑caustic scene: with MPG enabled but gate unmet, results equal (within noise) and render time within **±5%** of baseline.

**Engineering**
- Builds pass on **Windows/MSVC** and **Linux/Clang/GCC** for **CPU device**.
- Zero new warnings at default warning level in touched files.
- No TODO/FIXME placeholders. No unused functions. No dead flags.
- Code style follows Cycles conventions; functions documented with brief doxygen comments.

**Safety**
- If solver fails or returns `pdf<=0`, MPG contribution is skipped without affecting other proposals.
- All PDFs measured in **solid angle at D**; MIS uses **balance heuristic** consistently.

---

## 8) Guardrails (keep the agent on task)

- Do **not** refactor existing integrator architecture beyond inserting the MPG callsite and wiring options.
- Do **not** modify OpenPGL internals; only use its public API to get a guide summary and to push training samples.
- Do **not** add experimental features (volumes, rough microfacets, long chains). Defer with a clear comment in the README, **not** as code stubs.
- If an exact type or file path differs, **search by symbol** (e.g., “openpgl”, “guiding”, “shade_surface”, “bsdf_eval”) and adapt—**do not** leave partial edits.

---

## 9) Test plan (manual)

1. Build Blender with `WITH_CYCLES_MANIFOLD=ON` (CPU device).
2. Render the three acceptance scenes (provided or simple to construct).
3. Compare images (RMSE or eyeball) at low spp (variance) and high spp (bias check).
4. Toggle MPG on/off and move thresholds to validate **gating** behavior.

---

## 10) README note for the module

Create `intern/cycles/manifold/README.md` summarizing:
- What MPG does, v1 limitations, PDFs/MIS measure, gating thresholds.
- How to extend to 2‑bounce chains (if default is 1) and future volume support.
- How to run the acceptance scenes.
