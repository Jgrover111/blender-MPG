# AGENTS.md — Cycles CPU Manifold Path Guiding (MPG)

**Purpose**
Make the Cycles CPU MPG implementation **match the Mitsuba reference logic** (repo: `extern/mpg_ref`) for chains up to **2 specular bounces**: `D → S → L` and `D → S → S → L`. Avoid placeholders; align PDFs, measures, order-of-operations, and MIS exactly.

---

## 0) Non‑negotiables

* **CPU only.** Do not touch CUDA/HIP/Metal.
* **No volumes** in v1; skip if inside/crossing media.
* **No placeholders.** Every path fully implemented or feature gated off.
* **Unbiasedness preserved.** MPG is an *additional* proposal, combined by MIS with existing proposals (BSDF, NEE, OpenPGL) **in the same measure**.
* **Minimal blast radius.** Keep all new code in `intern/cycles/manifold/`. Only surgical touches in integrator/UI/build.
* **No OpenPGL internals.** Only public `sample()`/`pdf()` for the summary estimator.

---

## 1) Goal (single sentence)

Implement MPG for **1–2 specular bounces** at surface hits in Cycles and MIS‑combine its contribution with BSDF + NEE + OpenPGL, with OpenPGL‑driven gating.

---

## 2) Scope (v1)

* **Chains:** `D→S→L` and `D→S→S→L`.
* **Materials:** ideal reflection/refraction on meshes (ignore bump/normal map perturbs in the solver).
* **Lights:** area, sun/distant, environment (sun‑disc OK).
* **Where:** CPU surface integrator only.
* **OpenPGL usage:** sampling‑based **GuideSummary** at the hit (dominant dir, peak mass in cone, κ estimate) for *gating & seeding*.
* **Training:** optionally feed successful MPG samples back to OpenPGL via existing API.

Out of scope: volumes, rough microfacets, GPU, longer chains, SMS/VCM/VM.

---

## 3) New module & public API

Create `intern/cycles/manifold/` (names may already exist; update in place):

* `mpg.h/.cpp` – high‑level **try\_connect** entry (returns direction, visibility, pdf).
* `mpg_solve.h/.cpp` – Newton/trust‑region solver for `S` and `S+S` constraints (Snell/reflect). Output includes **J\_total** (geometric Jacobian) and shadowing.
* `mpg_pdf.h` – helper to compose final technique pdf (if not done inline in `mpg.cpp`).
* `mpg_seed.h/.cpp` – seed direction/params from **GuideSummary**; returns **seed\_pdf** in its own native measure.
* `mpg_pgl_summary.h/.cpp` – sampling‑based OpenPGL **GuideSummary** estimator (dominant dir, peak mass, κ).

### API sketch

```cpp
struct MpgOptions {
  int   max_bounces = 2;        // 1 or 2
  int   max_iters   = 6;
  float gate_w      = 0.35f;    // peak mass threshold [0..1]
  float gate_kappa  = 40.0f;    // vMF concentration threshold
  float angular_jitter = 0.02f; // rad; clamp >= 0.0087 (~0.5°)
};

struct GuideSummary { float3 mean_dir; float peak_weight, kappa, rbar; };

struct MpgResult { bool success; float3 wi; float pdf; float visibility; };

MpgResult mpg_try_connect(/* shading point D, bsdf, lights, */
                          const GuideSummary& g,
                          const MpgOptions& opt,
                          RNG& rng);

bool pgl_estimate_summary(/* OpenPGL dist at D, Ng, */ RNG&, GuideSummary&,
                          int n = 128, float cone = radians(10.0f));
```

---

## 4) **Parity rules with Mitsuba** (critical)

**Mirror the reference in these five aspects:**

### A. **Order of operations** (must match)

1. Build **seed** (dir/params) and **seed\_pdf** using GuideSummary.
2. Sample/choose light endpoint (record **pdf\_selection**); get light sample in *its native measure*.
3. **Solve** manifold for the chain (`S` or `S+S`), producing the specular points & **J\_total** and a **shadow term**.
4. **Update the light sample to the solved specular point** (position/normal) via the usual light update function.
5. **Compose the technique PDF** (see C below) *after* the update.
6. Evaluate BSDF & light emission, do visibility, then **MIS & accumulate**.

### B. **Measures** (must match)

* All MIS PDFs compared in **solid angle at the receiver D**.
* `ls.pdf` *after* `light_sample_update()` is solid angle **at the specular point**. Do **not** re‑introduce cos/d² factors.
* Multiply by **J\_total** to convert through the specular chain to **receiver solid angle**.
* The **NEE pdf** in the MIS denominator is the *standard* NEE pdf at **D** (pre‑MPG), **not** the updated light pdf.

### C. **Technique PDF formula** (identical to reference)

For either chain (`S` or `S+S`):

```
// Solid-angle pdf at D for the MPG proposal
p_mpg(D, ω) = p_seed * p_light(spec_pt) * J_total
```

Where:

* `p_seed` = seed sampler pdf (cone/area) — clamp to `[1e-16, +inf)`.
* `p_light(spec_pt)` = light solid‑angle pdf **after** moving the light sample to the solved specular point (restore `pdf_selection` if it was factored out).
* `J_total` = solver’s change‑of‑variables Jacobian for the solved chain (product over specular surfaces). **Apply exactly once.**

**Never** multiply cosines or distance² here; they are already part of the light sampling measure and BSDF evaluation.

### D. **MIS weight** (3–4 way balance heuristic)

Compare proposals present at this bounce (all solid angle @ D):

* `p_bsdf` (BSDF at D toward `wi`),
* `p_guided` (OpenPGL guided BSDF, if active),
* `p_nee` (standard NEE at D),
* `p_mpg` (as above).

```
w_mpg = p_mpg / (p_bsdf + p_guided + p_nee + p_mpg)
```

Use **balance** heuristic (β=1). Do not mix updated‑light pdf in `p_nee`.

### E. **Accumulation** (same plumbing as NEE)

```
contrib = throughput * bsdf_eval(D, wi) * light_eval(updated_ls) * visibility
contrib *= w_mpg / p_mpg;
accumulate_light(contrib);
```

---

## 5) Chain specifics

* **D→S→L:** solver returns one specular point; `J_total = J_S`.
* **D→S→S→L:** two specular points; `J_total = J_S1 * J_S2` from the solver. No extra geometric terms; the BSDFs at D are evaluated as usual.
* Use **receiver non‑singular labels** (do not mark D as delta). The specular surfaces in the chain are delta constraints handled by the solver/Jacobian.

---

## 6) Gating & seeding (OpenPGL‑driven)

* Compute `GuideSummary` via `pgl_estimate_summary()` using public OpenPGL sampling/pdf.
* **Gate:** skip MPG unless `peak_weight ≥ gate_w` **and** `kappa ≥ gate_kappa` **and** BSDF at D is non‑delta.
* **Seeding:** jitter within a cone around `mean_dir` (half‑angle = `max(angular_jitter, 0.0087f)`). Compute `p_seed = 1 / (2π (1 − cos θmax))` (extend with area terms if you seed more parameters). Ensure `p_seed ≥ 1e-16`.

---

## 7) Integrator hook (CPU)

At the surface shading site (same place as OpenPGL & NEE):

1. Build `GuideSummary`. If gate fails, **bail early**.
2. Call `mpg_try_connect(...)` (attempt once per bounce). If success:

    * **Do not** alter other proposals; MPG is additive.
    * Build `p_bsdf`, `p_guided` (if used), `p_nee` (pre‑MPG) and use `r.pdf` as `p_mpg`.
    * Compute `w_mpg` with balance heuristic, then accumulate as in §4E.
3. Optionally push training sample to OpenPGL.

---

## 8) Build & UI

* CMake option: `WITH_CYCLES_MANIFOLD` (CPU target only). Guard kernel/UI code.
* Scene options (defaults):

    * `manifold_guiding_enable=false`
    * `manifold_max_bounces=2`
    * `manifold_iters=6`
    * `manifold_gate_weight=0.35`
    * `manifold_gate_kappa=40.0`
    * (internal) `manifold_summary_samples=128`, `manifold_summary_cone_deg=10`
* UI: Sampling ▸ Guiding ▸ **Manifold Path Guiding (CPU, Experimental)**.

---

## 9) Reference mapping (for agents to compare)

Create `docs/MPG_COMPARE.yml` to map our functions to reference ones and to drive checks:

```yaml
pairs:
  - ours: intern/cycles/manifold/mpg_seed.cpp::mpg_build_seed
    ref:  extern/mpg_ref/src/.../seed.cpp::build_seed
  - ours: intern/cycles/manifold/mpg_solve.cpp::mpg_solve_connect
    ref:  extern/mpg_ref/src/.../solve.cpp::connect
  - ours: intern/cycles/manifold/mpg.cpp::compose_pdf_after_update
    ref:  extern/mpg_ref/src/.../integrator.cpp::compose_pdf
  - ours: intern/cycles/kernel/integrator/shade_surface.h::MPG block
    ref:  extern/mpg_ref/src/.../integrator.cpp::direct_mpg_sample
checklist:
  - Light pdf evaluated **after** moving light sample? (Yes)
  - `p_mpg = p_seed * p_light(spec_pt) * J_total` only once? (Yes)
  - MIS denominator uses **NEE at D**, not updated light pdf? (Yes)
  - All pdfs in **solid angle @ D**? (Yes)
  - No extra cos/d² or double Jacobian? (Yes)
```

---

## 10) Acceptance criteria

**Functional**

* Cornell slab caustic: MPG ON shows caustic at 64 spp; at 16k spp matches baseline within 1% RMSE (unbiased).
* Env sun caustic: fewer bright speckles at equal spp.
* Non‑caustic: MPG gated off → images/time \~ unchanged.

**Engineering**

* Builds on Win/Linux CPU; no new warnings; no dead code.

**Safety**

* If solve fails or `pdf≤0`, skip MPG silently.
* Clamp/validate all pdfs; guard against NaN/Inf.

---

## 11) Debug checklist (when images look wrong)

* Log once/frame: `p_seed`, `p_light(updated)`, `J_total`, `p_mpg`, `p_bsdf`, `p_nee`, `w_mpg`.
* If rare bright pixels → `p_mpg` too small or wrong measure; re‑check §4B–C and MIS denominator source.
* If MPG has no effect → block never accumulates, or `p_mpg` dwarfed by others (gating/MIS).
