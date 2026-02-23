# Specular Polynomials — Implementation Roadmap

## P1: Material Support

- [ ] **Glossy BSDF support** — Currently broken. The solver runs reflection constraints correctly but contribution evaluation fails for Glossy BSDF closures. Investigate IOR extraction path (`CLOSURE_IS_BSDF_GLOSSY` check) and fix the evaluation/Fresnel pipeline.

- [ ] **Principled BSDF reflection** — Most common material in Blender. Recognize Principled BSDF's specular/glossy component (`CLOSURE_BSDF_MICROFACET_GGX_ID` and similar) as a valid caustic caster. Ensure closure iteration finds the right component.

- [ ] **Conductor Fresnel for metals** — Currently uses scalar `fresnel_dielectric_cos()` for everything. Metals need conductor Fresnel (complex IOR). The reference supports named metals.

## P2: Refraction Caustics

- [ ] **Enable single-bounce refraction (T)** — `spoly_build_refraction_constraints()` is fully implemented but never called (`is_refraction = false` hardcoded). Wire up material detection to set `is_refraction = true`, extract correct IOR/eta from closure. Bezout matrix size constants already accommodate degree-6 refraction polynomials.

- [ ] **Principled BSDF transmission** — Glass-like transmission through Principled BSDF's transmission component. Detect transmission closure and route through the refraction path.

## P3: Light Types

- [ ] **Sun/distant light support** — Rejected at `ls->t == FLT_MAX` guard. Requires reformulating polynomial constraints to use a fixed direction instead of finite position. MNEE handles this via `light_fixed_direction` flag using `ls->D` and spherical coordinate Jacobians in the transfer matrix. The constraint changes from `d1 = xL - x1` to `d1 = constant_direction`.

- [ ] **Environment light support** — Same `FLT_MAX` rejection. More complex than sun lights since direction varies per sample. Potentially treat each environment sample as a directional light.

- [ ] **Mesh/triangle emitter support** — Filtered out in `shade_surface.h` (`ls.type != LIGHT_TRIANGLE`). The solver math works for finite-position lights — remove the filter and handle emissive geometry self-intersection.

## P4: Multi-Bounce

- [ ] **Bernstein bounds precomputation** — Replace interval arithmetic tree pruning with Bernstein polynomial bounds (Fan et al., SIGGRAPH 2025). Tighter pruning with quadratic convergence on subdivision. Required foundation for practical 2-bounce support. Introduces energy-aware probabilistic tuple sampling. Trade-off: adds a precomputation phase (9-32s).

- [ ] **Double-bounce reflection (RR)** — Extend solver to 4 unknowns (u1, v1, u2, v2) and larger Bezout matrices. Reference uses QZ eigenvalue decomposition for the higher-degree resultant. Requires Bernstein bounds for practical triangle-pair search.

- [ ] **Double-bounce refraction (TT)** — The classic glass caustic (refraction in + out). Reference uses piecewise rational approximation. Higher polynomial degree than RR.

- [ ] **Mixed chains (RT, TR, TRT)** — Referenced in the paper, diminishing practical importance beyond 2 bounces.

## P5: Numerical Accuracy & Solver Gaps

These are differences from the reference implementation that may affect correctness or robustness.

- [ ] **Float vs double precision** — Reference uses double precision throughout polynomial construction, Bezout matrix evaluation, and root finding. We use float32. Most impactful in: Bezout determinant evaluation (catastrophic cancellation in row reduction), high-degree polynomial evaluation (Horner's method accumulates error), and back-substitution Cxz(u, v_root) evaluation. Consider selective use of double for critical paths, or compensated summation.

- [ ] **Dichotomy sample count (129 vs 1001)** — Reference uses 1001 samples for v-root finding; we use 129. With 129 samples the step size is ~0.0078, meaning roots closer than this can be missed. For refraction (degree-6 Bezout, determinant degree up to 36), closely-spaced roots become likely. Consider increasing to 257 or 513 as a compromise between accuracy and performance.

- [ ] **Back-substitution uses bisection instead of analytical root finder** — Reference uses `cy::Polynomial::Roots()` (analytical polynomial root finder, double precision) for the u-root back-substitution step. We use `spoly_find_roots_in_01()` (bisection, float precision, 129 samples). The analytical solver is exact for the polynomial degree and cannot miss roots. Our bisection approach can miss roots with narrow basins. This is the single largest accuracy gap.

- [ ] **Newton Jacobian: finite differences vs analytical** — Reference uses automatic differentiation for exact Jacobian computation. We use finite differences with `eps = 1e-5` (3 constraint evaluations per iteration vs 1). FD is simpler but less accurate and 3x slower per Newton iteration. Consider analytical Jacobian using the half-vector constraint derivatives that are already partially computed in `spoly_compute_transfer_matrix()`.

- [ ] **Newton divergence bounds too tight** — Reference allows barycentrics to wander to ±10 during Newton iteration; we bail at ±0.05 (5%). This means solutions near triangle edges where Newton temporarily overshoots may be incorrectly rejected. Consider relaxing to ±0.5 or ±1.0 while keeping the strict post-convergence check.

- [ ] **Newton max iterations (20 vs 32)** — Reference uses 32 iterations; we use 20. Combined with our tighter divergence bounds, this gives Newton less room to converge for difficult edge cases.

- [ ] **Newton non-convergence fallback** — When Newton doesn't converge within the iteration limit, we accept solutions if the final residual < 1e-4 (line 1168). The reference strictly rejects non-converged solutions (`if (!is_find) continue`). Our fallback could accept spurious solutions. Consider removing the fallback to match the reference.

- [ ] **Newton early dedup during iteration** — Reference checks L1 distance < 0.1 against already-accepted solutions during each Newton iteration, aborting early if converging toward a known solution. We have no such check. Minor optimization that avoids wasting Newton iterations.

## P6: Code Quality & Polish

- [ ] **Double shader_setup_from_sample call** — `shader_setup_from_sample()` is called twice for the same specular point (lines 1834 and 1892): once for light evaluation, once for Fresnel/IOR extraction. These could be merged into a single call with the shader evaluation done once.

- [ ] **Spotlight cone pruning** — Spotlight works (finite position) but solver doesn't account for cone geometry. Solutions outside the cone get zero contribution via `light_sample_update()`, wasting solver budget. Prune tree based on cone intersection.

- [ ] **Flat-shaded triangle support** — Currently skips all objects without `SHADER_SMOOTH_NORMAL` (line 1585). Flat-shaded triangles can still have specular points (the normal is constant but the half-vector still varies with position). The polynomial constraint simplifies but is still solvable. Low priority since most caustic-producing objects use smooth normals.

- [ ] **Debug code cleanup** — Remove: unused `spoly_done:` label (line 2018), diagnostic negative return codes in transfer matrix (lines 1212-1215), hardcoded IOR fallback of 1.5 (line 1911), any remaining `#if 0` blocks.

- [ ] **Performance tuning** — Global solver call cap of 256 may need adjustment. Cross-triangle dedup uses linear scan. Investigate early termination heuristics and adaptive sample counts.

- [ ] **Transfer matrix for refraction** — Current `spoly_compute_transfer_matrix()` is hardcoded for reflection (`eta = 1.0f`, line 1242). Needs to accept and use the actual IOR/eta for refraction paths, matching MNEE's `mnee_compute_transfer_matrix()` which handles both cases.
