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

## P5: Code Quality & Polish

- [ ] **Spotlight cone pruning** — Spotlight works (finite position) but solver doesn't account for cone geometry. Solutions outside the cone get zero contribution via `light_sample_update()`, wasting solver budget. Prune tree based on cone intersection.

- [ ] **Debug code cleanup** — Remove: unused `spoly_done:` label, diagnostic negative return codes in transfer matrix, hardcoded IOR fallback of 1.5, any remaining `#if 0` blocks.

- [ ] **Performance tuning** — Global solver call cap of 256 may need adjustment. Cross-triangle dedup uses linear scan. Investigate early termination heuristics and adaptive sample counts.
