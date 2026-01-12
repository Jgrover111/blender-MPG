# MPG Implementation: Mitsuba Reference vs Cycles Comparison

This document details the remaining differences between the Mitsuba reference implementation (https://github.com/mollnn/manifold-path-guiding/tree/main/mitsuba/src/integrators/MPG) and the Cycles implementation after multiple rounds of fixes.

## Status Summary

**Fixed Issues** (implemented in Cycles):
- ✅ Stale seed prim/object after triangle walk
- ✅ Stale BSDF backfacing flag after triangle walk
- ✅ Direct 2×2 solver (avoiding normal equations)
- ✅ Tangent-frame derivatives in Jacobian
- ✅ Guided offset normals in constraint computation
- ✅ Post-convergence using current geometry/seeds
- ✅ Removed hard clamping of barycentric coordinates
- ✅ Relaxed Jacobian zero threshold (1e-30)
- ✅ Double-bounce Jacobian using current seeds

**Remaining Differences** (architectural or unimplemented):

---

## 1. Constraint Formulation: Half-Vector vs Angle-Difference

### Mitsuba
```cpp
// newton_solver.hpp line ~150
bool halfvector_constraints = false;  // DEFAULTS TO FALSE

if (halfvector_constraints) {
    // Half-vector formulation: C = H - N
    // where H = project(normalize(wi + eta*wo), tangent_frame)
} else {
    // Angle-difference formulation (DEFAULT):
    // C = [theta_d - theta_h, 0]
    // More stable for refraction, avoids singularities
}
```

### Cycles
```cpp
// mpg_solve.cpp lines 3268-3336
// ONLY implements half-vector formulation
// Constraint: C = [dot(s, h), dot(t, h)] where h = normalize(wi + eta*wo)

float3 h = wi + h_eta * wo;
if (params.is_refraction) {
    h = -h;
}
h = normalize(h);

// Project onto tangent frame
float residual_2d_u = dot(tangent_u, h_tangent) - offset_2d.x;
float residual_2d_v = dot(tangent_v, h_tangent) - offset_2d.y;
```

**Impact**:
- Angle-difference formulation is more numerically stable for refractive paths
- Half-vector can have singularities when `wi + eta*wo ≈ 0` (near grazing angles)
- Mitsuba defaults to angle-difference for this reason
- Cycles only has half-vector, which may explain remaining convergence issues

**To fix**: Would require implementing `compute_step_angle_difference()` alongside `compute_halfvector_jacobian()` and adding a runtime switch.

---

## 2. BSDF Frame vs make_orthonormals

### Mitsuba
```cpp
// compute_step_halfvector() in newton_solver.hpp
Frame3f frame = si.bsdf()->frame(si, smoothing);
Vector3f s = frame.s;  // BSDF tangent (encodes anisotropy)
Vector3f t = frame.t;  // BSDF bitangent
Vector3f n = frame.n;  // BSDF normal (may differ from geometric)

// Frame encodes:
// - Anisotropic roughness orientation
// - Texture-space tangent rotation
// - Normal map tangent rotation
// - Material-specific tangent conventions
```

### Cycles
```cpp
// mpg_solve.cpp lines 661-665
// Construct arbitrary orthonormal frame from BSDF normal
make_orthonormals(eval.normal, &eval.tangent_u, &eval.tangent_v);

// make_orthonormals() creates arbitrary tangents perpendicular to normal
// Does NOT encode anisotropy, texture rotation, or material tangent space
```

**Impact**:
- For isotropic materials with no normal maps: No difference
- For anisotropic materials (brushed metal, etc.): Constraint uses wrong frame
- For materials with rotated textures/normals: Frame misalignment causes convergence issues
- Cycles approximation works for simple cases but breaks for complex materials

**To fix**: Would require BSDF system to expose `get_tangent_frame()` method that returns material-aware tangents.

---

## 3. BSDF Frame Derivatives

### Mitsuba
```cpp
// compute_step_halfvector() in newton_solver.hpp
auto [ds_du, dt_du] = si.bsdf()->frame_derivative(si, smoothing, true);
auto [ds_dv, dt_dv] = si.bsdf()->frame_derivative(si, smoothing, false);

// Includes ALL sources of frame variation:
// - Smooth normal derivatives (geometric curvature)
// - Normal map derivatives (dN/duv from texture)
// - Anisotropic tangent rotation (texture-space rotation)
// - Bump map effects on frame orientation
```

### Cycles
```cpp
// mpg_solve.cpp lines 1555-1575, 1647-1658
// ONLY computes derivatives from smooth normal variation
compute_tangent_frame_derivatives(eval.normal, tangent_u, tangent_v,
                                 eval.dNdu, eval.dNdv,
                                 ds_du, ds_dv, dt_du, dt_dv);

// Uses rotating frame formula: ds/dx = (N × dN/dx) × s
// DOES NOT include:
// - Normal map texture derivatives
// - Anisotropic tangent rotation
// - Bump map tangent perturbation
```

**Impact**:
- Incomplete Jacobian when normal maps or anisotropy present
- Newton solver converges to wrong solution or diverges
- Particularly affects surfaces with:
  - Normal maps (common in production assets)
  - Anisotropic materials (brushed metal, hair, etc.)
  - Bump/displacement mapping

**To fix**: Would require `bsdf()->frame_derivative()` method in shader system that computes full tangent-space derivatives including texture effects.

---

## 4. Parameterization Orthonormalization

### Mitsuba
```cpp
// make_orthonormal() in newton_solver.hpp lines ~50-75
Spectrum make_orthonormal(SurfaceInteraction3f &si, bool smoothing) {
    // Get parametric derivatives from mesh
    Vector3f dp_du = si.dp_du;
    Vector3f dp_dv = si.dp_dv;

    // Gram-Schmidt orthonormalization
    Vector3f s = normalize(dp_du);
    Vector3f t = normalize(dp_dv - s * dot(s, dp_dv));
    Vector3f n = cross(s, t);

    // Store orthonormalized frame back into si
    si.dp_du = s;
    si.dp_dv = t;
    si.n = n;

    // Compute frame derivatives (propagate to dN_du, dN_dv)
    // Returns Jacobian determinant for area correction
    return det;
}
```

### Cycles
```cpp
// mpg_solve.cpp - NO equivalent function
// Uses raw mesh parametric derivatives directly
geometry.dPdu  // Raw from mesh, NOT orthonormalized
geometry.dPdv  // Raw from mesh, NOT orthonormalized

// These are used directly in:
// - derivative_normalized() for direction derivatives
// - compute_tangent_frame_derivatives() for frame derivatives
// - Jacobian computation
```

**Impact**:
- Non-orthogonal parameterization can cause:
  - Anisotropic Newton steps (converges slower in one direction)
  - Incorrect step scaling (doesn't account for parametric distortion)
  - Frame derivative errors (rotating frame formula assumes orthonormal basis)
- Particularly affects:
  - UV-distorted meshes (common in game assets)
  - Cylindrical/spherical parameterizations
  - Meshes with large parametric stretch

**To fix**: Implement `orthonormalize_parameterization()` function that:
1. Applies Gram-Schmidt to `dPdu`, `dPdv`
2. Propagates to `dNdu`, `dNdv` using chain rule
3. Stores orthonormalized derivatives in geometry struct
4. Returns Jacobian determinant for area correction

---

## 5. Offset Normals in Reproject and Scattering

### Mitsuba
```cpp
// reproject() in newton_solver.hpp
Vector3f m = vertex.s * n_offset[0] +   // s component of offset
             vertex.t * n_offset[1] +   // t component of offset
             vertex.n * n_offset[2];    // n component of offset (usually 1.0)

// Scattered direction computed using offset normal 'm' instead of 'n'
Vector3f wo = bsdf->sample(..., m, ...);  // Uses offset normal

// Validates that scattering with offset normal reaches expected target
```

### Cycles
```cpp
// mpg_solve.cpp lines 3336 - Offset normal computed
const float2 offset_2d = compute_guide_offset_normal(guide, tangent_u, tangent_v, eval.normal);

// Used in constraint:
float residual_2d_u = dot(tangent_u, h_tangent) - offset_2d.x;
float residual_2d_v = dot(tangent_v, h_tangent) - offset_2d.y;

// BUT reproject_single_bounce() lines 2765-2867:
// - Does NOT use offset normals in scattering validation
// - Computes specular direction with geometric/BSDF normal only
// - No reference to offset_2d in reproject functions
```

**Impact**:
- **Fundamental mismatch**: Newton solves for vertex that satisfies constraint with offset normal, but reproject validates without it
- Results in:
  - Newton converges to solution that reproject rejects
  - Infinite damping loop (beta → 0)
  - False "no valid path" failures
- Only affects guided paths (when path guiding provides offset normal)
- Pure specular paths (offset = 0) unaffected

**To fix**:
1. Pass `offset_2d` to reproject functions
2. Reconstruct 3D offset normal: `m = s*offset.x + t*offset.y + n*sqrt(1 - offset.x² - offset.y²)`
3. Use `m` instead of `n` in `compute_specular()` during reproject validation

---

## 6. project_barycentrics Still Clamps

### Mitsuba
```cpp
// reproject() in newton_solver.hpp
// Computes 3D position from UNCLAMPED barycentric coordinates
Point3f p_prop = v.p - step_scale * beta * (v.dp_du * dx[0] + v.dp_dv * dx[1]);

// Then ray-traces to find actual hit
// NO clamping of barycentric coordinates at any stage
```

### Cycles
```cpp
// mpg_solve.cpp lines 3216-3219
float u = seed.bary_u;  // No hard clamp (GOOD)
float v = seed.bary_v;
project_barycentrics(u, v);  // But this function still clamps

// project_barycentrics() implementation (in another file):
void project_barycentrics(float &u, float &v) {
    // Clamps negative values to 0 and renormalizes
    if (u < 0.0f) u = 0.0f;
    if (v < 0.0f) v = 0.0f;
    if (u + v > 1.0f) {
        float sum = u + v;
        u /= sum;
        v /= sum;
    }
}
```

**Impact**:
- Minimal in practice (already removed hard clamps in most places)
- Only affects initial barycentric coordinates if outside [0,1]
- Could cause slight shift from intersection point for edge cases
- Less critical than previous hard clamping in Newton loop

**To fix**: Remove call to `project_barycentrics()` or replace with simple validity check without clamping.

---

## 7. Stale BSDF Parameters Beyond Backfacing

### Current Status
```cpp
// mpg_solve.cpp lines 3036-3041 (double-bounce)
if (hit_primary_prim != expected_primary_prim) {
    // Recompute backfacing
    primary_params.backfacing = (dot(actual_gn, ray_to_primary) < 0.0f);

    // Invalidate stale BSDF normal
    if (primary_params.has_normal) {
        primary_params.has_normal = false;
    }
}

// But SpecularParameters contains many other fields:
// - base_eta, medium_eta (from shader evaluation)
// - microfacet alpha_x, alpha_y (roughness)
// - has_microfacet flag
// All derived from shader evaluation on OLD triangle
```

**Impact**:
- When Newton walks to adjacent triangle with different shader:
  - Old shader's eta used with new geometry
  - Old roughness values used
  - Wrong material properties in constraint/Jacobian
- Cycles currently only fixes `backfacing` and `has_normal`
- Full `SpecularParameters` should be recomputed after triangle walk

**Current mitigation**: Lines 3432-3437 (single-bounce) and 4018-4028 (double-bounce) call `specular_parameters_from_surface()` with NEW seed, which recomputes all parameters correctly.

**Status**: ✅ Actually FIXED - Full parameter reload happens via `specular_parameters_from_surface(kg, sd, new_geometry, current_seed, ...)` calls. The lines 3036-3041 are just an optimization for early backfacing check before full reload.

---

## 8. Additional Differences Found

### 8.1 Solver Threshold and Max Iterations

**Mitsuba**:
```cpp
float solver_threshold = 1e-4f;  // Same as Cycles
int max_solver_iterations = 20;   // Default
```

**Cycles**:
```cpp
// Lines 3366, 3902
if (residual_norm < 1e-4f) { break; }  // Same threshold ✅
// max_iters from options (usually 20) ✅
```

**Status**: ✅ **SAME** - Both use 1e-4 threshold and ~20 iterations

### 8.2 Step Scaling Strategy

**Mitsuba**:
```cpp
float step_scale = 1.0f;  // Unit steps
float beta = 1.0f;        // Damping factor
// Step: p_new = p - step_scale * beta * (dp_du * dx[0] + dp_dv * dx[1])
```

**Cycles**:
```cpp
// Lines 3394-3395, 3954-3957
float proposed_u = u - options.step_scale * beta * delta.x;
// options.step_scale typically 1.0
float beta = 1.0f;  // Initialized, halved on rejection
```

**Status**: ✅ **SAME** - Both use `step_scale * beta` damping

### 8.3 Beta Damping on Rejection

**Mitsuba**:
```cpp
if (!reproject_success) {
    beta *= 0.5f;
    needs_jacobian_recompute = false;
    continue;
}
```

**Cycles**:
```cpp
// Lines 3407-3410, 3984-3986
if (!reproject_single_bounce(...)) {
    beta *= 0.5f;
    needs_step_update = false;  // Reuse Jacobian ✅
    continue;
}
```

**Status**: ✅ **SAME** - Both halve beta and reuse Jacobian

### 8.4 Convergence Check Location

**Mitsuba**:
```cpp
// Checks convergence at TOP of loop (after residual update)
for (int iter = 0; iter < max_iters; ++iter) {
    if (residual < threshold) break;  // Check first
    // ... compute step ...
}
```

**Cycles**:
```cpp
// Lines 3366-3368, 3901-3906
for (int iter = 0; iter < options.max_iters; ++iter) {
    if (residual_norm < 1e-4f) { break; }  // Check first ✅
    // ... compute step ...
}
```

**Status**: ✅ **SAME** - Both check convergence before computing step

### 8.5 Jacobian Zero Handling

**Mitsuba**:
```cpp
// Always normalizes and proceeds
// Relies on condition number checks in solver
// No explicit "Jacobian is zero" threshold
```

**Cycles**:
```cpp
// Lines 1627-1632
if (!(g_len > 1e-30f) || !isfinite_safe(g_len)) {
    J[0] = make_float3(0.0f, 0.0f, 0.0f);
    J[1] = make_float3(0.0f, 0.0f, 0.0f);
    return;  // Returns zero Jacobian
}
```

**Status**: ⚠️ **DIFFERENCE** - Cycles has explicit threshold, Mitsuba always normalizes. However, 1e-30 is so small it's effectively the same (only catches true degeneracy).

### 8.6 Triangle Walk Shape Consistency Check

**Mitsuba**:
```cpp
// reproject() checks si.shape == expected_shape
// Shape is the mesh object
if (si.shape != v.si.shape) {
    return false;  // Reject step - walked to different mesh
}
```

**Cycles**:
```cpp
// Lines 2838-2845
if (isect.object != expected_object) {
    return false;  // Reject - different mesh object
}
// If same object but different prim: ALLOW (triangle walk)
```

**Status**: ✅ **SAME** - Both check object/shape, allow triangle walks within same object

---

## 9. Critical Issues (Priority Order)

### Priority 1: High Impact, Frequent
1. **Half-vector vs angle-difference constraints** - Stability issues on refractive paths
2. **Offset normals not in reproject** - Breaks guided paths, mismatch between constraint and validation

### Priority 2: Medium Impact, Material-Specific
3. **BSDF frame vs make_orthonormals** - Wrong for anisotropic materials
4. **Missing BSDF frame derivatives** - Wrong Jacobian with normal maps, anisotropy

### Priority 3: Low Impact, Edge Cases
5. **Parameterization non-orthogonal** - Slower convergence on distorted UVs
6. **project_barycentrics clamps** - Minor shift at triangle edges

---

## 10. Verification Checklist

For each difference above, tested scenarios:

- ✅ Simple mirror (planar, isotropic) - Works in both
- ✅ Simple glass (planar, isotropic) - Works in both
- ⚠️ Curved glass (smooth normals) - Works after tangent-frame derivative fix
- ❌ Normal-mapped glass - Needs BSDF frame derivatives (Issue #4)
- ❌ Anisotropic metal - Needs BSDF frame (Issue #3)
- ❌ Guided caustics - Needs offset normals in reproject (Issue #2)
- ⚠️ Grazing angle refraction - May need angle-difference (Issue #1)

---

## 11. Recommended Fix Order

1. **Offset normals in reproject** (Issue #2, #5) - Highest impact for guided paths
2. **Angle-difference constraint** (Issue #1) - Improves refraction stability
3. **BSDF frame access** (Issue #3) - Enables anisotropic materials
4. **BSDF frame derivatives** (Issue #4) - Enables normal-mapped materials
5. **Parameterization orthonormalization** (Issue #7) - Polish, improves convergence rate
6. **Remove project_barycentrics clamp** (Issue #6) - Minor cleanup

---

## 12. Summary

**Fixed in Cycles** ✅:
- Stale seeds after triangle walk (critical)
- Direct 2×2 solver (condition number)
- Tangent-frame derivatives (smooth normals)
- Guided offset normals in constraint
- Post-convergence validation fixes
- Barycentric clamping removal
- Jacobian threshold tuning

**Remaining Architectural Differences**:
- Constraint formulation (half-vector only, missing angle-difference)
- BSDF frame (arbitrary frame vs material-aware frame)
- BSDF frame derivatives (missing texture/anisotropy derivatives)
- Parameterization orthonormalization (missing Gram-Schmidt)
- Offset normals in reproject (mismatch between constraint and validation)

**Estimated Impact**:
- Current implementation: Works for simple isotropic speculars (mirrors, simple glass)
- Remaining issues affect: Anisotropy, normal maps, guided paths, extreme refraction angles
- Most critical remaining issue: Offset normals in reproject (breaks guided paths)
