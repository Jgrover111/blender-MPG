# MPG Implementation Differences: Mitsuba Reference vs Cycles

This document lists the remaining differences between the Mitsuba reference implementation (https://github.com/mollnn/manifold-path-guiding/tree/main/mitsuba/src/integrators/MPG) and the Cycles implementation. Each item represents work needed to bring Cycles in line with the reference.

---

## TODO #1: Constraint Formulation - Add Angle-Difference Alternative

**Priority**: High Impact, Frequent

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

### Cycles (Current)
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

### What Needs to Be Done
- Implement `compute_angle_difference_constraint()` function
- Implement `compute_angle_difference_jacobian()` function
- Add runtime switch to choose between formulations
- Default to angle-difference for stability (matching Mitsuba)

### Impact
- Angle-difference formulation is more numerically stable for refractive paths
- Half-vector has singularities when `wi + eta*wo ≈ 0` (near grazing angles)
- Affects convergence on grazing angle refraction

---

## TODO #2: Use Offset Normals in Reproject

**Priority**: High Impact, Critical for Guided Paths

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

### Cycles (Current)
```cpp
// mpg_solve.cpp line 3336 - Offset normal computed
const float2 offset_2d = compute_guide_offset_normal(guide, tangent_u, tangent_v, eval.normal);

// Used in constraint:
float residual_2d_u = dot(tangent_u, h_tangent) - offset_2d.x;
float residual_2d_v = dot(tangent_v, h_tangent) - offset_2d.y;

// BUT reproject_single_bounce() lines 2765-2867:
// - Does NOT use offset normals in scattering validation
// - Computes specular direction with geometric/BSDF normal only
// - No reference to offset_2d in reproject functions
```

### What Needs to Be Done
1. Pass `offset_2d` parameter to `reproject_single_bounce()` and `reproject_double_bounce()`
2. Reconstruct 3D offset normal: `m = s*offset.x + t*offset.y + n*sqrt(1 - offset.x² - offset.y²)`
3. Pass offset normal `m` to `compute_specular()` instead of geometric normal `n`
4. Ensure scattering validation uses same offset normal as constraint

### Impact
- **CRITICAL MISMATCH**: Newton solves for vertex satisfying constraint with offset normal, but reproject validates without it
- Results in:
  - Newton converges to solution that reproject rejects
  - Infinite damping loop (beta → 0)
  - False "no valid path" failures
- Only affects guided paths (when path guiding provides offset normal)
- Pure specular paths (offset = 0) unaffected

---

## TODO #3: Use BSDF-Aware Tangent Frame

**Priority**: Medium Impact, Material-Specific

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

### Cycles (Current)
```cpp
// mpg_solve.cpp lines 661-665
// Construct arbitrary orthonormal frame from BSDF normal
make_orthonormals(eval.normal, &eval.tangent_u, &eval.tangent_v);

// make_orthonormals() creates arbitrary tangents perpendicular to normal
// Does NOT encode anisotropy, texture rotation, or material tangent space
```

### What Needs to Be Done
- Extend BSDF system to expose `get_tangent_frame()` method
- Method should return material-aware tangents that encode:
  - Anisotropic roughness orientation
  - Texture-space tangent rotation
  - Normal map tangent space
- Replace `make_orthonormals()` call with BSDF frame query
- Fallback to `make_orthonormals()` for materials without defined tangent space

### Impact
- For isotropic materials with no normal maps: No difference
- For anisotropic materials (brushed metal, etc.): Constraint uses wrong frame → convergence failure
- For materials with rotated textures/normals: Frame misalignment causes wrong solution

---

## TODO #4: Implement BSDF Frame Derivatives

**Priority**: Medium Impact, Material-Specific

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

### Cycles (Current)
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

### What Needs to Be Done
- Extend BSDF system to implement `get_frame_derivative()` method
- Method should compute full tangent-space derivatives including:
  - Smooth normal geometric curvature (already implemented)
  - Normal map texture derivatives `dN/duv`
  - Anisotropic tangent rotation from texture coordinates
  - Bump/displacement map tangent perturbation
- Integrate into `compute_halfvector_jacobian()` Jacobian computation
- Requires shader system changes to expose texture-space derivatives

### Impact
- Incomplete Jacobian when normal maps or anisotropy present
- Newton solver converges to wrong solution or diverges
- Particularly affects surfaces with:
  - Normal maps (common in production assets)
  - Anisotropic materials (brushed metal, hair, etc.)
  - Bump/displacement mapping

---

## TODO #5: Orthonormalize Parameterization

**Priority**: Low Impact, Edge Cases

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

### Cycles (Current)
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

### What Needs to Be Done
1. Implement `orthonormalize_parameterization()` function
2. Apply Gram-Schmidt to `geometry.dPdu`, `geometry.dPdv`
3. Propagate orthonormalization to `geometry.dNdu`, `geometry.dNdv` using chain rule
4. Store orthonormalized derivatives back in geometry struct
5. Return Jacobian determinant for area correction
6. Call this function after loading geometry, before Newton iterations

### Impact
- Non-orthogonal parameterization causes:
  - Anisotropic Newton steps (converges slower in one direction)
  - Incorrect step scaling (doesn't account for parametric distortion)
  - Frame derivative errors (rotating frame formula assumes orthonormal basis)
- Particularly affects:
  - UV-distorted meshes (common in game assets)
  - Cylindrical/spherical parameterizations
  - Meshes with large parametric stretch

---

## TODO #6: Remove project_barycentrics Clamping

**Priority**: Low Impact, Edge Cases

### Mitsuba
```cpp
// reproject() in newton_solver.hpp
// Computes 3D position from UNCLAMPED barycentric coordinates
Point3f p_prop = v.p - step_scale * beta * (v.dp_du * dx[0] + v.dp_dv * dx[1]);

// Then ray-traces to find actual hit
// NO clamping of barycentric coordinates at any stage
```

### Cycles (Current)
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

### What Needs to Be Done
- Remove call to `project_barycentrics()` at lines 3216-3219
- Replace with simple validity check if needed (without clamping)
- Or use barycentric coordinates directly without any projection

### Impact
- Minimal in practice (already removed hard clamps in most places)
- Only affects initial barycentric coordinates if outside [0,1]
- Could cause slight shift from intersection point for edge cases
- Less critical than previous hard clamping in Newton loop

---

## Priority Summary

### Priority 1: High Impact, Must Fix
1. **TODO #2**: Offset normals in reproject - Breaks guided paths completely
2. **TODO #1**: Angle-difference constraints - Stability issues on refractive paths

### Priority 2: Medium Impact, Material-Specific
3. **TODO #3**: BSDF-aware tangent frame - Required for anisotropic materials
4. **TODO #4**: BSDF frame derivatives - Required for normal-mapped materials

### Priority 3: Low Impact, Polish
5. **TODO #5**: Parameterization orthonormalization - Convergence rate improvement
6. **TODO #6**: Remove project_barycentrics clamp - Minor edge case cleanup

---

## Test Coverage Matrix

After implementing each TODO, verify with:

| Test Case | TODO #1 | TODO #2 | TODO #3 | TODO #4 | TODO #5 | TODO #6 |
|-----------|---------|---------|---------|---------|---------|---------|
| Simple mirror | - | - | - | - | - | - |
| Simple glass | ✓ | - | - | - | - | - |
| Curved glass | ✓ | - | - | - | ✓ | - |
| Normal-mapped glass | ✓ | - | - | **✓** | - | - |
| Anisotropic metal | - | - | **✓** | **✓** | - | - |
| Guided caustics | - | **✓** | - | - | - | - |
| Grazing refraction | **✓** | - | - | - | - | - |
| UV-distorted mesh | - | - | - | - | **✓** | **✓** |

Legend:
- **✓** = This TODO is required for this test case
- ✓ = This TODO may improve this test case
- - = This TODO has no impact on this test case

---

## Implementation Notes

### Dependencies Between TODOs

- TODO #4 depends on TODO #3 (need BSDF frame before computing its derivative)
- TODO #3 and #4 require shader system changes (may be large refactor)
- TODO #1, #2, #5, #6 can be implemented independently

### Recommended Implementation Order

1. **TODO #2** (offset normals in reproject) - Self-contained, high impact
2. **TODO #6** (remove clamp) - Trivial, cleanup
3. **TODO #1** (angle-difference) - Self-contained, moderate complexity
4. **TODO #5** (orthonormalization) - Self-contained, moderate complexity
5. **TODO #3** (BSDF frame) - Requires shader system extension
6. **TODO #4** (frame derivatives) - Requires shader system extension, depends on #3

### Estimated Complexity

- **TODO #2**: Medium (modify reproject functions, add offset normal reconstruction)
- **TODO #6**: Trivial (remove function call)
- **TODO #1**: Medium-High (new constraint formulation and Jacobian)
- **TODO #5**: Medium (implement Gram-Schmidt, propagate to derivatives)
- **TODO #3**: High (shader system API extension)
- **TODO #4**: Very High (shader system derivative computation, texture sampling)
