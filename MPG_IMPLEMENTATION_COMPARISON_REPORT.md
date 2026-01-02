# Manifold Path Guiding Implementation Comparison Report

**Date**: 2026-01-02
**Comparing**: Blender MPG (mpg-cpu branch) vs Mitsuba Reference Implementation
**Repository**: https://github.com/mollnn/manifold-path-guiding/tree/main/mitsuba/src/integrators/MPG

---

## Executive Summary

This report documents a detailed comparison between the Blender Cycles Manifold Path Guiding implementation and the Mitsuba reference implementation. **Several critical issues and bugs were identified** that affect the correctness of the MPG algorithm.

---

## Critical Issues Found

### 🔴 **CRITICAL BUG #1: Incorrect Seed Lobe Classification**

**Location**: `intern/cycles/manifold/mpg_seed.cpp:391-401`

**Issue**: The `classify_seed_lobe()` function incorrectly returns `SeedLobe::Dual` for ALL non-transmission closures, including pure reflective materials.

```cpp
const auto classify_seed_lobe = [](const ShaderClosure &closure) {
  if (CLOSURE_IS_BSDF_TRANSPARENT(closure.type) ||
      CLOSURE_IS_BSDF_TRANSMISSION(closure.type))
  {
    return SeedLobe::Transmission;
  }
  if (CLOSURE_IS_GLASS(closure.type)) {
    return SeedLobe::Dual;
  }
  return SeedLobe::Dual;  // ❌ WRONG: Should check for reflection-only
};
```

**Expected Behavior** (based on Mitsuba reference):
- Pure **reflective** materials (mirrors, glossy) should return `SeedLobe::Reflection`
- **Glass/dielectric** materials should return `SeedLobe::Dual`
- **Transmission-only** materials should return `SeedLobe::Transmission`

**Impact**:
- Causes incorrect reflection/refraction probability calculations for pure reflective materials
- May waste samples attempting refraction on reflection-only surfaces
- Affects seed generation efficiency and correctness

**Fix Required**:
```cpp
return SeedLobe::Dual;  // Should be: SeedLobe::Reflection
```

Add proper checking for `CLOSURE_IS_BSDF_REFLECTION` or similar reflection-only classification.

---

### 🟡 **ISSUE #2: Overly Complex Reflection Probability Logic**

**Location**: `intern/cycles/manifold/mpg_seed.cpp:458-559`

**Issue**: The Blender implementation has significantly more complex logic for determining reflection vs refraction probability compared to the Mitsuba reference.

**Mitsuba Reference** (simple and direct):
```cpp
auto [F, unused_0, unused_1, unused_2] = fresnel(cos_theta, eta);
reflect_prob = avg(F);  // Simple Fresnel average

if (sampler->next_1d() < reflect_prob) {
    // Reflection
} else {
    // Refraction
}
```

**Blender Implementation** (complex):
- Uses `prefer_transmission_from_guide` logic
- Has multiple fallback paths (`prefer_transmission_decided`)
- Special cases for different seed lobes
- Computes Fresnel through `mpg_compute_dielectric_reflection_probability()`
- Has fallback to 0.25/0.75 probabilities when Fresnel fails

**Analysis**:
While the Blender implementation attempts to be more sophisticated by considering guidance information, it may introduce unnecessary complexity that deviates from the proven reference implementation. The multiple code paths increase the risk of bugs.

**Recommendation**: Simplify to match the Mitsuba reference unless there's strong justification for the added complexity.

---

### 🟡 **ISSUE #3: Bounce Count Distribution Differences**

**Location**: `intern/cycles/manifold/mpg_seed.cpp:688-741`

**Mitsuba Reference**:
```cpp
float acc = 1.0f;
for (int i = 0; i < MAX_CHAIN_LENGTH; i++) {
    a[i] = acc;
    if (i + 1 > m_sms_rr_depth) {
        acc *= 0.95;  // Hard-coded survival probability
    }
}
```

**Blender Implementation**:
```cpp
float geom_single_weight = 1.0f;
float geom_double_weight = allow_double_bounce ? 1.0f : 0.0f;
// ... complex mixing with guided_pdf_single/double
const float bounce_alpha = guided_distribution_available ? 0.5f : 1.0f;
pdf_single = bounce_alpha * geom_pdf_single + (1.0f - bounce_alpha) * guided_pdf_single;
pdf_double = bounce_alpha * geom_pdf_double + (1.0f - bounce_alpha) * guided_pdf_double;
```

**Analysis**:
- Mitsuba uses a simple geometric distribution with Russian roulette
- Blender uses a mixture distribution combining geometric and guided components
- Blender is limited to `max_bounces=2` (single/double), while Mitsuba supports up to `MAX_CHAIN_LENGTH=10`
- The `bounce_alpha = 0.5` mixing factor is arbitrary and not present in reference

**Potential Issues**:
- The 50/50 mixture may not be optimal
- Limited to 2 bounces reduces algorithm capability
- Different probability distribution may affect convergence

---

### 🟢 **VERIFIED CORRECT: Tau Bit Encoding**

**Location**: `intern/cycles/manifold/mpg_seed.cpp:32-47` and `mpg_solve.cpp:80-95`

**Status**: ✅ Correct implementation matching Mitsuba reference

```cpp
ccl_device_forceinline void set_chaintype_bit(uint8_t &tau, int position, bool is_refraction)
{
  tau &= ~(1u << position);
  if (is_refraction) {
    tau |= (1u << position);
  }
}

ccl_device_forceinline bool get_chaintype_bit(uint8_t tau, int position)
{
  return ((tau >> position) & 1u) != 0u;
}
```

This correctly implements the Mitsuba reference bit manipulation for encoding reflection/refraction chain types.

---

## Detailed Component Comparison

### 1. Seed Generation

| Component | Mitsuba Reference | Blender Implementation | Status |
|-----------|------------------|----------------------|--------|
| Direction sampling | Guided distribution or uniform sphere | Cone-based sampling with rejection | ⚠️ Different approach |
| Lobe classification | Reflection/Transmission/Dual | **Always Dual for reflections** | ❌ **BUG** |
| Fresnel probability | Simple `avg(F)` | Complex with fallbacks | ⚠️ Overly complex |
| Bounce distribution | Geometric with RR (0.95) | Mixture of geometric + guided | ⚠️ Different |
| Hemisphere constraints | Dot product > 0 check | Complex cone acceptance calculations | ⚠️ Different |

### 2. Newton Solver

| Component | Mitsuba Reference | Blender Implementation | Status |
|-----------|------------------|----------------------|--------|
| Constraint formulation | Half-vector constraints | Half-vector constraints | ✅ Same |
| Iteration limit | Not specified | 20 iterations (`max_iters`) | ✅ Reasonable |
| Convergence threshold | `solver_threshold = 1e-4` | `1e-4` | ✅ Same |
| Update strategy | Newton steps | Newton steps with damping | ✅ Good |

### 3. Jacobian Calculation

**Location**: `intern/cycles/manifold/mpg_solve.cpp:1292-1333, 1338-1391, 1816-2009`

**Formula** (single bounce):
```cpp
const float geometric_factor = cos_theta / distance_sq;
const float jacobian = fabsf(determinant) * geometric_factor;
```

**Formula** (double bounce):
```cpp
const float jacobian_total = jacobian_primary * jacobian_secondary;
```

**Status**: ✅ Appears correct (product of individual Jacobians)

**Notes**:
- Uses determinant of constraint Jacobian matrix
- Multiplies by geometric correction factor
- For double bounce, multiplies individual Jacobians
- Includes proper checks for numerical stability

### 4. PDF Calculations

**Location**: `intern/cycles/manifold/mpg_pdf.cpp:77-152`

**Formula**:
```cpp
pdf = p_seed * p_light * J;
```

Where:
- `p_seed` = seed sampling probability (direction × branch × scatter × bounce)
- `p_light` = light sampling probability (solid angle measure)
- `J` = Jacobian determinant (absolute value)

**Status**: ✅ Matches expected MPG formula

**Notes**:
- Properly rebuilds seed PDF from components
- Applies acceptance probability normalization (`inv_expected_trials`)
- Folds bounce PDF into seed PDF

---

## Minor Issues and Observations

### 5. Visibility Testing

**Location**: `intern/cycles/manifold/mpg.cpp:77-139`

- Blender skips visibility tests for directional lights (`light_sample.t != FLT_MAX`)
- Uses `MPG_DISTANT_LIGHT_VISIBILITY_DISTANCE = 1.0e6f` for distant lights
- Special handling for single-bounce refraction through closed geometry

**Status**: ✅ Reasonable implementation with practical considerations

### 6. Numerical Constants

| Constant | Mitsuba | Blender | Notes |
|----------|---------|---------|-------|
| Max iterations | Not specified | 20 | Reasonable |
| Convergence threshold | `1e-4` | `1e-4` | ✅ Same |
| RR survival probability | `0.95` | N/A (limited to 2 bounces) | Different approach |
| Min Jacobian | Not specified | `1.0e-12f` | Conservative |
| Min cone angle | Not specified | `0.00872664626` (~0.5°) | Conservative |

---

## Recommendations

### Priority 1 - Critical Fixes Required

1. **Fix seed lobe classification bug** (Issue #1)
   - Properly classify reflection-only vs dual vs transmission-only materials
   - Test with pure mirror surfaces to verify correct behavior

2. **Simplify reflection probability logic** (Issue #2)
   - Consider matching Mitsuba's simple Fresnel approach
   - Remove unnecessary fallback paths unless justified
   - Document why complexity is needed if kept

### Priority 2 - Algorithm Improvements

3. **Review bounce count distribution** (Issue #3)
   - Consider if 50/50 mixture (`bounce_alpha = 0.5`) is optimal
   - Evaluate supporting more than 2 bounces
   - Match Mitsuba's geometric distribution more closely

4. **Validate directional sampling**
   - Verify cone-based sampling produces equivalent results to Mitsuba's approach
   - Compare convergence rates between implementations

### Priority 3 - Code Quality

5. **Add more validation**
   - Ensure PDF values match between implementations on test scenes
   - Add debug output comparing intermediate values
   - Create unit tests for critical components

6. **Documentation**
   - Document intentional deviations from reference
   - Add comments explaining complex logic
   - Reference Mitsuba paper/code for key algorithms

---

## Testing Recommendations

1. **Unit Tests**:
   - Test `classify_seed_lobe()` with various closure types
   - Verify Fresnel probability calculations match reference
   - Validate Jacobian determinant calculations

2. **Integration Tests**:
   - Render test scenes with pure mirrors (should use reflection only)
   - Render glass scenes and verify PDF values
   - Compare output images with Mitsuba reference

3. **Regression Tests**:
   - After fixing bugs, ensure no visual regressions
   - Verify convergence isn't degraded

---

## Conclusion

The Blender MPG implementation is generally well-structured but contains **one critical bug** (seed lobe classification) and several areas where it deviates from the proven Mitsuba reference without clear justification.

**Key Actions**:
1. ✅ **Fix the seed lobe classification bug immediately** - this affects correctness
2. ⚠️ **Review and simplify reflection probability logic** - reduce complexity
3. ⚠️ **Reconsider bounce distribution approach** - align with reference or justify deviation

The Jacobian calculations and PDF formulas appear correct, which is positive. The main concern is the seed generation phase where multiple deviations from the reference may compound to produce incorrect results.

---

## Files Analyzed

### Blender MPG Implementation (mpg-cpu branch)
- `intern/cycles/manifold/mpg.h`
- `intern/cycles/manifold/mpg.cpp`
- `intern/cycles/manifold/mpg_types.h`
- `intern/cycles/manifold/mpg_seed.h`
- `intern/cycles/manifold/mpg_seed.cpp`
- `intern/cycles/manifold/mpg_solve.h`
- `intern/cycles/manifold/mpg_solve.cpp`
- `intern/cycles/manifold/mpg_pdf.h`
- `intern/cycles/manifold/mpg_pdf.cpp`
- `intern/cycles/manifold/mpg_pgl_summary.h`
- `intern/cycles/manifold/mpg_pgl_summary.cpp`
- `intern/cycles/kernel/integrator/shade_surface.h`

### Mitsuba Reference Implementation
- `manifold_path_guiding.cpp`
- `manifold_path_guiding.h`
- `chain_distribution.h`
- `util.h`

---

**Report prepared by**: Claude Code Analysis
**Branch**: mpg-cpu
**Comparison date**: 2026-01-02
