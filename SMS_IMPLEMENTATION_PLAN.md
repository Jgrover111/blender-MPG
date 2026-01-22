# Specular Manifold Sampling (SMS) Implementation Plan for Blender Cycles

## Overview

This document outlines the implementation plan for adding Specular Manifold Sampling (SMS) to Blender Cycles as an alternative to the existing Manifold Next Event Estimation (MNEE) implementation.

### Key Differences: MNEE vs SMS

**MNEE (Current Implementation)**
- **Scope**: Shadow caustics only (caustics that appear in shadow regions)
- **Technique**: Manifold walking from receiver to light through refractive interfaces
- **Use Case**: Renders caustics that would otherwise appear as black/dark in shadows
- **Invocation**: Only called during direct light sampling on receiver surfaces

**SMS (To Be Implemented)**
- **Scope**: Full caustic paths (both shadow and non-shadow caustics)
- **Technique**: Manifold walking for complete specular chains (e.g., L-S-D-S-E paths)
- **Use Case**: Handles all types of caustic light transport, including glossy caustics
- **Invocation**: Can be called at any bounce, not limited to direct lighting

## UI Changes

### Proposed Naming

Replace the current boolean caustics properties with an enum offering three modes:

1. **"Off"** - No caustics rendering
2. **"Shadow Caustics"** - Current MNEE implementation (shadow caustics only)
3. **"Caustic Paths"** - New SMS implementation (full manifold path sampling)

Alternative naming considered:
- "Full Caustics" (simpler but less descriptive)
- "Specular Manifold Sampling" (too technical for UI)
- "Complete Caustics" (ambiguous)

### UI Property Locations

**Objects** (Object Properties > Shading > Caustics):
- Caster mode: Enum with Off/Shadow Caustics/Caustic Paths
- Receiver mode: Enum with Off/Shadow Caustics/Caustic Paths

**Lights** (Light Properties > Light):
- Caustics mode: Enum with Off/Shadow Caustics/Caustic Paths

**World** (World Properties > Settings > Surface):
- Caustics mode: Enum with Off/Shadow Caustics/Caustic Paths

## Implementation Phases

### Phase 1: Property System Refactoring (Foundation)

**1.1 RNA Property Changes**
- File: `intern/cycles/blender/addon/properties.py`
- Convert `is_caustics_caster`, `is_caustics_receiver`, `is_caustics_light` from BoolProperty to EnumProperty
- Define caustics mode enum: `CAUSTICS_OFF`, `CAUSTICS_SHADOW`, `CAUSTICS_FULL`

**1.2 DNA/Socket Changes**
- File: `intern/cycles/scene/object.h`
- Replace boolean sockets with enum sockets for caustics mode
- Update Object class with caustics_caster_mode and caustics_receiver_mode

**1.3 UI Updates**
- File: `intern/cycles/blender/addon/ui.py`
- Update panels to show enum dropdowns instead of checkboxes
- Maintain backward compatibility with version upgrade system

**1.4 Sync Code Updates**
- File: `intern/cycles/blender/object.cpp`
- Update Blender->Cycles sync to read enum instead of boolean
- File: `intern/cycles/blender/light.cpp`
- Update light caustics property sync

**1.5 Scene Setup**
- File: `intern/cycles/scene/scene.cpp`
- Update caustics detection to handle three modes
- Separate flags for MNEE vs SMS activation

### Phase 2: Kernel Infrastructure

**2.1 Type Definitions**
- File: `intern/cycles/kernel/types.h`
- Add `KERNEL_FEATURE_SMS` flag
- Add `CausticsMode` enum to object flags
- Add SMS-specific path state flags

**2.2 Path State Management**
- Extend PathState to track SMS manifold vertices
- Add SMS-specific bounce counting
- Track whether current path is SMS candidate

**2.3 Intersection Handling**
- File: `intern/cycles/kernel/integrator/intersect_closest.h`
- Update to track SMS caustic chain state
- Mark surfaces that can participate in SMS paths

### Phase 3: SMS Algorithm Implementation

**3.1 Core SMS Kernel**
- Create new file: `intern/cycles/kernel/integrator/sms.h`
- Implement manifold vertex structure for SMS
- Implement manifold walking algorithm

**3.2 Key SMS Components**

**3.2.1 Manifold Path Construction**
- Build complete specular chain (e.g., light -> glass -> diffuse -> glass -> camera)
- Track up to N specular vertices (suggest N=8 for initial implementation)
- Store per-vertex differential geometry and constraints

**3.2.2 Newton Solver for SMS**
- Multi-dimensional Newton solver for complete path
- Solve for all specular vertex positions simultaneously
- Handle both reflection and refraction at each vertex
- Adaptive step size control

**3.2.3 Path Contribution**
- Calculate complete path throughput including:
  - BSDF evaluations at all vertices
  - Geometric terms (dot products, distances)
  - Fresnel reflections/refractions
  - Multiple importance sampling weight

**3.3 Integration Points**

**3.3.1 During Camera Ray Intersection**
- When hitting specular surface, check if SMS path is possible
- Record specular vertex in manifold chain

**3.3.2 During Path Extension**
- At each bounce, evaluate if SMS should be attempted
- Build manifold constraint system incrementally
- Invoke SMS solver when complete S-D-S pattern detected

**3.3.3 Light Sampling**
- When sampling lights, check for specular connections
- Use SMS to find manifold path to light source
- Replace standard NEE with SMS-guided connection

### Phase 4: Material System Integration

**4.1 BSDF Queries**
- Extend BSDF interface to provide manifold derivatives
- Support both reflection and refraction manifolds
- Handle microfacet roughness in SMS (challenging!)

**4.2 Supported Materials (Initial)**
- Start with perfect specular (mirror, glass)
- Extend to low-roughness microfacet
- Glossy caustics will be Phase 5 extension

**4.3 Material Validation**
- Reject incompatible BSDF types from SMS chain
- Fall back to standard path tracing when SMS not applicable
- Validate IOR matching for refractive paths

### Phase 5: Optimization and Extensions

**5.1 Performance Optimization**
- Adaptive solver iteration count
- Early termination for unlikely paths
- Caching of manifold vertex computations
- Parallel manifold evaluation

**5.2 Advanced Features**
- Glossy manifolds (rough glass caustics)
- Nested dielectrics handling
- Volumetric scattering in SMS paths
- Spectral caustics (wavelength-dependent)

**5.3 Quality Improvements**
- Multiple importance sampling with standard PT
- Probabilistic path selection (SMS vs standard)
- Russian roulette for SMS paths
- Variance reduction techniques

### Phase 6: Testing and Validation

**6.1 Test Scenes**
- Simple glass sphere on plane (classic caustic test)
- Stanford bunny in glass
- Wine glass with liquid
- Prism/dispersion test
- Nested glass objects
- Glossy metal caustics

**6.2 Validation**
- Compare against ground truth (long path tracing)
- Performance benchmarks vs MNEE
- Memory usage profiling
- Edge case testing (grazing angles, total internal reflection)

**6.3 Documentation**
- User manual updates
- API documentation
- Example blend files
- Performance tuning guide

## Technical Challenges

### Challenge 1: Solver Complexity
- **Issue**: SMS requires solving for multiple vertices simultaneously
- **Solution**: Implement block Newton solver with sparse Jacobian
- **Fallback**: Limit max specular vertices to keep Jacobian manageable

### Challenge 2: Glossy Manifolds
- **Issue**: Rough surfaces create multi-valued manifolds
- **Solution**: Phase 1 restricts to smooth specular, Phase 5 adds glossy sampling
- **Research**: May need multiple manifold solutions with importance sampling

### Challenge 3: Performance
- **Issue**: SMS is computationally expensive (matrix ops, iterations)
- **Solution**: Only invoke on high-importance paths, use caching
- **Monitoring**: Add debug counters for SMS invocations and success rate

### Challenge 4: Numerical Stability
- **Issue**: Near-singular configurations (grazing angles, near-parallel surfaces)
- **Solution**: Regularization terms, adaptive convergence threshold
- **Validation**: Extensive testing with edge cases

### Challenge 5: MIS Weight Calculation
- **Issue**: Balancing SMS with standard path tracing
- **Solution**: Implement proper probability density estimation for manifold paths
- **Research**: May need to adapt existing MIS theory

## Code Architecture

### New Files to Create

```
intern/cycles/kernel/integrator/sms.h          # Main SMS implementation
intern/cycles/kernel/integrator/sms_solver.h   # Newton solver for SMS
intern/cycles/kernel/integrator/sms_util.h     # Utility functions
```

### Modified Files

```
intern/cycles/blender/addon/properties.py      # Enum properties
intern/cycles/blender/addon/ui.py              # UI panels
intern/cycles/blender/object.cpp               # Sync code
intern/cycles/blender/light.cpp                # Light sync
intern/cycles/scene/object.h                   # C++ properties
intern/cycles/scene/object.cpp                 # Object class
intern/cycles/scene/scene.cpp                  # Scene setup
intern/cycles/kernel/types.h                   # Type definitions
intern/cycles/kernel/integrator/shade_surface.h # SMS invocation
intern/cycles/kernel/integrator/intersect_closest.h # Path tracking
intern/cycles/kernel/film/light_passes.h       # Debug passes (optional)
```

## Implementation Timeline Estimate

**Phase 1** (Property System):
- Foundation work, critical for all other phases
- Estimated complexity: Low-Medium

**Phase 2** (Kernel Infrastructure):
- Setting up data structures and flags
- Estimated complexity: Medium

**Phase 3** (SMS Algorithm):
- Core implementation, most complex phase
- Estimated complexity: High
- Sub-phases:
  - 3.1: Basic manifold walking for simple cases
  - 3.2: Full solver with multiple vertices
  - 3.3: Integration with path tracer

**Phase 4** (Material Integration):
- Extending BSDF system for manifolds
- Estimated complexity: Medium-High

**Phase 5** (Optimization):
- Performance tuning and advanced features
- Estimated complexity: Medium
- Can be done incrementally after Phase 3

**Phase 6** (Testing):
- Validation and documentation
- Estimated complexity: Medium
- Ongoing throughout implementation

## Success Criteria

1. **Functional**: SMS successfully renders caustic paths that MNEE cannot
2. **Quality**: Reduced noise in caustic regions compared to standard PT
3. **Performance**: Competitive rendering time vs path tracing for caustic scenes
4. **Correctness**: Unbiased results (or known bias with compensation)
5. **Robustness**: Handles edge cases gracefully, falls back when needed
6. **Usability**: Clear UI, good defaults, helpful documentation

## References

### Academic Papers
- "Specular Manifold Sampling for Rendering High-Frequency Caustics and Glints" (Zeltner et al., 2020)
- "Manifold Next Event Estimation" (Hanika et al., 2015) - current MNEE implementation
- "Manifold Exploration" (Jakob & Marschner, 2012) - foundational work

### Implementation References
- Mitsuba 3 SMS implementation (open source reference)
- PBRT-v4 manifold code (if available)
- Current Blender MNEE code (intern/cycles/kernel/integrator/mnee.h)

## Open Questions

1. **Sampling Strategy**: When should SMS be invoked vs standard path tracing?
   - Per-path probability?
   - Based on material properties?
   - Adaptive based on scene analysis?

2. **Maximum Chain Length**: What's the practical limit for specular vertices?
   - Memory constraints
   - Solver convergence with high dimensions
   - Typical use cases

3. **Glossy Cutoff**: At what roughness does SMS become impractical?
   - Need research on manifold multi-valuedness
   - May need different strategy for rough surfaces

4. **Debug Visualization**: What debug passes would help artists?
   - SMS success/failure map
   - Iteration count visualization
   - Manifold vertex positions
   - Path contribution breakdown

5. **Backward Compatibility**: How to handle old scenes with boolean properties?
   - Version upgrade system
   - Default mode for existing caustics-enabled objects
   - Python API compatibility layer

## Next Steps

1. ✓ Create this planning document
2. Implement Phase 1 (Property System Refactoring)
   - Start with RNA property changes
   - Update UI to show enum options
   - Test that existing MNEE still works with new enum system
3. Review and refine plan based on initial implementation experience
4. Begin Phase 2 (Kernel Infrastructure) once Phase 1 is validated

---

**Document Version**: 1.0
**Last Updated**: 2026-01-22
**Author**: Claude (AI Assistant)
**Status**: Initial Planning Phase
