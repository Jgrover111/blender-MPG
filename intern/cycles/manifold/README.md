# Cycles Manifold Path Guiding (MPG)

This module implements a CPU-only prototype of manifold path guiding for single-bounce
reflection and refraction events on triangle meshes. The API exposes
`mpg_try_connect()` which attempts to connect a diffuse shading point to a light via a
specular surface using Newton-style manifold refinement. The implementation keeps the
PDF in solid angle at the shading point and expects MIS with other proposals in the
same measure.

## Scope and limitations

* Only surface caustics are supported. Volume transport, microfacet roughness, and
  multi-bounce chains beyond a single specular interaction are outside v1.
* Triangle shading normals drive the specular constraint; normal mapping is ignored.
* The solver rejects configurations that trigger total internal reflection, run out of
  iterations, or violate barycentric bounds.
* Seeding depends on OpenPGL `GuideSummary` parameters and jitters the dominant guided
  direction. Seeds are rejected when no specular triangle is intersected or when lights
  cannot provide a deterministic endpoint.
* The proposal PDF multiplies the seed density, the light's solid-angle pdf at the
  shading point, the solver's surface Jacobian `|dX/du × dX/dv| / r_ds²`, and the
  specular cosine term. All components are evaluated explicitly so MIS combines in
  the same solid-angle measure as BSDF, guided, and NEE samples.

## Extending the module

* Two-bounce chains require augmenting the solver state to carry a second specular
  vertex and updating the Jacobian to include the additional constraints. The current
  structure keeps all intermediate derivatives explicit to make this extension
  straightforward.
* Volume support would need seeding and solvers that respect phase functions and media
  attenuation. The README documents the omission to avoid stubbed code paths in the
  implementation.

## Acceptance testing

1. Build Cycles with `WITH_CYCLES_MANIFOLD=ON`.
2. Render the Cornell box with a glass block and a sun lamp at 64 spp. MPG should
   produce visible caustics relative to the baseline.
3. For an environment map with a sharp sun pixel, compare MPG on/off at equal spp to
   ensure noise reduction and absence of bias at high spp (e.g. 16k spp).
4. Verify disabling MPG or failing the OpenPGL gate yields baseline renders within
   normal variance and comparable render time.