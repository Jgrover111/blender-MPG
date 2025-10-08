# Cycles Manifold Path Guiding (MPG)

This module implements a CPU-only prototype of manifold path guiding for single-bounce
reflection and refraction events on triangle meshes. The API exposes
`mpg_try_connect()` which attempts to connect a diffuse shading point to a light via a
specular surface using Newton-style manifold refinement. The implementation keeps the
PDF in solid angle at the shading point and expects MIS with other proposals in the
same measure. MPG is developed independently from the existing Cycles MNEE feature and
does not consult per-object *Cast/Receive Shadow Caustics* or per-light *Shadow
Caustics* toggles.

## Scope and limitations

* Only surface caustics are supported. Volume transport, microfacet roughness, and
  multi-bounce chains beyond a single specular interaction are outside v1. The
  `manifold_max_bounces` option is clamped to the [1, 2] range while two-bounce
  support for the solver is under active development. All Cycles materials that
  emit sharp reflective or refractive lobes (e.g., Principled BSDF in glass or
  transmission mode, Glass BSDF, Glossy BSDF, or custom node setups with sharp
  refraction) are expected to work with MPG as long as they satisfy the above
  assumptions.
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

## Debug AOVs

When Cycles is compiled with `WITH_CYCLES_DEBUG` and manifold guiding is enabled, a set
of auxiliary passes is available to inspect the guiding signals in the viewport. Each
pass writes a three-channel value (RGB) unless noted otherwise:

* **MPG Summary** – encodes the OpenPGL guide summary used to seed the solver.
  * `X`: peak guide weight (how dominant the strongest lobe is).
  * `Y`: von Mises–Fisher concentration `κ` (higher values mean a tighter lobe).
  * `Z`: `r̄`, the length of the mean direction (close to zero means no confident
    direction yet).
* **MPG Gate Flags** – shows which gating heuristics let MPG run for the current path.
  * `X`: set to 1 when a guide summary was available.
  * `Y`: set to 1 when the strict gate thresholds passed.
  * `Z`: set to 1 when the solver was allowed to run via relaxed/bootstrap gating.
* **MPG Attempt** – tracks solver bookkeeping per shading point.
  * `X`: number of solver attempts taken for this connection.
  * `Y`: bit mask of gate states accumulated from `MpgGateMask` (1=active gate,
    2=has relaxed direction, 4=has strict direction, 8=strict pass, 16=relaxed pass,
    32=bootstrap pass).
  * `Z`: failure code from `MpgFailureCode` (0=no failure, 1=seed, 2=TIR, …, 15=unknown).
* **MPG PDF Factors** – breaks down the manifold proposal density.
  * `X`: seed pdf from the guide cone.
  * `Y`: light pdf after snapping the light sample to the solved specular point.
  * `Z`: absolute solver Jacobian for the chain.
* **MPG Competing PDFs** – the other proposals considered in MIS.
  * `X`: BSDF pdf at the receiver (including any guiding blend weights).
  * `Y`: guided BSDF pdf (zero when guiding is inactive).
  * `Z`: next-event estimation pdf at the receiver.
* **MPG MIS** – MIS weight diagnostics.
  * `X`: manifold proposal pdf in solid angle at the receiver.
  * `Y`: MIS denominator (sum of all competing pdfs including MPG).
  * `Z`: final MIS weight applied to the contribution.
* **MPG Contribution** – raw spectral contribution from the manifold sample (RGB).

## Acceptance testing

1. Build Cycles with `WITH_CYCLES_MANIFOLD=ON`.
2. Render the Cornell box with a glass block and a sun lamp at 64 spp. MPG should
   produce visible caustics relative to the baseline.
3. For an environment map with a sharp sun pixel, compare MPG on/off at equal spp to
   ensure noise reduction and absence of bias at high spp (e.g. 16k spp).
4. Verify disabling MPG or failing the OpenPGL gate yields baseline renders within
   normal variance and comparable render time.