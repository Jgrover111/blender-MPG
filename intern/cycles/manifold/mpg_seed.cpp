/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_seed.h"

#include "kernel/bvh/bvh.h"
#include "kernel/bvh/util.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/geom/object.h"
#include "kernel/geom/triangle.h"
#include "kernel/light/light.h"
#include "kernel/sample/mapping.h"
#include "kernel/integrator/surface_shader.h"
#include "kernel/svm/types.h"
#include "kernel/types.h"

#include "util/math_base.h"
#include "util/math_float4.h"
#include "util/math_intersect.h"

#include <cfloat>

CCL_NAMESPACE_BEGIN

static inline bool has_specular_bsdf_at_hit(KernelGlobals kg,
                                            const Ray &ray,
                                            const Intersection &isect,
                                            bool &has_smooth_normals)
{
  has_smooth_normals = false;
  ShaderData spec_sd = {};
  shader_setup_from_ray(kg, &spec_sd, &ray, const_cast<Intersection *>(&isect));

  const ConstIntegratorState integrator_state = nullptr;
  /* We only need closures, not emission; this is fast enough and CPU-safe. */
  surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE>(
      kg, integrator_state, &spec_sd, nullptr, PATH_RAY_CAMERA, true);

  has_smooth_normals = (spec_sd.shader & SHADER_SMOOTH_NORMAL) != 0;

  for (int i = 0; i < spec_sd.num_closure; ++i) {
    const ShaderClosure *c = &spec_sd.closure[i];
    if (!CLOSURE_IS_BSDF(c->type)) {
      continue;
    }
    const bool is_micro    = CLOSURE_IS_BSDF_MICROFACET(c->type);
    const bool is_singular = CLOSURE_IS_BSDF_SINGULAR(c->type);
    if (is_singular) {
      return true;
    }
    if (is_micro) {
      const MicrofacetBsdf *mf = reinterpret_cast<const MicrofacetBsdf *>(c);
      const float a = fmaxf(mf->alpha_x, mf->alpha_y);
      if (a <= 0.02f) {
        return true; /* razor-sharp microfacet behaves like specular for MPG v1 */
      }
    }
  }
  return false;
}

float3 mpg_surface_ray_offset(KernelGlobals kg,
                              const ShaderData &sd,
                              const float3 ray_P,
                              const float3 ray_D)
{
  if (!(sd.type & PRIMITIVE_TRIANGLE)) {
    return ray_P;
  }

  float3 verts[3];
  if (sd.type == PRIMITIVE_TRIANGLE) {
    triangle_vertices(kg, sd.prim, verts);
  }
  else {
    kernel_assert(sd.type == PRIMITIVE_MOTION_TRIANGLE);
    motion_triangle_vertices(kg, sd.object, sd.prim, sd.time, verts);
  }

  float3 local_ray_P = ray_P;
  float3 local_ray_D = ray_D;

  if (!(sd.object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    const Transform itfm = object_get_inverse_transform(kg, &sd);
    local_ray_P = transform_point(&itfm, local_ray_P);
    local_ray_D = transform_direction(&itfm, local_ray_D);
  }

  if (ray_triangle_intersect_self(local_ray_P, local_ray_D, verts)) {
    return ray_P;
  }

  return ray_offset(ray_P, sd.Ng);
}

bool mpg_generate_seed(KernelGlobals kg,
                       const ShaderData &sd,
                       const ShaderClosure &bsdf,
                       const GuideSummary &guide,
                       const MpgOptions &options,
                       const uint32_t path_flag,
                       const int bounce,
                       const RNGState &rng_state,
                       MpgSeedRay &seed,
                       MpgFailureCode &failure_code)
{
  seed = MpgSeedRay();
  failure_code = MPG_FAILURE_NONE;

  (void)bsdf;

  /* Determine the dominant seed direction from the guided mean with optional jitter. */
  float3 axis = guide.mean_dir;
  if (is_zero(axis)) {
    axis = sd.N;
  }

  const bool axis_valid = !is_zero(axis);
  if (axis_valid) {
    axis = normalize(axis);
  }

  const bool relaxed_gate = options.relax_gate;
  /* Relaxed gating shares the bootstrap cone/budget to aggressively search nearby
   * speculars, matching the Mitsuba fallback when the guide loses directional signal. */
  const bool bootstrap_seed = relaxed_gate || (guide.rbar <= 1.0e-4f);
  const bool use_uniform_fallback = !axis_valid;

  const float min_cone_angle = 0.00872664626f; /* ~0.5 degrees. */
  float jitter = fmaxf(options.angular_jitter, min_cone_angle);
  if (bootstrap_seed) {
    /* With no directional signal yet, or when the integrator relaxes the gate, explore a wide
     * bootstrap cone similar to the Mitsuba reference implementation. */
    jitter = 0.6f * M_PI_F;
  }
  float seed_pdf = 0.0f;
  float3 seed_direction = zero_float3();
  const int seed_branch_count = (bootstrap_seed || use_uniform_fallback) ? 32 : 16;
  const int max_seed_attempts = bootstrap_seed ? 32 : (use_uniform_fallback ? 32 : 16);
  bool seed_valid = false;
  Intersection isect = {};
  MpgFailureCode last_failure = MPG_FAILURE_SEED;

  auto try_seed_sample = [&](const float3 &candidate_direction,
                             const float candidate_pdf,
                             Intersection &out_isect) -> bool {
    if (is_zero(candidate_direction) || candidate_pdf <= 0.0f) {
      last_failure = MPG_FAILURE_INVALID_SEED_PDF;
      return false;
    }

    const float3 normalized_direction = normalize(candidate_direction);
    const float clamped_pdf = fmaxf(candidate_pdf, 1.0e-16f);

    Ray ray;
    ray.P = mpg_surface_ray_offset(kg, sd, sd.P, normalized_direction);
    ray.D = normalized_direction;
    ray.tmin = 0.0f;
    ray.tmax = FLT_MAX;
    ray.time = sd.time;
    ray.self.prim = sd.prim;
    ray.self.object = sd.object;
    ray.self.light_prim = PRIM_NONE;
    ray.self.light_object = OBJECT_NONE;

    Intersection candidate_isect = {};
    if (!scene_intersect(kg, &ray, PATH_RAY_ALL_VISIBILITY, &candidate_isect)) {
      last_failure = MPG_FAILURE_SEED;
      return false;
    }

    if (!(candidate_isect.type & PRIMITIVE_TRIANGLE)) {
      last_failure = MPG_FAILURE_GEOMETRY;
      return false;
    }

    bool has_smooth_normals = false;
    if (!has_specular_bsdf_at_hit(kg, ray, candidate_isect, has_smooth_normals)) {
      last_failure = MPG_FAILURE_NO_SPECULAR;
      return false;
    }

    out_isect = candidate_isect;
    seed_direction = normalized_direction;
    seed_pdf = clamped_pdf;
    seed.use_smooth_normals = has_smooth_normals;
    return true;
  };

  for (int attempt = 0; attempt < max_seed_attempts && !seed_valid; ++attempt) {
    const float2 rand = path_branched_rng_2D(
        kg, &rng_state, attempt, seed_branch_count, PRNG_SURFACE_BSDF);

    if (use_uniform_fallback) {
      seed_direction = sample_uniform_sphere(rand);
      seed_pdf = M_1_4PI_F;
    }
    else {
      float unused_cos = 0.0f;
      seed_direction = sample_uniform_cone(
          axis, one_minus_cos(jitter), rand, &unused_cos, &seed_pdf);
    }

    seed_valid = try_seed_sample(seed_direction, seed_pdf, isect);
  }

  if (!seed_valid && !use_uniform_fallback) {
    const int fallback_branch_count = 32;
    const int fallback_attempts = 32;
    const int attempt_offset = max_seed_attempts;
    const float fallback_one_minus_cos = one_minus_cos(0.6f * M_PI_F);

    for (int attempt = 0; attempt < fallback_attempts && !seed_valid; ++attempt) {
      const float2 rand = path_branched_rng_2D(
          kg, &rng_state, attempt + attempt_offset, fallback_branch_count, PRNG_SURFACE_BSDF);

      if (axis_valid) {
        float unused_cos = 0.0f;
        seed_direction = sample_uniform_cone(
            axis, fallback_one_minus_cos, rand, &unused_cos, &seed_pdf);
      }
      else {
        seed_direction = sample_uniform_sphere(rand);
        seed_pdf = M_1_4PI_F;
      }

      seed_valid = try_seed_sample(seed_direction, seed_pdf, isect);
    }
  }

  if (!seed_valid) {
    failure_code = last_failure;
    return false;
  }

  /* Sample an emitter using the Cycles light sampling routine. */
  const float3 rand_light = path_state_rng_3D(kg, &rng_state, PRNG_LIGHT);
  LightSample light_sample;
  if (!light_sample_from_position(kg,
                                  rand_light,
                                  sd.time,
                                  sd.P,
                                  sd.N,
                                  light_link_receiver_nee(kg, &sd),
                                  sd.flag,
                                  bounce,
                                  path_flag,
                                  &light_sample))
  {
    failure_code = MPG_FAILURE_SEED;
    return false;
  }

  if (light_sample.pdf <= 0.0f) {
    failure_code = MPG_FAILURE_INVALID_LIGHT_PDF;
    return false;
  }

  seed.direction = seed_direction;
  seed.seed_pdf = seed_pdf;
  seed.light_sample = light_sample;
  seed.path_flag = path_flag;
  const float light_distance = (light_sample.t == FLT_MAX) ? 1.0e6f : light_sample.t;
  seed.light_sample.P = sd.P + light_sample.D * light_distance;

  if (!is_zero(seed.light_sample.D)) {
    seed.light_sample.D = normalize(seed.light_sample.D);
  }

  seed.object = isect.object;
  seed.prim = isect.prim;
  seed.bary_u = isect.u;
  seed.bary_v = isect.v;
  return true;
}

CCL_NAMESPACE_END