/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_seed.h"

#include "kernel/bvh/bvh.h"
#include "kernel/bvh/util.h"
#include "kernel/light/light.h"
#include "kernel/sample/mapping.h"
#include "kernel/svm/types.h"

#include "util/math_base.h"
#include "util/math_float4.h"

#include <cfloat>

CCL_NAMESPACE_BEGIN

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

  const bool bootstrap_seed = (guide.rbar <= 1.0e-4f);
  const bool use_uniform_fallback = bootstrap_seed || is_zero(axis);
  if (!use_uniform_fallback) {
    axis = normalize(axis);
  }

  const float min_cone_angle = 0.00872664626f; /* ~0.5 degrees. */
  float jitter = fmaxf(options.angular_jitter, min_cone_angle);
  if (guide.rbar <= 1.0e-4f) {
    /* With no directional signal yet, explore a wide bootstrap cone similar to the Mitsuba
     * reference implementation. */
    jitter = 0.5f * M_PI_F;
  }
  const float2 rand = path_state_rng_2D(kg, &rng_state, PRNG_SURFACE_BSDF);
  float seed_pdf = 0.0f;
  float3 seed_direction = zero_float3();

  if (use_uniform_fallback) {
    seed_direction = sample_uniform_sphere(rand);
    seed_pdf = M_1_4PI_F;
  }
  else {
    float unused_cos = 0.0f;
    seed_direction = sample_uniform_cone(axis, one_minus_cos(jitter), rand, &unused_cos, &seed_pdf);
  }

  if (is_zero(seed_direction) || seed_pdf <= 0.0f) {
    failure_code = MPG_FAILURE_SEED;
    return false;
  }
  seed_direction = normalize(seed_direction);
  seed_pdf = fmaxf(seed_pdf, 1.0e-16f);

  /* Trace the seed ray to locate the candidate specular surface. */
  Ray ray;
  ray.P = ray_offset(sd.P, sd.Ng);
  ray.D = seed_direction;
  ray.tmin = 0.0f;
  ray.tmax = FLT_MAX;
  ray.time = sd.time;
  ray.self.prim = sd.prim;
  ray.self.object = sd.object;
  ray.self.light_prim = PRIM_NONE;
  ray.self.light_object = OBJECT_NONE;

  Intersection isect;
  if (!scene_intersect(kg, &ray, PATH_RAY_ALL_VISIBILITY, &isect)) {
    failure_code = MPG_FAILURE_SEED;
    return false;
  }

  if (!(isect.type & PRIMITIVE_TRIANGLE)) {
    failure_code = MPG_FAILURE_GEOMETRY;
    return false;
  }

  seed.use_smooth_normals = false;

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

  seed.object = isect.object;
  seed.prim = isect.prim;
  seed.bary_u = isect.u;
  seed.bary_v = isect.v;
  return true;
}

CCL_NAMESPACE_END