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
                       MpgSeedRay &seed)
{
  seed = MpgSeedRay();

  (void)bsdf;

  const bool gate_active = (options.gate_w > 0.0f) || (options.gate_kappa > 0.0f);
  if (gate_active) {
    const bool has_direction_relaxed = (guide.rbar > 1.0e-4f);
    const bool has_direction_strict = (guide.rbar > 1.0e-3f);
    const bool strict_gate = has_direction_strict && (guide.peak_weight >= options.gate_w) &&
                             (guide.kappa >= options.gate_kappa);
    const bool relaxed_gate = options.relax_gate && has_direction_relaxed;
    if (!strict_gate && !relaxed_gate) {
      return false;
    }
  }
  else if (guide.rbar <= 1.0e-5f) {
    return false;
  }

  /* Determine the dominant seed direction from the guided mean with optional jitter. */
  float3 axis = guide.mean_dir;
  if (is_zero(axis)) {
    axis = sd.N;
  }
  if (is_zero(axis)) {
    return false;
  }
  axis = normalize(axis);

  float3 seed_direction = axis;
  float seed_pdf = 1.0f;
  if (options.angular_jitter > 0.0f) {
    const float2 rand = path_state_rng_2D(kg, &rng_state, PRNG_SURFACE_BSDF);
    float unused_cos = 0.0f;
    float pdf = 0.0f;
    seed_direction = sample_uniform_cone(
        axis, one_minus_cos(options.angular_jitter), rand, &unused_cos, &pdf);
    seed_pdf = pdf;
  }

  if (is_zero(seed_direction) || seed_pdf <= 0.0f) {
    return false;
  }
  seed_direction = normalize(seed_direction);

  /* Trace the seed ray to locate the candidate specular surface. */
  Ray ray;
  ray.P = sd.P;
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
    return false;
  }

  if (!(isect.type & PRIMITIVE_TRIANGLE)) {
    return false;
  }

  const int shader_id = intersection_get_shader(kg, &isect);
  const KernelShader &kshader = kernel_data_fetch(shaders, shader_id);
  seed.use_smooth_normals = (kshader.flags & SHADER_SMOOTH_NORMAL) != 0;

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
    return false;
  }

  if (light_sample.pdf <= 0.0f) {
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