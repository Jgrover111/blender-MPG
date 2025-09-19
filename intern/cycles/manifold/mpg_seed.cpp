/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_seed.h"

#include "kernel/bvh/bvh.h"
#include "kernel/bvh/util.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/triangle.h"
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
                       const RNGState &rng_state,
                       MpgSeedRay &seed)
{
  seed = MpgSeedRay();

  if (guide.peak_weight < options.gate_w || guide.kappa < options.gate_kappa) {
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
  if (!(kshader.flags & SHADER_SMOOTH_NORMAL)) {
    return false;
  }

  const int object_flags = kernel_data_fetch(object_flag, isect.object);

  float3 verts[3];
  float3 normals[3];
  if (object_flags & SD_OBJECT_MOTION) {
    motion_triangle_vertices_and_normals(
        kg, isect.object, isect.prim, sd.time, verts, normals);
  }
  else {
    triangle_vertices_and_normals(kg, isect.prim, verts, normals);
  }

  /* Sample an emitter using the Cycles light sampling routine. */
  const float3 rand_light = path_state_rng_3D(kg, &rng_state, PRNG_LIGHT);
  LightSample light_sample;
  if (!light_sample_from_position(kg,
                                  rand_light,
                                  sd.time,
                                  sd.P,
                                  sd.N,
                                  sd.object,
                                  sd.flag,
                                  0,
                                  PATH_RAY_DIFFUSE,
                                  &light_sample))
  {
    return false;
  }

  if (light_sample.pdf <= 0.0f) {
    return false;
  }

  seed.direction = seed_direction;
  seed.seed_pdf = seed_pdf;
  seed.light = light_sample;
  seed.emitter_pdf = light_sample.pdf;
  seed.emitter_shader = light_sample.shader;
  seed.emitter_normal = make_float3(light_sample.Ng.x, light_sample.Ng.y, light_sample.Ng.z);
  const float light_distance = (light_sample.t == FLT_MAX) ? 1.0e6f : light_sample.t;
  seed.emitter_position = sd.P + light_sample.D * light_distance;
  seed.visibility = 1.0f;

  seed.object = isect.object;
  seed.prim = isect.prim;
  seed.bary_u = isect.u;
  seed.bary_v = isect.v;
  seed.tri_v0 = verts[0];
  seed.tri_v1 = verts[1];
  seed.tri_v2 = verts[2];
  seed.tri_n0 = normals[0];
  seed.tri_n1 = normals[1];
  seed.tri_n2 = normals[2];

  seed.is_refraction = CLOSURE_IS_REFRACTION(bsdf.type) || CLOSURE_IS_GLASS(bsdf.type);
  seed.eta = 1.0f;
  if (seed.is_refraction) {
    const MicrofacetBsdf *microfacet = reinterpret_cast<const MicrofacetBsdf *>(&bsdf);
    seed.eta = fmaxf(microfacet->ior, 1e-5f);
  }

  return true;
}

CCL_NAMESPACE_END