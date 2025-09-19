/* SPDX-FileCopyrightText: 2024 Blender Foundation
*
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "manifold/mpg.h"
#include "manifold/mpg_types.h"
#include "manifold/mpg_seed.h"

CCL_NAMESPACE_BEGIN

struct MpgSolverOutput {
  bool success = false;
  float3 wi = zero_float3();
  float3 specular_point = zero_float3();
  float3 specular_normal = zero_float3();
  float3 dir_ds = zero_float3();
  float3 dir_sl = zero_float3();
  float distance_ds = 0.0f;
  float distance_sl = 0.0f;
  float3 dXdu = zero_float3();
  float3 dXdv = zero_float3();
  float3 dNdu = zero_float3();
  float3 dNdv = zero_float3();
  float u = 0.0f;
  float v = 0.0f;
  float jacobian = 0.0f;
  float visibility = 1.0f;
};

bool mpg_solve_single_bounce(const ShadingPoint &D,
                             const ClosureBSDF &bsdf,
                             const Lights &lights,
                             const MpgSeedRay &seed,
                             const MpgOptions &options,
                             RNG &rng,
                             MpgSolverOutput &result);

CCL_NAMESPACE_END