/* SPDX-FileCopyrightText: 2024 Blender Foundation
*
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "manifold/mpg.h"
#include "manifold/mpg_types.h"

CCL_NAMESPACE_BEGIN

struct MpgSeedRay {
  float3 direction = zero_float3();
  float3 emitter_position = zero_float3();
  float3 emitter_normal = zero_float3();
  int emitter_index = -1;
  float seed_pdf = 0.0f;
  float emitter_pdf = 0.0f;
  int triangle_index = -1;
  float bary_u = 0.0f;
  float bary_v = 0.0f;
  bool is_refraction = false;
  float eta = 1.0f;
  float visibility = 0.0f;
};

bool mpg_generate_seed(const ShadingPoint &D,
                       const ClosureBSDF &bsdf,
                       const Lights &lights,
                       const GuideSummary &guide,
                       const MpgOptions &options,
                       RNG &rng,
                       MpgSeedRay &seed);

CCL_NAMESPACE_END