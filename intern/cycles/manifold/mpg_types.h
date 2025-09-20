/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/light/common.h"

#include "util/math.h"

CCL_NAMESPACE_BEGIN

struct ShadingPoint {
  float3 position = zero_float3();
  float3 geometric_normal = zero_float3();
  float3 shading_normal = zero_float3();
  float3 wo = zero_float3();
  float time = 0.0f;
};

struct MpgSeedRay {
  float3 direction = zero_float3();
  float seed_pdf = 0.0f;
  LightSample light_sample = {};
  int object = -1;
  int prim = -1;
  float bary_u = 0.0f;
  float bary_v = 0.0f;
};

struct MpgSolverOutput {
  bool success = false;
  float3 wi = zero_float3();
  float3 specular_point = zero_float3();
  float3 specular_normal = zero_float3();
  float3 dir_ds = zero_float3();
  float3 dir_sl = zero_float3();
  float distance_ds = 0.0f;
  float distance_sl = 0.0f;
  bool is_refraction = false;
  Spectrum spec_weight = zero_spectrum();
  float3 dXdu = zero_float3();
  float3 dXdv = zero_float3();
  float3 dNdu = zero_float3();
  float3 dNdv = zero_float3();
  float u = 0.0f;
  float v = 0.0f;
  float jacobian = 0.0f;
  float visibility = 1.0f;
  int object = -1;
  int prim = -1;
};

CCL_NAMESPACE_END