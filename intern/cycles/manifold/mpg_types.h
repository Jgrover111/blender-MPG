/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/light/common.h"

#include "util/math.h"

#include <cstdint>

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
  uint32_t path_flag = 0;
  int object = -1;
  int prim = -1;
  float bary_u = 0.0f;
  float bary_v = 0.0f;
  bool use_smooth_normals = false;
};

struct MpgSpecularVertex {
  float3 position = zero_float3();
  float3 normal = zero_float3();
  float3 dir_in = zero_float3();
  float3 dir_out = zero_float3();
  float distance_in = 0.0f;
  float distance_out = 0.0f;
  float eta = 1.0f;
  float cos_theta_in = 0.0f;
  float cos_theta_out = 0.0f;
  Spectrum throughput = zero_spectrum();
  float3 dXdu = zero_float3();
  float3 dXdv = zero_float3();
  float3 dNdu = zero_float3();
  float3 dNdv = zero_float3();
  float u = 0.0f;
  float v = 0.0f;
  float jacobian = 0.0f;
  bool is_refraction = false;
  bool total_internal_reflection = false;
  int object = -1;
  int prim = -1;
};

struct MpgSolverOutput {
  bool success = false;
  int specular_vertex_count = 0;
  float3 wi = zero_float3();
  float visibility = 1.0f;
  float jacobian_total = 0.0f;
  Spectrum specular_throughput = zero_spectrum();
  MpgSpecularVertex specular_vertices[2];

  /* Legacy single-bounce fields kept for callers that have not yet
   * transitioned to the multi-vertex representation. These mirror the
   * contents of the first entry in `specular_vertices` when present. */
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
  int object = -1;
  int prim = -1;
};

CCL_NAMESPACE_END