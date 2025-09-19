/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/math.h"
#include "util/vector.h"

#include <random>

CCL_NAMESPACE_BEGIN

struct ShadingPoint {
  float3 position = zero_float3();
  float3 geometric_normal = zero_float3();
  float3 shading_normal = zero_float3();
  float3 wo = zero_float3();
  float time = 0.0f;
};

struct ClosureBSDF {
  enum class Type {
    Reflection,
    Refraction,
  };

  Type type = Type::Reflection;
  float eta = 1.0f;
  float3 normal = zero_float3();
  float3 tangent = zero_float3();
  float3 bitangent = zero_float3();

  bool is_refraction() const
  {
    return type == Type::Refraction;
  }
};

class RNG {
 public:
  explicit RNG(uint64_t seed = 0);

  float uniform_float();
  float2 uniform_float2();
  float3 uniform_direction();

 private:
  std::mt19937_64 engine_;
  std::uniform_real_distribution<float> dist_;
};

class Lights {
 public:
  struct SpecularTriangle {
    float3 v0;
    float3 v1;
    float3 v2;
    float3 n0;
    float3 n1;
    float3 n2;
    float eta;
    bool is_refraction;
    bool enabled;

    float3 normal(const float u, const float v) const;
    float3 position(const float u, const float v) const;
    float3 dXdu() const;
    float3 dXdv() const;
  };

  struct Emitter {
    float3 position;
    float3 normal;
    float area;
    bool enabled;
  };

  void add_specular_triangle(const SpecularTriangle &tri);
  void add_emitter(const Emitter &emitter);

  bool intersect_specular(const float3 &origin,
                          const float3 &direction,
                          int &triangle_index,
                          float &t,
                          float &u,
                          float &v) const;

  bool sample_emitter(RNG &rng, int &index, float3 &position, float3 &normal, float &pdf) const;

  const SpecularTriangle &triangle(const int index) const;
  const Emitter &emitter(const int index) const;
  int num_triangles() const;
  int num_emitters() const;

 private:
  vector<SpecularTriangle> triangles_;
  vector<Emitter> emitters_;
};

CCL_NAMESPACE_END