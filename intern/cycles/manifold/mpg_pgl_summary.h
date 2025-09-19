/* SPDX-FileCopyrightText: 2024 Blender Foundation
*
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "integrator/guiding.h"

#include "util/math.h"

struct RNG;

#if defined(WITH_PATH_GUIDING)
namespace openpgl {
namespace cpp {
struct SurfaceSamplingDistribution;
}  // namespace cpp
}  // namespace openpgl

using OpenPGLSurfaceDistribution = openpgl::cpp::SurfaceSamplingDistribution;
#endif

CCL_NAMESPACE_BEGIN

#if defined(WITH_PATH_GUIDING)

bool pgl_estimate_summary(const OpenPGLSurfaceDistribution &dist_world,
                          const float3 &Ng_world,
                          RNG &rng,
                          GuideSummary &out,
                          int n_samples = 128,
                          float cone_half_angle_rad = (10.0f * (M_PI_F / 180.0f)));

#endif

CCL_NAMESPACE_END