/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Directional Tree (DTree) - OpenPGL Wrapper for MPG
 *
 * This file corresponds to: dtree.h in the Mitsuba MPG reference
 * https://github.com/mollnn/manifold-path-guiding/tree/main/mitsuba/src/integrators/MPG
 *
 * Key differences from Mitsuba:
 * - Uses OpenPGL's directional distributions instead of custom DTree
 * - Provides GuideSummary estimation from OpenPGL distributions
 * - Simpler interface since OpenPGL handles the tree building internally
 *
 * In Mitsuba, DTree implements a custom directional quad-tree for importance sampling.
 * In Cycles, we delegate this to OpenPGL's SurfaceSamplingDistribution and extract
 * summary statistics (mean direction, concentration, peak weight) for MPG seed generation.
 */

#pragma once

#include "integrator/guiding.h"

#include "util/math.h"

#include <cstdint>

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

/* Estimate directional summary from OpenPGL distribution
 *
 * This function analyzes an OpenPGL SurfaceSamplingDistribution to extract
 * summary statistics for MPG:
 * - mean_dir: dominant direction (analogous to DTree's peak direction)
 * - kappa: concentration parameter (estimated from coherence)
 * - peak_weight: fraction of samples in dominant cluster
 * - rbar: mean resultant length (coherence metric)
 *
 * Parameters:
 * - dist_world: OpenPGL distribution in world space
 * - Ng_world: geometric normal for hemisphere constraint
 * - rng_seed: random seed for sampling
 * - out: output GuideSummary
 * - n_samples: number of samples to draw for analysis (default: 128)
 * - cone_half_angle_rad: clustering cone angle (default: 10 degrees)
 *
 * Returns: true if summary estimation succeeded, false otherwise
 */
bool pgl_estimate_summary(const OpenPGLSurfaceDistribution &dist_world,
                          const float3 &Ng_world,
                          uint32_t rng_seed,
                          GuideSummary &out,
                          int n_samples = 128,
                          float cone_half_angle_rad = (10.0f * (M_PI_F / 180.0f)));

#endif

CCL_NAMESPACE_END
