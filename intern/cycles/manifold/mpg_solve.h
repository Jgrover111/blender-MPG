/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "manifold/mpg.h"
#include "manifold/mpg_types.h"
#include "manifold/mpg_seed.h"

CCL_NAMESPACE_BEGIN

bool mpg_solve_single_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
                             const MpgOptions &options,
                             RNGState &rng_state,
                             MpgSolverOutput &result,
                             MpgFailureCode &failure_code);

bool mpg_solve_double_bounce(KernelGlobals kg,
                             const ShaderData &sd,
                             const ShaderClosure &bsdf,
                             const MpgSeedRay &seed,
                             const MpgOptions &options,
                             RNGState &rng_state,
                             MpgSolverOutput &result,
                             MpgFailureCode &failure_code);

float mpg_compute_segment_visibility(KernelGlobals kg,
                                     const float3 &start_point,
                                     const float3 &start_normal,
                                     const float3 &end_point,
                                     float time,
                                     int skip_object,
                                     int skip_prim,
                                     int skip_light_object,
                                     int skip_light_prim);

CCL_NAMESPACE_END