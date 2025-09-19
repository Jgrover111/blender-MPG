/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "manifold/mpg_seed.h"

#include "util/math_float4.h"

CCL_NAMESPACE_BEGIN

bool mpg_generate_seed(KernelGlobals kg,
                       const ShaderData &sd,
                       const ShaderClosure &bsdf,
                       const GuideSummary &guide,
                       const MpgOptions &options,
                       const RNGState &rng_state,
                       MpgSeedRay &seed)
{
  (void)kg;
  (void)sd;
  (void)bsdf;
  (void)rng_state;

  seed = MpgSeedRay();

  if (guide.peak_weight < options.gate_w || guide.kappa < options.gate_kappa) {
    return false;
  }

  return false;
}

CCL_NAMESPACE_END