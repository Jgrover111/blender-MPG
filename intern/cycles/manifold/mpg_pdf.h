/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "manifold/mpg_solve.h"

CCL_NAMESPACE_BEGIN

bool mpg_evaluate_pdf(KernelGlobals kg,
                      const ShaderData &sd,
                      const ShaderClosure &bsdf,
                      const GuideSummary &guide,
                      const MpgSeedRay &seed,
                      const MpgSolverOutput &solution,
                      float &pdf);

float mpg_light_sample_pdf_solid(KernelGlobals kg,
                                 const ShaderData &sd,
                                 const LightSample &light_sample);

CCL_NAMESPACE_END