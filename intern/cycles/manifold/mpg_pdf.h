/* SPDX-FileCopyrightText: 2024 Blender Foundation
*
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "manifold/mpg_solve.h"

CCL_NAMESPACE_BEGIN

bool mpg_evaluate_pdf(const ShadingPoint &D,
                      const ClosureBSDF &bsdf,
                      const MpgSeedRay &seed,
                      const MpgSolverOutput &solution,
                      float &pdf);

CCL_NAMESPACE_END