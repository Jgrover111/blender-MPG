/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/device/cpu/globals.h"
#include "kernel/osl/globals.h"

#ifdef WITH_CYCLES_MANIFOLD
#  include "BLI_rand.h"
#endif

#include "util/guiding.h"  // IWYU pragma: keep
#include "util/profiling.h"

CCL_NAMESPACE_BEGIN

#ifdef WITH_CYCLES_MANIFOLD
void BLIManifoldRngDeleter::operator()(::RNG *rng) const
{
  if (rng != nullptr) {
    BLI_rng_free(rng);
  }
}
#endif

ThreadKernelGlobalsCPU::ThreadKernelGlobalsCPU(const KernelGlobalsCPU &kernel_globals,
                                               OSLGlobals *osl_globals,
                                               Profiler &cpu_profiler,
                                               const int thread_index)
    : KernelGlobalsCPU(kernel_globals),
#ifdef WITH_OSL
      osl(osl_globals, thread_index),
#endif
      cpu_profiler_(cpu_profiler)
{
#ifndef WITH_OSL
  (void)thread_index;
  (void)osl_globals;
#endif

#if defined(WITH_PATH_GUIDING)
  opgl_path_segment_storage = make_unique<openpgl::cpp::PathSegmentStorage>();
#endif

#ifdef WITH_CYCLES_MANIFOLD
  manifold_rng = unique_ptr<::RNG, BLIManifoldRngDeleter>(
      BLI_rng_new(0x9e3779b9u + uint(thread_index)), BLIManifoldRngDeleter());
#endif
}

void ThreadKernelGlobalsCPU::start_profiling()
{
  cpu_profiler_.add_state(&profiler);
}

void ThreadKernelGlobalsCPU::stop_profiling()
{
  cpu_profiler_.remove_state(&profiler);
}

CCL_NAMESPACE_END
