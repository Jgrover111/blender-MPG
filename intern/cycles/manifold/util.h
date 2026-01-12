/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Utility functions for Manifold Path Guiding
 *
 * This file corresponds to: util.h in the Mitsuba MPG reference
 * https://github.com/mollnn/manifold-path-guiding/tree/main/mitsuba/src/integrators/MPG
 *
 * In Mitsuba, util.h contains:
 * - Bit manipulation helpers (set_bit, get_bit, set_chaintype_bit)
 * - SubpathSample structure for storing path segments
 * - Atomic operations for thread-safe accumulation
 * - Timing utilities
 *
 * In Cycles, most of these utilities are already provided by:
 * - util/math.h: Math utilities
 * - util/hash.h: Hashing and bit operations
 * - util/atomic.h: Atomic operations
 *
 * This file is kept as a placeholder for future MPG-specific utilities
 * that don't fit elsewhere.
 */

#pragma once

#include "util/math.h"
#include "util/types.h"

#include <cstdint>

CCL_NAMESPACE_BEGIN

/* Placeholder for future MPG-specific utilities.
 * Currently, all needed utilities are provided by Cycles' existing util/ headers.
 *
 * Examples of what might go here in the future:
 * - MPG-specific data structures for path caching
 * - Specialized math operations for manifold constraints
 * - Debugging and profiling helpers
 */

CCL_NAMESPACE_END
