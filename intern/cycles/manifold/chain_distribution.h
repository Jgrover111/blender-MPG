/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Chain Distribution - OpenPGL Integration Documentation
 *
 * This file corresponds to: chain_distribution.h in the Mitsuba MPG reference
 * https://github.com/mollnn/manifold-path-guiding/tree/main/mitsuba/src/integrators/MPG
 *
 * ARCHITECTURAL NOTE:
 * ====================
 * In Mitsuba, chain_distribution.h implements hierarchical sampling distributions:
 * - Bounce distribution: Samples path length (number of specular bounces)
 * - Type distribution: Selects scattering event types at each bounce
 * - Direction distribution: Guides directional sampling using learned radiance patterns
 *
 * This creates a sophisticated adaptive sampling strategy that learns from
 * previously rendered samples to improve future sampling decisions.
 *
 * In Cycles' MPG implementation, this functionality is delegated to OpenPGL:
 * - OpenPGL learns directional distributions automatically
 * - Bounce selection uses simpler heuristics (see mpg_generate_seed)
 * - Type selection (reflection vs refraction) uses Fresnel-based probabilities
 *
 * Why this file exists:
 * ---------------------
 * This stub file maintains naming parity with the Mitsuba reference to make
 * cross-referencing easier. If you're comparing Cycles with Mitsuba and see
 * references to chain_distribution.h, this file explains where that functionality
 * lives in Cycles (answer: partially in OpenPGL, partially in seed generation).
 *
 * Mitsuba's ChainDistribution class:
 * -----------------------------------
 * Manages hierarchical distributions over:
 * 1. Bounce count (1-N specular bounces)
 * 2. Scattering type at each bounce (reflection/refraction)
 * 3. Directional sampling biased by learned radiance
 *
 * Uses vMF (von Mises-Fisher) distributions or tree-based directional sampling.
 * Integrates with spatial_structure.h for nearest-neighbor lookups.
 *
 * Cycles equivalent:
 * ------------------
 * Bounce distribution:
 *   - See mpg_generate_seed() in manifold_path_guiding.cpp
 *   - Simplified version using fixed weights and optional guided biasing
 *
 * Scattering type:
 *   - See mpg_compute_dielectric_reflection_probability()
 *   - Uses Fresnel equations for physical accuracy
 *
 * Directional sampling:
 *   - OpenPGL's SurfaceSamplingDistribution handles this
 *   - See dtree.h for extracting summary statistics
 *   - pgl_estimate_summary() extracts mean direction, concentration, etc.
 *
 * Learning and adaptation:
 *   - OpenPGL handles distribution learning internally
 *   - GuidingField accumulates samples and builds distributions
 *   - See integrator/guiding.h for integration points
 *
 * If you need to understand distribution sampling in Cycles MPG:
 * 1. For bounce selection: See mpg_generate_seed() implementation
 * 2. For directional bias: See dtree.h and pgl_estimate_summary()
 * 3. For learning: Refer to OpenPGL documentation and integrator/guiding.h
 */

#pragma once

/* This file intentionally contains no code.
 * It exists purely for documentation and cross-reference with Mitsuba.
 * Chain distribution functionality is split between:
 * - Bounce/type selection: manifold_path_guiding.cpp (mpg_generate_seed)
 * - Directional learning: OpenPGL (accessed via dtree.h) */
