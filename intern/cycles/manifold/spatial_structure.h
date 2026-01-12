/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Spatial Structure - OpenPGL Integration Documentation
 *
 * This file corresponds to: spatial_structure.h in the Mitsuba MPG reference
 * https://github.com/mollnn/manifold-path-guiding/tree/main/mitsuba/src/integrators/MPG
 *
 * ARCHITECTURAL NOTE:
 * ====================
 * In Mitsuba, spatial_structure.h implements custom 6D spatial indexing
 * (k-d trees, spatial subdivision trees) for storing and querying subpath samples.
 * This enables nearest-neighbor lookups to find similar light paths.
 *
 * In Cycles' MPG implementation, this functionality is delegated to OpenPGL:
 * - OpenPGL maintains spatial data structures internally
 * - OpenPGL's SurfaceSamplingDistribution provides spatial and directional queries
 * - No explicit 6D spatial indexing is needed in Cycles
 *
 * Why this file exists:
 * ---------------------
 * This stub file maintains naming parity with the Mitsuba reference to make
 * cross-referencing easier. If you're comparing Cycles with Mitsuba and see
 * references to spatial_structure.h, this file explains where that functionality
 * lives in Cycles (answer: inside OpenPGL).
 *
 * Mitsuba's SpatialStructure classes:
 * ------------------------------------
 * - SpatialStructureANN: Uses ANN library for k-d tree nearest neighbor queries
 * - SpatialStructureSTree: Custom spatial subdivision tree
 * Both organize SubpathSample data in 6D space (xD, xL coordinates).
 *
 * Cycles equivalent:
 * ------------------
 * OpenPGL handles this internally. The GuidingField maintains the spatial
 * structure, and SurfaceSamplingDistribution provides access to it.
 * See: integrator/guiding.h for the Cycles integration.
 *
 * If you need to understand spatial queries in Cycles MPG:
 * 1. Check integrator/guiding.h for GuidingField interface
 * 2. Refer to OpenPGL documentation for spatial indexing details
 * 3. See dtree.h for how Cycles extracts directional information
 */

#pragma once

/* This file intentionally contains no code.
 * It exists purely for documentation and cross-reference with Mitsuba.
 * All spatial structure functionality is handled by OpenPGL. */
