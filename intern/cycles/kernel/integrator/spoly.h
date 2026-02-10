/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/geom/object.h"
#include "kernel/geom/triangle.h"
#include "kernel/light/sample.h"

/*
 * Specular Polynomials
 *
 * Deterministic specular path finding on triangle meshes by reformulating the specular
 * reflection/refraction constraints as bivariate polynomial equations in barycentric
 * coordinates, then solving them via Bezout resultant theory and bisection root isolation.
 *
 * Based on:
 * [1] Specular Polynomials
 * Zheng et al., SIGGRAPH 2024.
 *
 * For single-bounce reflection (R), the constraint polynomials are degree 4 in (u,v),
 * yielding a 4x4 Bezout matrix whose determinant roots give candidate v-values.
 * For single-bounce transmission (T), degree 6, yielding 6x6 Bezout matrix.
 */

CCL_NAMESPACE_BEGIN

/* Maximum polynomial degrees for single-bounce cases. */
#define SPOLY_MAX_BEZOUT_SIZE 7 /* max bivariate degree + 1 */
#define SPOLY_MAX_UNI_COEFFS 20
#define SPOLY_MAX_BVCOEFFS 7 /* degree+1 for bivariate, max T=6 -> 7 */
#define SPOLY_MAX_ROOTS 32
#define SPOLY_BISECT_ITERATIONS 20
#define SPOLY_ROOT_EPS 1e-6f
#define SPOLY_NUM_DICHOTOMY_SAMPLES 65

/* ============================================================================
 * Univariate polynomial: P(t) = c[0] + c[1]*t + ... + c[n]*t^n
 * ============================================================================ */

struct SPolyUni {
  float coeffs[SPOLY_MAX_UNI_COEFFS];
  int degree; /* highest non-zero power, -1 for zero polynomial */
};

ccl_device_inline void spoly_uni_zero(ccl_private SPolyUni *p)
{
  p->degree = -1;
  for (int i = 0; i < SPOLY_MAX_UNI_COEFFS; i++) {
    p->coeffs[i] = 0.0f;
  }
}

ccl_device_inline bool spoly_uni_is_zero(ccl_private const SPolyUni *p)
{
  if (p->degree < 0)
    return true;
  for (int i = 0; i <= p->degree; i++) {
    if (fabsf(p->coeffs[i]) > 1e-30f)
      return false;
  }
  return true;
}

ccl_device_inline float spoly_uni_eval(ccl_private const SPolyUni *p, float t)
{
  if (p->degree < 0)
    return 0.0f;
  /* Horner's method. */
  float result = p->coeffs[p->degree];
  for (int i = p->degree - 1; i >= 0; i--) {
    result = result * t + p->coeffs[i];
  }
  return result;
}

ccl_device_inline void spoly_uni_add(ccl_private SPolyUni *result,
                                     ccl_private const SPolyUni *a,
                                     ccl_private const SPolyUni *b)
{
  int deg = max(a->degree, b->degree);
  result->degree = -1;
  for (int i = 0; i <= deg && i < SPOLY_MAX_UNI_COEFFS; i++) {
    float ca = (i <= a->degree) ? a->coeffs[i] : 0.0f;
    float cb = (i <= b->degree) ? b->coeffs[i] : 0.0f;
    result->coeffs[i] = ca + cb;
    if (fabsf(result->coeffs[i]) > 1e-30f)
      result->degree = i;
  }
  for (int i = deg + 1; i < SPOLY_MAX_UNI_COEFFS; i++) {
    result->coeffs[i] = 0.0f;
  }
}

ccl_device_inline void spoly_uni_sub(ccl_private SPolyUni *result,
                                     ccl_private const SPolyUni *a,
                                     ccl_private const SPolyUni *b)
{
  int deg = max(a->degree, b->degree);
  result->degree = -1;
  for (int i = 0; i <= deg && i < SPOLY_MAX_UNI_COEFFS; i++) {
    float ca = (i <= a->degree) ? a->coeffs[i] : 0.0f;
    float cb = (i <= b->degree) ? b->coeffs[i] : 0.0f;
    result->coeffs[i] = ca - cb;
    if (fabsf(result->coeffs[i]) > 1e-30f)
      result->degree = i;
  }
  for (int i = deg + 1; i < SPOLY_MAX_UNI_COEFFS; i++) {
    result->coeffs[i] = 0.0f;
  }
}

ccl_device_inline void spoly_uni_mul(ccl_private SPolyUni *result,
                                     ccl_private const SPolyUni *a,
                                     ccl_private const SPolyUni *b)
{
  spoly_uni_zero(result);
  if (a->degree < 0 || b->degree < 0)
    return;
  int deg = a->degree + b->degree;
  if (deg >= SPOLY_MAX_UNI_COEFFS)
    deg = SPOLY_MAX_UNI_COEFFS - 1;
  for (int i = 0; i <= a->degree; i++) {
    for (int j = 0; j <= b->degree; j++) {
      if (i + j < SPOLY_MAX_UNI_COEFFS) {
        result->coeffs[i + j] += a->coeffs[i] * b->coeffs[j];
      }
    }
  }
  result->degree = -1;
  for (int i = deg; i >= 0; i--) {
    if (fabsf(result->coeffs[i]) > 1e-30f) {
      result->degree = i;
      break;
    }
  }
}

ccl_device_inline void spoly_uni_scale(ccl_private SPolyUni *p, float s)
{
  if (p->degree < 0)
    return;
  for (int i = 0; i <= p->degree; i++) {
    p->coeffs[i] *= s;
  }
}

/* Set to constant value 1 (identity for multiplication). */
ccl_device_inline void spoly_uni_set_one(ccl_private SPolyUni *p)
{
  spoly_uni_zero(p);
  p->coeffs[0] = 1.0f;
  p->degree = 0;
}

/* ============================================================================
 * Bivariate polynomial: P(u,v) = sum_{i,j} c[i][j] * u^i * v^j
 * where i is the u-power and j is the v-power.
 * ============================================================================ */

struct SPolyBiv {
  float coeffs[SPOLY_MAX_BVCOEFFS][SPOLY_MAX_BVCOEFFS];
  int degree_u; /* max power of u */
  int degree_v; /* max power of v */
};

ccl_device_inline void spoly_biv_zero(ccl_private SPolyBiv *p)
{
  p->degree_u = -1;
  p->degree_v = -1;
  for (int i = 0; i < SPOLY_MAX_BVCOEFFS; i++) {
    for (int j = 0; j < SPOLY_MAX_BVCOEFFS; j++) {
      p->coeffs[i][j] = 0.0f;
    }
  }
}

ccl_device_inline void spoly_biv_set_const(ccl_private SPolyBiv *p, float val)
{
  spoly_biv_zero(p);
  p->coeffs[0][0] = val;
  if (fabsf(val) > 1e-30f) {
    p->degree_u = 0;
    p->degree_v = 0;
  }
}

/* Set P(u,v) = a + b*u + c*v (linear). */
ccl_device_inline void spoly_biv_set_linear(ccl_private SPolyBiv *p,
                                            float a,
                                            float b,
                                            float c)
{
  spoly_biv_zero(p);
  p->coeffs[0][0] = a;
  p->coeffs[1][0] = b;
  p->coeffs[0][1] = c;
  p->degree_u = (fabsf(b) > 1e-30f) ? 1 : ((fabsf(a) > 1e-30f) ? 0 : -1);
  p->degree_v = (fabsf(c) > 1e-30f) ? 1 : ((fabsf(a) > 1e-30f) ? 0 : -1);
}

ccl_device_inline void spoly_biv_update_degrees(ccl_private SPolyBiv *p)
{
  p->degree_u = -1;
  p->degree_v = -1;
  for (int i = 0; i < SPOLY_MAX_BVCOEFFS; i++) {
    for (int j = 0; j < SPOLY_MAX_BVCOEFFS; j++) {
      if (fabsf(p->coeffs[i][j]) > 1e-30f) {
        if (i > p->degree_u)
          p->degree_u = i;
        if (j > p->degree_v)
          p->degree_v = j;
      }
    }
  }
}

ccl_device_inline void spoly_biv_add(ccl_private SPolyBiv *result,
                                     ccl_private const SPolyBiv *a,
                                     ccl_private const SPolyBiv *b)
{
  for (int i = 0; i < SPOLY_MAX_BVCOEFFS; i++) {
    for (int j = 0; j < SPOLY_MAX_BVCOEFFS; j++) {
      result->coeffs[i][j] = a->coeffs[i][j] + b->coeffs[i][j];
    }
  }
  spoly_biv_update_degrees(result);
}

ccl_device_inline void spoly_biv_sub(ccl_private SPolyBiv *result,
                                     ccl_private const SPolyBiv *a,
                                     ccl_private const SPolyBiv *b)
{
  for (int i = 0; i < SPOLY_MAX_BVCOEFFS; i++) {
    for (int j = 0; j < SPOLY_MAX_BVCOEFFS; j++) {
      result->coeffs[i][j] = a->coeffs[i][j] - b->coeffs[i][j];
    }
  }
  spoly_biv_update_degrees(result);
}

ccl_device_inline void spoly_biv_mul(ccl_private SPolyBiv *result,
                                     ccl_private const SPolyBiv *a,
                                     ccl_private const SPolyBiv *b)
{
  spoly_biv_zero(result);
  if (a->degree_u < 0 || b->degree_u < 0)
    return;
  for (int i1 = 0; i1 <= a->degree_u; i1++) {
    for (int j1 = 0; j1 <= a->degree_v; j1++) {
      if (fabsf(a->coeffs[i1][j1]) < 1e-30f)
        continue;
      for (int i2 = 0; i2 <= b->degree_u; i2++) {
        for (int j2 = 0; j2 <= b->degree_v; j2++) {
          if (i1 + i2 < SPOLY_MAX_BVCOEFFS && j1 + j2 < SPOLY_MAX_BVCOEFFS) {
            result->coeffs[i1 + i2][j1 + j2] += a->coeffs[i1][j1] * b->coeffs[i2][j2];
          }
        }
      }
    }
  }
  spoly_biv_update_degrees(result);
}

ccl_device_inline void spoly_biv_scale(ccl_private SPolyBiv *p, float s)
{
  for (int i = 0; i < SPOLY_MAX_BVCOEFFS; i++) {
    for (int j = 0; j < SPOLY_MAX_BVCOEFFS; j++) {
      p->coeffs[i][j] *= s;
    }
  }
}

/* Divide all coefficients by max * 1e-6 (matching reference normalization).
 * The reference uses findMax() (raw max, not abs) scaled by 1e-6.
 * This effectively upscales coefficients for better numerical conditioning. */
ccl_device_inline void spoly_biv_divide_by_max(ccl_private SPolyBiv *p)
{
  float max_val = p->coeffs[0][0];
  for (int i = 0; i < SPOLY_MAX_BVCOEFFS; i++) {
    for (int j = 0; j < SPOLY_MAX_BVCOEFFS; j++) {
      if (p->coeffs[i][j] > max_val)
        max_val = p->coeffs[i][j];
    }
  }
  float divisor = max_val * 1e-6f;
  if (divisor != 0.0f) {
    float inv = 1.0f / divisor;
    for (int i = 0; i < SPOLY_MAX_BVCOEFFS; i++) {
      for (int j = 0; j < SPOLY_MAX_BVCOEFFS; j++) {
        p->coeffs[i][j] *= inv;
      }
    }
  }
}

/* Evaluate at a specific v, producing a univariate polynomial in u. */
ccl_device_inline void spoly_biv_eval_at_v(ccl_private const SPolyBiv *p,
                                           float v,
                                           ccl_private SPolyUni *result)
{
  spoly_uni_zero(result);
  if (p->degree_u < 0)
    return;
  for (int i = 0; i <= p->degree_u && i < SPOLY_MAX_BVCOEFFS; i++) {
    float val = 0.0f;
    float vpow = 1.0f;
    for (int j = 0; j <= p->degree_v && j < SPOLY_MAX_BVCOEFFS; j++) {
      val += p->coeffs[i][j] * vpow;
      vpow *= v;
    }
    result->coeffs[i] = val;
    if (fabsf(val) > 1e-30f)
      result->degree = i;
  }
}

/* Extract row i as a univariate polynomial in v (coefficients of u^i).
 * This matches the reference toUnivariatePolynomials(): row i has coefficients
 * coeffs[i][0..n-i-1] where n is the bivariate size. */
ccl_device_inline void spoly_biv_row_as_uni(ccl_private const SPolyBiv *p,
                                            int row,
                                            int biv_size,
                                            ccl_private SPolyUni *result)
{
  spoly_uni_zero(result);
  int max_j = biv_size - row;
  if (max_j > SPOLY_MAX_UNI_COEFFS)
    max_j = SPOLY_MAX_UNI_COEFFS;
  if (row < 0 || row >= SPOLY_MAX_BVCOEFFS)
    return;
  for (int j = 0; j < max_j && j < SPOLY_MAX_BVCOEFFS; j++) {
    result->coeffs[j] = p->coeffs[row][j];
    if (fabsf(result->coeffs[j]) > 1e-30f)
      result->degree = j;
  }
}

/* ============================================================================
 * BVP3: Vector of 3 bivariate polynomials (x, y, z components).
 * ============================================================================ */

struct SPolyBVP3 {
  SPolyBiv x, y, z;
};

ccl_device_inline void spoly_bvp3_zero(ccl_private SPolyBVP3 *p)
{
  spoly_biv_zero(&p->x);
  spoly_biv_zero(&p->y);
  spoly_biv_zero(&p->z);
}

ccl_device_inline void spoly_bvp3_set_const(ccl_private SPolyBVP3 *p, float3 v)
{
  spoly_biv_set_const(&p->x, v.x);
  spoly_biv_set_const(&p->y, v.y);
  spoly_biv_set_const(&p->z, v.z);
}

/* Set to barycentric: v0 + (v1-v0)*u + (v2-v0)*v. */
ccl_device_inline void spoly_bvp3_set_barycentric(ccl_private SPolyBVP3 *p,
                                                  float3 v0,
                                                  float3 v1,
                                                  float3 v2)
{
  spoly_biv_set_linear(&p->x, v0.x, v1.x - v0.x, v2.x - v0.x);
  spoly_biv_set_linear(&p->y, v0.y, v1.y - v0.y, v2.y - v0.y);
  spoly_biv_set_linear(&p->z, v0.z, v1.z - v0.z, v2.z - v0.z);
}

ccl_device_inline void spoly_bvp3_sub(ccl_private SPolyBVP3 *result,
                                      ccl_private const SPolyBVP3 *a,
                                      ccl_private const SPolyBVP3 *b)
{
  spoly_biv_sub(&result->x, &a->x, &b->x);
  spoly_biv_sub(&result->y, &a->y, &b->y);
  spoly_biv_sub(&result->z, &a->z, &b->z);
}

ccl_device_inline void spoly_bvp3_add(ccl_private SPolyBVP3 *result,
                                      ccl_private const SPolyBVP3 *a,
                                      ccl_private const SPolyBVP3 *b)
{
  spoly_biv_add(&result->x, &a->x, &b->x);
  spoly_biv_add(&result->y, &a->y, &b->y);
  spoly_biv_add(&result->z, &a->z, &b->z);
}

ccl_device_inline void spoly_bvp3_dot(ccl_private SPolyBiv *result,
                                      ccl_private const SPolyBVP3 *a,
                                      ccl_private const SPolyBVP3 *b)
{
  SPolyBiv tx, ty, tz, tmp;
  spoly_biv_mul(&tx, &a->x, &b->x);
  spoly_biv_mul(&ty, &a->y, &b->y);
  spoly_biv_mul(&tz, &a->z, &b->z);
  spoly_biv_add(&tmp, &tx, &ty);
  spoly_biv_add(result, &tmp, &tz);
}

ccl_device_inline void spoly_bvp3_cross(ccl_private SPolyBVP3 *result,
                                        ccl_private const SPolyBVP3 *a,
                                        ccl_private const SPolyBVP3 *b)
{
  SPolyBiv t1, t2;
  spoly_biv_mul(&t1, &a->y, &b->z);
  spoly_biv_mul(&t2, &a->z, &b->y);
  spoly_biv_sub(&result->x, &t1, &t2);
  spoly_biv_mul(&t1, &a->z, &b->x);
  spoly_biv_mul(&t2, &a->x, &b->z);
  spoly_biv_sub(&result->y, &t1, &t2);
  spoly_biv_mul(&t1, &a->x, &b->y);
  spoly_biv_mul(&t2, &a->y, &b->x);
  spoly_biv_sub(&result->z, &t1, &t2);
}

/* Select the BVP3 component with the largest max absolute coefficient.
 * This avoids the degeneracy of always projecting to x which can fail
 * for certain triangle orientations. */
ccl_device_inline void spoly_bvp3_best_component(ccl_private const SPolyBVP3 *p,
                                                  ccl_private SPolyBiv *result)
{
  float max_x = 0.0f, max_y = 0.0f, max_z = 0.0f;
  for (int i = 0; i < SPOLY_MAX_BVCOEFFS; i++) {
    for (int j = 0; j < SPOLY_MAX_BVCOEFFS; j++) {
      float ax = fabsf(p->x.coeffs[i][j]);
      float ay = fabsf(p->y.coeffs[i][j]);
      float az = fabsf(p->z.coeffs[i][j]);
      if (ax > max_x)
        max_x = ax;
      if (ay > max_y)
        max_y = ay;
      if (az > max_z)
        max_z = az;
    }
  }
  if (max_x >= max_y && max_x >= max_z) {
    *result = p->x;
  }
  else if (max_y >= max_z) {
    *result = p->y;
  }
  else {
    *result = p->z;
  }
}

/* Scalar-multiply a BVP3 by a bivariate polynomial. */
ccl_device_inline void spoly_bvp3_mul_biv(ccl_private SPolyBVP3 *result,
                                          ccl_private const SPolyBVP3 *a,
                                          ccl_private const SPolyBiv *b)
{
  spoly_biv_mul(&result->x, &a->x, b);
  spoly_biv_mul(&result->y, &a->y, b);
  spoly_biv_mul(&result->z, &a->z, b);
}

/* ============================================================================
 * Bezout Matrix and Determinant
 *
 * The Bezout matrix eliminates u from Czy(u,v) and Cxz(u,v).
 * Each bivariate polynomial is viewed as a univariate in u with coefficients
 * that are univariate polynomials in v (extracted via toUnivariatePolynomials).
 * ============================================================================ */

/* Evaluate an n×n matrix of univariate polynomials at a specific v value,
 * producing a plain n×n float matrix. Then compute its determinant via
 * Gaussian elimination. This is the DICHOTOMY method from the reference. */
ccl_device_inline float spoly_eval_bezout_det_at_v(
    ccl_private SPolyUni bezout[][SPOLY_MAX_BEZOUT_SIZE],
    int n,
    float v)
{
  float mat[SPOLY_MAX_BEZOUT_SIZE][SPOLY_MAX_BEZOUT_SIZE];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat[i][j] = spoly_uni_eval(&bezout[i][j], v);
    }
  }

  /* Gaussian elimination with partial pivoting. */
  float det = 1.0f;
  for (int col = 0; col < n; col++) {
    int pivot = -1;
    float max_val = 0.0f;
    for (int row = col; row < n; row++) {
      float av = fabsf(mat[row][col]);
      if (av > max_val) {
        max_val = av;
        pivot = row;
      }
    }
    if (pivot < 0 || max_val < 1e-20f)
      return 0.0f;

    if (pivot != col) {
      for (int j = 0; j < n; j++) {
        float t = mat[col][j];
        mat[col][j] = mat[pivot][j];
        mat[pivot][j] = t;
      }
      det = -det;
    }
    det *= mat[col][col];
    float inv_pivot = 1.0f / mat[col][col];

    for (int row = col + 1; row < n; row++) {
      float factor = mat[row][col] * inv_pivot;
      for (int j = col + 1; j < n; j++) {
        mat[row][j] -= factor * mat[col][j];
      }
    }
  }
  return det;
}

/* Build the Bezout matrix from two bivariate polynomials.
 * Matches the reference implementation exactly:
 *   Step 1: f[i][j] = a[i]*b[j+1] - b[i]*a[j+1]  (upper triangle)
 *   Step 2: f[i][j] += f[i-1][j+1]                (delta-Bezout accumulation)
 *   Step 3: f[i][j] = f[j][i]                      (symmetrize)
 *   Step 4: Pad zero rows with identity             (numerical stability)
 */
ccl_device_inline int spoly_build_bezout(ccl_private const SPolyBiv *poly_f,
                                         ccl_private const SPolyBiv *poly_g,
                                         ccl_private SPolyUni bezout[][SPOLY_MAX_BEZOUT_SIZE])
{
  /* Determine the bivariate polynomial size.
   * n = max(degree_u of Czy, degree_u of Cxz) + 1 (number of rows). */
  int size = max(poly_f->degree_u, poly_g->degree_u) + 1;
  if (size <= 1)
    return 0;
  if (size > SPOLY_MAX_BEZOUT_SIZE)
    size = SPOLY_MAX_BEZOUT_SIZE;

  /* Extract rows: a[i] is the coefficient of u^i as a univariate polynomial in v.
   * This matches toUnivariatePolynomials() from the reference. */
  SPolyUni a[SPOLY_MAX_BEZOUT_SIZE + 1];
  SPolyUni b[SPOLY_MAX_BEZOUT_SIZE + 1];
  for (int i = 0; i <= size; i++) {
    spoly_biv_row_as_uni(poly_f, i, size, &a[i]);
    spoly_biv_row_as_uni(poly_g, i, size, &b[i]);
  }

  int n = size - 1; /* Bezout matrix is (n x n) where n = degree */
  if (n <= 0)
    return 0;
  if (n > SPOLY_MAX_BEZOUT_SIZE)
    n = SPOLY_MAX_BEZOUT_SIZE;

  /* Initialize all entries to zero. */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      spoly_uni_zero(&bezout[i][j]);
    }
  }

  /* Step 1: Basic Bezout entries (upper triangle).
   * f[i][j] = a[i] * b[j+1] - b[i] * a[j+1] */
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      SPolyUni prod1, prod2;
      spoly_uni_mul(&prod1, &a[i], &b[j + 1]);
      spoly_uni_mul(&prod2, &b[i], &a[j + 1]);
      spoly_uni_sub(&bezout[i][j], &prod1, &prod2);
    }
  }

  /* Step 2: Delta-Bezout accumulation.
   * f[i][j] += f[i-1][j+1] for i >= 1, j < n-1 */
  for (int i = 1; i < n - 1; i++) {
    for (int j = i; j < n - 1; j++) {
      SPolyUni sum;
      spoly_uni_add(&sum, &bezout[i][j], &bezout[i - 1][j + 1]);
      bezout[i][j] = sum;
    }
  }

  /* Step 3: Symmetrize (fill lower triangle). */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      bezout[i][j] = bezout[j][i];
    }
  }

  /* Step 4: Find the actual size (last non-zero row/col) and pad remainder
   * with identity entries for numerical stability. */
  int m = -1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      if (!spoly_uni_is_zero(&bezout[i][j])) {
        m = max(m, max(i, j));
      }
    }
  }
  for (int k = m + 1; k < n; k++) {
    spoly_uni_set_one(&bezout[k][k]);
  }

  return n;
}

/* ============================================================================
 * Root finding via DICHOTOMY method.
 *
 * Instead of computing the resultant polynomial symbolically, we directly
 * evaluate det(Bezout matrix) at sample points in [0,1] and detect sign
 * changes, then bisect to find roots. This is more numerically stable.
 * ============================================================================ */

ccl_device_inline int spoly_find_roots_dichotomy(
    ccl_private SPolyUni bezout[][SPOLY_MAX_BEZOUT_SIZE],
    int n,
    ccl_private float roots[SPOLY_MAX_ROOTS])
{
  int num_roots = 0;
  const int num_samples = SPOLY_NUM_DICHOTOMY_SAMPLES;

  float prev_val = spoly_eval_bezout_det_at_v(bezout, n, 0.0f);

  if (fabsf(prev_val) < 1e-30f) {
    if (num_roots < SPOLY_MAX_ROOTS) {
      roots[num_roots++] = 0.0f;
    }
  }

  for (int s = 1; s < num_samples; s++) {
    float v = (float)s / (float)(num_samples - 1);
    float val = spoly_eval_bezout_det_at_v(bezout, n, v);

    if (fabsf(val) < 1e-30f) {
      if (num_roots < SPOLY_MAX_ROOTS) {
        roots[num_roots++] = v;
      }
    }
    else if (prev_val * val < 0.0f && fabsf(prev_val) > 1e-30f) {
      /* Sign change detected - bisect to find root. */
      float lo = (float)(s - 1) / (float)(num_samples - 1);
      float hi = v;
      float lo_val = prev_val;

      for (int iter = 0; iter < SPOLY_BISECT_ITERATIONS; iter++) {
        float mid = 0.5f * (lo + hi);
        float mid_val = spoly_eval_bezout_det_at_v(bezout, n, mid);
        if (mid_val * val < 0.0f) {
          lo = mid;
          lo_val = mid_val;
        }
        else {
          hi = mid;
        }
        (void)lo_val;
      }

      float root = 0.5f * (lo + hi);
      if (root >= -SPOLY_ROOT_EPS && root <= 1.0f + SPOLY_ROOT_EPS) {
        if (num_roots < SPOLY_MAX_ROOTS) {
          roots[num_roots++] = clamp(root, 0.0f, 1.0f);
        }
      }
    }

    prev_val = val;
  }

  return num_roots;
}

/* Root finding in [0,1] for a univariate polynomial (for back-substitution). */
ccl_device_inline int spoly_find_roots_in_01(ccl_private const SPolyUni *poly,
                                             ccl_private float roots[SPOLY_MAX_ROOTS])
{
  int num_roots = 0;
  const int num_samples = SPOLY_NUM_DICHOTOMY_SAMPLES;

  float prev_val = spoly_uni_eval(poly, 0.0f);

  for (int s = 1; s < num_samples; s++) {
    float v = (float)s / (float)(num_samples - 1);
    float val = spoly_uni_eval(poly, v);

    if (prev_val * val <= 0.0f && (fabsf(prev_val) > 1e-30f || fabsf(val) > 1e-30f)) {
      float lo = (float)(s - 1) / (float)(num_samples - 1);
      float hi = v;
      float lo_val = prev_val;

      for (int iter = 0; iter < SPOLY_BISECT_ITERATIONS; iter++) {
        float mid = 0.5f * (lo + hi);
        float mid_val = spoly_uni_eval(poly, mid);
        if (lo_val * mid_val <= 0.0f) {
          hi = mid;
        }
        else {
          lo = mid;
          lo_val = mid_val;
        }
      }

      float root = 0.5f * (lo + hi);
      if (root >= -SPOLY_ROOT_EPS && root <= 1.0f + SPOLY_ROOT_EPS) {
        if (num_roots < SPOLY_MAX_ROOTS) {
          roots[num_roots++] = clamp(root, 0.0f, 1.0f);
        }
      }
    }

    prev_val = val;
  }

  return num_roots;
}

/* ============================================================================
 * Specular constraint formulation for reflection (R), chain_type = 1.
 *
 * From the reference (resultant.h lines 102-121):
 *   Czy = (d0·n_hat) * (d1·t_hat2) + (d0·t_hat2) * (d1·n_hat)
 *   s = xL - xD
 *   cop = (d0 × s) × (n_hat × s)
 *   Cxz = cop.x  (projected to x-axis)
 *
 * where t_hat2 = n_hat × p12 (p12 is the second edge vector).
 * ============================================================================ */

ccl_device_inline void spoly_build_reflection_constraints(float3 xD,
                                                          float3 xL,
                                                          float3 P0,
                                                          float3 P1,
                                                          float3 P2,
                                                          float3 N0,
                                                          float3 N1,
                                                          float3 N2,
                                                          ccl_private SPolyBiv *Czy,
                                                          ccl_private SPolyBiv *Cxz)
{
  SPolyBVP3 x1;
  spoly_bvp3_set_barycentric(&x1, P0, P1, P2);

  SPolyBVP3 n1_hat;
  spoly_bvp3_set_barycentric(&n1_hat, N0, N1, N2);

  SPolyBVP3 xD_bvp, xL_bvp;
  spoly_bvp3_set_const(&xD_bvp, xD);
  spoly_bvp3_set_const(&xL_bvp, xL);

  SPolyBVP3 d0, d1;
  spoly_bvp3_sub(&d0, &x1, &xD_bvp);
  spoly_bvp3_sub(&d1, &xL_bvp, &x1);

  /* Edge vector p12 (second edge). */
  float3 e2 = P2 - P0;
  SPolyBVP3 e2_bvp;
  spoly_bvp3_set_const(&e2_bvp, e2);

  /* Tangent: t_hat2 = n_hat × p12. */
  SPolyBVP3 t_hat2;
  spoly_bvp3_cross(&t_hat2, &n1_hat, &e2_bvp);

  /* Dot products for Czy. */
  SPolyBiv d0_dot_n, d1_dot_n, d0_dot_t2, d1_dot_t2;
  spoly_bvp3_dot(&d0_dot_n, &d0, &n1_hat);
  spoly_bvp3_dot(&d1_dot_n, &d1, &n1_hat);
  spoly_bvp3_dot(&d0_dot_t2, &d0, &t_hat2);
  spoly_bvp3_dot(&d1_dot_t2, &d1, &t_hat2);

  /* Czy = d0·n * d1·t2 + d0·t2 * d1·n */
  SPolyBiv term1, term2;
  spoly_biv_mul(&term1, &d0_dot_n, &d1_dot_t2);
  spoly_biv_mul(&term2, &d0_dot_t2, &d1_dot_n);
  spoly_biv_add(Czy, &term1, &term2);

  /* Cxz: cop = (d0 × s) × (n_hat × s), project to best axis.
   * s = xL - xD (constant vector).
   * The reference always uses x-axis but warns it can fail for some orientations.
   * We pick the component with the largest magnitude for robustness. */
  float3 s = xL - xD;
  SPolyBVP3 s_bvp;
  spoly_bvp3_set_const(&s_bvp, s);

  SPolyBVP3 d0_cross_s, n_cross_s, cop;
  spoly_bvp3_cross(&d0_cross_s, &d0, &s_bvp);
  spoly_bvp3_cross(&n_cross_s, &n1_hat, &s_bvp);
  spoly_bvp3_cross(&cop, &d0_cross_s, &n_cross_s);
  spoly_bvp3_best_component(&cop, Cxz);
}

/* ============================================================================
 * Specular constraint formulation for transmission (T), chain_type = 2.
 *
 * From the reference (resultant.h lines 123-141):
 *   c0 = d0 × n_hat, c1 = d1 × n_hat
 *   c = c0*c0*|d1|^2*eta^2 - c1*c1*|d0|^2
 *   Czy = c.x  (x-component of the BVP3)
 *   s = xL - xD
 *   cop = (d0 × s) × (n_hat × s)
 *   Cxz = cop.x
 * ============================================================================ */

ccl_device_inline void spoly_build_refraction_constraints(float3 xD,
                                                          float3 xL,
                                                          float3 P0,
                                                          float3 P1,
                                                          float3 P2,
                                                          float3 N0,
                                                          float3 N1,
                                                          float3 N2,
                                                          float eta,
                                                          ccl_private SPolyBiv *Czy,
                                                          ccl_private SPolyBiv *Cxz)
{
  SPolyBVP3 x1;
  spoly_bvp3_set_barycentric(&x1, P0, P1, P2);

  SPolyBVP3 n1_hat;
  spoly_bvp3_set_barycentric(&n1_hat, N0, N1, N2);

  SPolyBVP3 xD_bvp, xL_bvp;
  spoly_bvp3_set_const(&xD_bvp, xD);
  spoly_bvp3_set_const(&xL_bvp, xL);

  SPolyBVP3 d0, d1;
  spoly_bvp3_sub(&d0, &x1, &xD_bvp);
  spoly_bvp3_sub(&d1, &xL_bvp, &x1);

  /* c0 = d0 × n_hat, c1 = d1 × n_hat. */
  SPolyBVP3 c0, c1;
  spoly_bvp3_cross(&c0, &d0, &n1_hat);
  spoly_bvp3_cross(&c1, &d1, &n1_hat);

  /* |d0|^2, |d1|^2. */
  SPolyBiv d0_sq, d1_sq;
  spoly_bvp3_dot(&d0_sq, &d0, &d0);
  spoly_bvp3_dot(&d1_sq, &d1, &d1);

  /* c = c0*c0*d1_norm2*eta^2 - c1*c1*d0_norm2
   * Here c0*c0 means element-wise: BVP3 where each component is c0.comp * c0.comp.
   * This is NOT a dot product; it's component-wise multiplication yielding a BVP3. */
  float eta2 = eta * eta;
  SPolyBVP3 c0_sq_bvp, c1_sq_bvp;
  spoly_biv_mul(&c0_sq_bvp.x, &c0.x, &c0.x);
  spoly_biv_mul(&c0_sq_bvp.y, &c0.y, &c0.y);
  spoly_biv_mul(&c0_sq_bvp.z, &c0.z, &c0.z);
  spoly_biv_mul(&c1_sq_bvp.x, &c1.x, &c1.x);
  spoly_biv_mul(&c1_sq_bvp.y, &c1.y, &c1.y);
  spoly_biv_mul(&c1_sq_bvp.z, &c1.z, &c1.z);

  /* c0*c0 * d1_norm2 * eta^2 */
  SPolyBVP3 term1_bvp;
  spoly_bvp3_mul_biv(&term1_bvp, &c0_sq_bvp, &d1_sq);
  spoly_biv_scale(&term1_bvp.x, eta2);
  spoly_biv_scale(&term1_bvp.y, eta2);
  spoly_biv_scale(&term1_bvp.z, eta2);

  /* c1*c1 * d0_norm2 */
  SPolyBVP3 term2_bvp;
  spoly_bvp3_mul_biv(&term2_bvp, &c1_sq_bvp, &d0_sq);

  /* c = term1 - term2 */
  SPolyBVP3 c;
  spoly_bvp3_sub(&c, &term1_bvp, &term2_bvp);

  /* Czy: pick best component of c (reference uses c.bvp[0] = x). */
  spoly_bvp3_best_component(&c, Czy);

  /* Cxz: same cop formulation as reflection, best component. */
  float3 s = xL - xD;
  SPolyBVP3 s_bvp;
  spoly_bvp3_set_const(&s_bvp, s);
  SPolyBVP3 d0_cross_s, n_cross_s, cop;
  spoly_bvp3_cross(&d0_cross_s, &d0, &s_bvp);
  spoly_bvp3_cross(&n_cross_s, &n1_hat, &s_bvp);
  spoly_bvp3_cross(&cop, &d0_cross_s, &n_cross_s);
  spoly_bvp3_best_component(&cop, Cxz);
}

/* ============================================================================
 * Main solver: find specular points on a triangle.
 * ============================================================================ */

struct SPolySolution {
  float u, v;
  float3 position;
};

ccl_device_inline int spoly_solve(float3 xD,
                                  float3 xL,
                                  float3 P0,
                                  float3 P1,
                                  float3 P2,
                                  float3 N0,
                                  float3 N1,
                                  float3 N2,
                                  bool is_refraction,
                                  float eta,
                                  ccl_private SPolySolution solutions[SPOLY_MAX_ROOTS])
{
  SPolyBiv Czy, Cxz;

  if (is_refraction) {
    spoly_build_refraction_constraints(xD, xL, P0, P1, P2, N0, N1, N2, eta, &Czy, &Cxz);
  }
  else {
    spoly_build_reflection_constraints(xD, xL, P0, P1, P2, N0, N1, N2, &Czy, &Cxz);
  }

  /* Normalize constraints for numerical stability. */
  spoly_biv_divide_by_max(&Czy);
  spoly_biv_divide_by_max(&Cxz);

  /* Build Bezout matrix. */
  SPolyUni bezout[SPOLY_MAX_BEZOUT_SIZE][SPOLY_MAX_BEZOUT_SIZE];
  int n = spoly_build_bezout(&Czy, &Cxz, bezout);
  if (n <= 0)
    return 0;

  /* Find v-roots via DICHOTOMY method (sign-change detection + bisection
   * on numerical determinant evaluations). */
  float v_roots[SPOLY_MAX_ROOTS];
  int num_v_roots = spoly_find_roots_dichotomy(bezout, n, v_roots);

  int num_solutions = 0;

  /* Back-substitution: for each v-root, solve Cxz(u, v) = 0 for u. */
  for (int ri = 0; ri < num_v_roots; ri++) {
    float v = v_roots[ri];

    SPolyUni cxz_u;
    spoly_biv_eval_at_v(&Cxz, v, &cxz_u);

    float u_roots[SPOLY_MAX_ROOTS];
    int num_u_roots = spoly_find_roots_in_01(&cxz_u, u_roots);

    for (int ui = 0; ui < num_u_roots; ui++) {
      float u = u_roots[ui];

      /* Check barycentric constraint: u + v <= 1. */
      if (u < -SPOLY_ROOT_EPS || v < -SPOLY_ROOT_EPS || (u + v) > 1.0f + SPOLY_ROOT_EPS)
        continue;

      u = clamp(u, 0.0f, 1.0f);
      v = clamp(v, 0.0f, 1.0f - u);

      /* Verify the other constraint is also satisfied. */
      SPolyUni czy_u;
      spoly_biv_eval_at_v(&Czy, v, &czy_u);
      float czy_val = spoly_uni_eval(&czy_u, u);
      float cxz_val = spoly_uni_eval(&cxz_u, u);

      /* Tolerance: BOTH constraints must be near zero for a valid solution.
       * Use relative tolerance since divideByMax upscales coefficients. */
      float max_residual = fmaxf(fabsf(czy_val), fabsf(cxz_val));
      if (max_residual > 1.0f)
        continue;

      if (num_solutions < SPOLY_MAX_ROOTS) {
        float w = 1.0f - u - v;
        solutions[num_solutions].u = u;
        solutions[num_solutions].v = v;
        solutions[num_solutions].position = w * P0 + u * P1 + v * P2;
        num_solutions++;
      }
    }
  }

  return num_solutions;
}

/* ============================================================================
 * 4-ary Tree Node Access Helpers
 *
 * Each node occupies 4 float4s in the spoly_tree_nodes array.
 * Layout:
 *   [node*4+0] = (pos_min.x, pos_min.y, pos_min.z, pos_max.x)
 *   [node*4+1] = (pos_max.y, pos_max.z, nor_min.x, nor_min.y)
 *   [node*4+2] = (nor_min.z, nor_max.x, nor_max.y, nor_max.z)
 *   [node*4+3] = (child_offset, num_children, triangle_prim, pos_area)
 *                 (first 3 are int-as-float encoded)
 * ============================================================================ */

struct SPolyTreeNode {
  float3 pos_min, pos_max;
  float3 nor_min, nor_max;
  int child_offset;
  int num_children;
  int triangle_prim;
  float pos_area;
};

ccl_device_inline SPolyTreeNode spoly_tree_node_fetch(KernelGlobals kg, int node_idx)
{
  const int base = node_idx * 4;
  const float4 n0 = kernel_data_fetch(spoly_tree_nodes, base + 0);
  const float4 n1 = kernel_data_fetch(spoly_tree_nodes, base + 1);
  const float4 n2 = kernel_data_fetch(spoly_tree_nodes, base + 2);
  const float4 n3 = kernel_data_fetch(spoly_tree_nodes, base + 3);

  SPolyTreeNode node;
  node.pos_min = make_float3(n0.x, n0.y, n0.z);
  node.pos_max = make_float3(n0.w, n1.x, n1.y);
  node.nor_min = make_float3(n1.z, n1.w, n2.x);
  node.nor_max = make_float3(n2.y, n2.z, n2.w);
  node.child_offset = __float_as_int(n3.x);
  node.num_children = __float_as_int(n3.y);
  node.triangle_prim = __float_as_int(n3.z);
  node.pos_area = n3.w;
  return node;
}

/* ============================================================================
 * Interval Arithmetic for Conservative Pruning
 *
 * Interval1D represented as float2(lo, hi).
 * Used for hemisphere and half-vector pruning checks on tree nodes.
 * ============================================================================ */

ccl_device_inline float2 spoly_iv_sub(float2 a, float2 b)
{
  return make_float2(a.x - b.y, a.y - b.x);
}

ccl_device_inline float2 spoly_iv_mul(float2 a, float2 b)
{
  const float p1 = a.x * b.x, p2 = a.x * b.y, p3 = a.y * b.x, p4 = a.y * b.y;
  return make_float2(fminf(fminf(p1, p2), fminf(p3, p4)),
                     fmaxf(fmaxf(p1, p2), fmaxf(p3, p4)));
}

ccl_device_inline float2 spoly_iv_add(float2 a, float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}

/* Interval dot product of two Interval3D vectors.
 * Each Interval3D is 3 float2 intervals (lo, hi). */
ccl_device_inline float2 spoly_iv3_dot(float2 ax,
                                        float2 ay,
                                        float2 az,
                                        float2 bx,
                                        float2 by,
                                        float2 bz)
{
  return spoly_iv_add(spoly_iv_add(spoly_iv_mul(ax, bx), spoly_iv_mul(ay, by)),
                      spoly_iv_mul(az, bz));
}

/* Check if a tree node could contain a valid single-bounce reflection path.
 *
 * Pruning checks (conservative — may allow false positives but no false negatives):
 * 1. Hemisphere test (receiver): dot(norBox, recv_P - posBox).max > threshold
 * 2. Hemisphere test (light):    dot(norBox, light_P - posBox).max > threshold
 * 3. Half-vector alignment: max dot(H_center, norBox) > threshold - angular_margin
 *
 * Returns true if the node passes all checks (cannot be pruned). */
ccl_device_inline bool spoly_tree_node_valid_reflection(const SPolyTreeNode &node,
                                                         const float3 recv_P,
                                                         const float3 light_P)
{
  /* Normal interval. */
  const float2 nx = make_float2(node.nor_min.x, node.nor_max.x);
  const float2 ny = make_float2(node.nor_min.y, node.nor_max.y);
  const float2 nz = make_float2(node.nor_min.z, node.nor_max.z);

  /* Direction interval from posBox to receiver: recv_P - posBox.
   * For a point P and interval [lo, hi]: P - [lo, hi] = [P - hi, P - lo]. */
  const float2 drx = make_float2(recv_P.x - node.pos_max.x, recv_P.x - node.pos_min.x);
  const float2 dry = make_float2(recv_P.y - node.pos_max.y, recv_P.y - node.pos_min.y);
  const float2 drz = make_float2(recv_P.z - node.pos_max.z, recv_P.z - node.pos_min.z);

  /* Check 1: Hemisphere test — some normal must face the receiver.
   * Use a small positive threshold to reject grazing configurations. */
  const float2 dot_recv = spoly_iv3_dot(nx, ny, nz, drx, dry, drz);
  if (dot_recv.y <= 0.001f) {
    return false;
  }

  /* Direction interval from posBox to light. */
  const float2 dlx = make_float2(light_P.x - node.pos_max.x, light_P.x - node.pos_min.x);
  const float2 dly = make_float2(light_P.y - node.pos_max.y, light_P.y - node.pos_min.y);
  const float2 dlz = make_float2(light_P.z - node.pos_max.z, light_P.z - node.pos_min.z);

  /* Check 2: Hemisphere test — some normal must face the light. */
  const float2 dot_light = spoly_iv3_dot(nx, ny, nz, dlx, dly, dlz);
  if (dot_light.y <= 0.001f) {
    return false;
  }

  /* Check 3: Half-vector alignment with normal interval.
   * Compute H at the center of posBox and check if any normal in norBox
   * could align with it, accounting for angular spread from posBox extent. */
  const float3 center = (node.pos_min + node.pos_max) * 0.5f;
  const float3 to_recv = recv_P - center;
  const float3 to_light = light_P - center;

  const float len_recv = len(to_recv);
  const float len_light = len(to_light);

  if (len_recv < 1e-6f || len_light < 1e-6f) {
    return true; /* Point is inside the box — don't prune. */
  }

  const float3 H_center = normalize(to_recv / len_recv + to_light / len_light);

  /* Max dot(H_center, N) over the normal interval.
   * For each component: take max of H*nor_min and H*nor_max. */
  const float h_dot_max = fmaxf(H_center.x * node.nor_min.x, H_center.x * node.nor_max.x) +
                           fmaxf(H_center.y * node.nor_min.y, H_center.y * node.nor_max.y) +
                           fmaxf(H_center.z * node.nor_min.z, H_center.z * node.nor_max.z);

  /* Angular spread: how much H varies across the posBox.
   * Clamp to avoid overly permissive margin on large/close nodes. */
  const float3 extent = node.pos_max - node.pos_min;
  const float diag = len(extent);
  const float min_dist = fminf(len_recv, len_light);
  const float angular_spread = fminf(diag / fmaxf(min_dist, 1e-6f), 0.5f);

  /* Require that the best possible alignment exceeds a meaningful threshold.
   * This is stricter than just checking > 0, matching the old per-triangle
   * prefilter behavior (dot(H, avg_N) > 0.5) but conservatively loosened. */
  if (h_dot_max + angular_spread < 0.1f) {
    return false;
  }

  return true;
}

/* Maximum tree traversal stack depth.
 * For 4-ary tree: depth = ceil(log4(N)) + 1. Stack can hold up to
 * 4 children per level, so max entries = 4 * max_depth.
 * 64 supports up to ~4M triangles per caster. */
#define SPOLY_TREE_STACK_SIZE 64

/* Maximum solver calls per shading point across all casters.
 * Prevents pathological runtime on meshes where pruning is weak. */
#define SPOLY_MAX_SOLVER_CALLS 256

/* ============================================================================
 * Integrator-level specular polynomial caustic sampling.
 *
 * Runs on DIFFUSE receiver surfaces (like MNEE). Uses a hierarchical 4-ary
 * tree with interval-arithmetic pruning to efficiently search caster triangles
 * for valid specular paths via polynomial constraint solving.
 *
 * Returns the number of specular vertices found (0 = failure, 1 = success).
 * On success, throughput contains the full path contribution and sd_mnee
 * is set up at the specular vertex for shadow ray construction.
 * ============================================================================ */

ccl_device_forceinline int kernel_path_spoly_sample(KernelGlobals kg,
                                                    IntegratorState state,
                                                    ccl_private ShaderData *sd,
                                                    ccl_private ShaderData *sd_mnee,
                                                    const ccl_private RNGState *rng_state,
                                                    ccl_private LightSample *ls,
                                                    ccl_private BsdfEval *throughput)
{
  /* Need a finite light position. */
  if (ls->t == FLT_MAX)
    return 0;

  const int num_caster_objects = kernel_data.integrator.num_spoly_caster_objects;
  if (num_caster_objects == 0)
    return 0;

  const float3 recv_P = sd->P;
  const float3 light_P = ls->P;

  /* Global solver call budget across all casters. */
  int total_solver_calls = 0;

  /* Iterate over each caustic caster object's 4-ary tree. */
  for (int ci = 0; ci < num_caster_objects; ci++) {
    const int caster_object = (int)kernel_data_fetch(spoly_caster_object_index, ci);
    const int tree_offset = (int)kernel_data_fetch(spoly_caster_tree_offset, ci);

    const int object_flags = kernel_data_fetch(object_flag, caster_object);
    const bool need_transform = !(object_flags & SD_OBJECT_TRANSFORM_APPLIED);
    const int prim_offset = kernel_data_fetch(object_prim_offset, caster_object);

    /* Get object transform if needed (for fetching triangle data from kernel arrays). */
    Transform tfm, itfm;
    if (need_transform) {
      tfm = object_fetch_transform(kg, caster_object, OBJECT_TRANSFORM);
      itfm = object_fetch_transform(kg, caster_object, OBJECT_INVERSE_TRANSFORM);
    }

    /* Check if mesh has smooth normals. */
    const int first_shader = kernel_data_fetch(tri_shader, prim_offset);
    if (!(first_shader & SHADER_SMOOTH_NORMAL)) {
      continue;
    }

    /* Reflection-only for now. */
    const bool is_refraction = false;
    const float eta = 1.0f;

    /* Object-level culling: check root node's AABB before entering traversal.
     * This avoids tree traversal overhead for geometrically irrelevant casters. */
    {
      const SPolyTreeNode root = spoly_tree_node_fetch(kg, tree_offset);
      if (!is_refraction && !spoly_tree_node_valid_reflection(root, recv_P, light_P)) {
        continue;
      }
    }

    /* Stack-based tree traversal with interval-arithmetic pruning.
     * Stack entries: (node_index, triangle_prim). When prim >= 0, the entry
     * is a leaf and we skip the tree node re-fetch, solving directly.
     * When prim == -1, the entry is an internal node that needs expansion. */
    int stack_node[SPOLY_TREE_STACK_SIZE];
    int stack_prim[SPOLY_TREE_STACK_SIZE];
    int stack_top = 0;
    stack_node[stack_top] = tree_offset;
    stack_prim[stack_top] = -1;
    stack_top++;

    while (stack_top > 0 && total_solver_calls < SPOLY_MAX_SOLVER_CALLS) {
      stack_top--;
      const int node_idx = stack_node[stack_top];
      const int entry_prim = stack_prim[stack_top];

      if (entry_prim >= 0) {
        /* Leaf entry: solve polynomial constraint on this triangle.
         * The prim index was cached when pushed, avoiding a re-fetch. */
        const int prim = entry_prim;
        total_solver_calls++;

        /* Load triangle vertices and normals from kernel arrays. */
        float3 verts[3], normals[3];
        triangle_vertices_and_normals(kg, prim, verts, normals);

        /* Apply instance transforms if needed. */
        if (need_transform) {
          for (int v = 0; v < 3; v++) {
            verts[v] = transform_point(&tfm, verts[v]);
            normals[v] = normalize(transform_direction_transposed(&itfm, normals[v]));
          }
        }
        else {
          for (int v = 0; v < 3; v++) {
            normals[v] = normalize(normals[v]);
          }
        }

        /* Run the polynomial solver. */
        SPolySolution solutions[SPOLY_MAX_ROOTS];
        int num_solutions = spoly_solve(
            recv_P, light_P, verts[0], verts[1], verts[2],
            normals[0], normals[1], normals[2], is_refraction, eta, solutions);

        if (num_solutions == 0)
          continue;

        /* Validate each solution. */
        for (int si = 0; si < num_solutions; si++) {
          const float3 spec_pos = solutions[si].position;
          const float u = solutions[si].u;
          const float v = solutions[si].v;

          float dist_to_spec;
          const float3 dir_to_spec = normalize_len(spec_pos - recv_P, &dist_to_spec);
          if (dist_to_spec < 1e-6f)
            continue;

          float dist_to_light;
          const float3 dir_to_light = normalize_len(light_P - spec_pos, &dist_to_light);
          if (dist_to_light < 1e-6f)
            continue;

          const float w = 1.0f - u - v;
          float3 spec_N = normalize(w * normals[0] + u * normals[1] + v * normals[2]);

          if (!is_refraction) {
            const float3 H = normalize(-dir_to_spec + dir_to_light);
            if (dot(H, spec_N) < 0.5f)
              continue;
            if (dot(-dir_to_spec, spec_N) < 0.0f || dot(dir_to_light, spec_N) < 0.0f)
              continue;
          }
          else {
            if (dot(-dir_to_spec, spec_N) < 0.0f) {
              spec_N = -spec_N;
            }
          }

          /* Visibility check: receiver to specular point. */
          {
            Ray vis_ray;
            vis_ray.P = recv_P;
            vis_ray.D = dir_to_spec;
            vis_ray.tmin = 0.0f;
            vis_ray.tmax = dist_to_spec;
            vis_ray.self.object = sd->object;
            vis_ray.self.prim = sd->prim;
            vis_ray.self.light_object = OBJECT_NONE;
            vis_ray.self.light_prim = PRIM_NONE;
            vis_ray.dP = differential_make_compact(sd->dP);
            vis_ray.dD = differential_zero_compact();
            vis_ray.time = sd->time;

            Intersection vis_isect;
            if (scene_intersect(kg, &vis_ray, PATH_RAY_TRANSMIT, &vis_isect)) {
              const int hit_object = (vis_isect.object == OBJECT_NONE) ?
                                         kernel_data_fetch(prim_object, vis_isect.prim) :
                                         vis_isect.object;
              if (hit_object != caster_object ||
                  fabsf(dist_to_spec - vis_isect.t) > 0.01f) {
                continue;
              }
            }
          }

          /* Visibility check: specular point to light. */
          {
            Ray vis_ray;
            vis_ray.P = spec_pos;
            vis_ray.D = dir_to_light;
            vis_ray.tmin = 0.0f;
            vis_ray.tmax = dist_to_light;
            vis_ray.self.object = caster_object;
            vis_ray.self.prim = prim;
            vis_ray.self.light_object = ls->object;
            vis_ray.self.light_prim = ls->prim;
            vis_ray.dP = differential_zero_compact();
            vis_ray.dD = differential_zero_compact();
            vis_ray.time = sd->time;

            Intersection vis_isect;
            if (scene_intersect(kg, &vis_ray, PATH_RAY_TRANSMIT, &vis_isect)) {
              if (fabsf(dist_to_light - vis_isect.t) > 0.01f) {
                continue;
              }
            }
          }

          /* === Valid specular path found. Compute contribution. === */

          /* Evaluate receiver BSDF toward the specular point. */
          surface_shader_bsdf_eval(kg, state, sd, dir_to_spec, throughput, ls->shader);

          /* Fresnel at specular point. */
          const float cos_i_spec = fabsf(dot(-dir_to_spec, spec_N));
          float F;
          if (is_refraction) {
            F = 1.0f - fresnel_dielectric_cos(cos_i_spec, eta);
          }
          else {
            F = fresnel_dielectric_cos(cos_i_spec, 1.5f);
          }

          /* Geometry factor (solid angle Jacobian). */
          const float dw0_dx1 = cos_i_spec / fmaxf(sqr(dist_to_spec), 1e-8f);
          const float G = fminf(dw0_dx1, 2.0f);

          bsdf_eval_mul(throughput, F * G);

          /* Update light sample relative to specular point. */
          const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
          const int transmission_bounce = INTEGRATOR_STATE(state, path, transmission_bounce);
          const int diffuse_bounce = INTEGRATOR_STATE(state, path, diffuse_bounce);
          const int bounce = INTEGRATOR_STATE(state, path, bounce);

          INTEGRATOR_STATE_WRITE(state, path, diffuse_bounce) = diffuse_bounce + 1;
          INTEGRATOR_STATE_WRITE(state, path, bounce) = bounce + 1;

          light_sample_update(kg, ls, spec_pos, spec_N, path_flag);

          /* Setup sd_mnee at specular point for light evaluation. */
          const int tri_shader_val = kernel_data_fetch(tri_shader, prim);
          shader_setup_from_sample(kg,
                                   sd_mnee,
                                   spec_pos,
                                   spec_N,
                                   -dir_to_spec,
                                   tri_shader_val,
                                   caster_object,
                                   prim,
                                   u,
                                   v,
                                   dist_to_spec,
                                   sd->time,
                                   false,
                                   false);

          const Spectrum light_eval = light_sample_shader_eval(
              kg, state, sd_mnee, ls, sd->time);
          bsdf_eval_mul(throughput, light_eval / ls->pdf);

          /* Restore bounce state. */
          INTEGRATOR_STATE_WRITE(state, path, transmission_bounce) = transmission_bounce;
          INTEGRATOR_STATE_WRITE(state, path, diffuse_bounce) = diffuse_bounce;
          INTEGRATOR_STATE_WRITE(state, path, bounce) = bounce;

          return 1;
        }
      }
      else {
        /* Internal node: fetch and expand children with interval pruning. */
        const SPolyTreeNode node = spoly_tree_node_fetch(kg, node_idx);

        /* Collect children that pass pruning, with distance for front-to-back ordering. */
        int valid_children[4];
        int valid_prims[4];
        float valid_dist[4];
        int num_valid = 0;

        const float3 midpoint = (recv_P + light_P) * 0.5f;

        for (int c = 0; c < node.num_children; c++) {
          const int child_idx = node.child_offset + c;
          const SPolyTreeNode child = spoly_tree_node_fetch(kg, child_idx);

          /* Apply interval-arithmetic pruning. */
          if (!is_refraction) {
            if (!spoly_tree_node_valid_reflection(child, recv_P, light_P)) {
              continue;
            }
          }

          if (child.num_children == 0 && child.triangle_prim < 0) {
            continue; /* Empty leaf. */
          }

          /* Distance heuristic for front-to-back ordering:
           * closer children are more likely to produce valid paths. */
          const float3 child_center = (child.pos_min + child.pos_max) * 0.5f;
          const float d = len(child_center - midpoint);

          valid_children[num_valid] = child_idx;
          valid_prims[num_valid] = (child.num_children == 0) ? child.triangle_prim : -1;
          valid_dist[num_valid] = d;
          num_valid++;
        }

        /* Push in farthest-first order so closest children are popped first.
         * Simple insertion sort is fine for at most 4 elements. */
        for (int i = 1; i < num_valid; i++) {
          for (int j = i; j > 0 && valid_dist[j] > valid_dist[j - 1]; j--) {
            /* Swap j and j-1 (push farther ones first). */
            int tmp_c = valid_children[j];
            valid_children[j] = valid_children[j - 1];
            valid_children[j - 1] = tmp_c;
            int tmp_p = valid_prims[j];
            valid_prims[j] = valid_prims[j - 1];
            valid_prims[j - 1] = tmp_p;
            float tmp_d = valid_dist[j];
            valid_dist[j] = valid_dist[j - 1];
            valid_dist[j - 1] = tmp_d;
          }
        }

        for (int i = 0; i < num_valid; i++) {
          if (stack_top < SPOLY_TREE_STACK_SIZE) {
            stack_node[stack_top] = valid_children[i];
            stack_prim[stack_top] = valid_prims[i];
            stack_top++;
          }
        }
      }
    }
  }

  return 0;
}

CCL_NAMESPACE_END
