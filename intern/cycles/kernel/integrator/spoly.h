/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/closure/bsdf_microfacet.h"
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
 * Integrator-level specular polynomial caustic sampling.
 *
 * Runs on DIFFUSE receiver surfaces (like MNEE). Shoots a probe ray toward
 * the light to discover specular caster triangles, then solves the polynomial
 * constraint system on those triangles to find exact specular bounce points.
 *
 * Handles both reflection (glossy) and refraction (glass) caustics.
 * Returns the number of specular vertices found (0 = failure, 1 = success).
 * On success, throughput contains the full path contribution and emission_sd
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

  /* Setup probe ray from receiver toward the light. */
  Ray probe_ray;
  probe_ray.self.object = sd->object;
  probe_ray.self.prim = sd->prim;
  probe_ray.self.light_object = ls->object;
  probe_ray.self.light_prim = ls->prim;
  probe_ray.P = sd->P;
  probe_ray.D = normalize_len(ls->P - sd->P, &probe_ray.tmax);
  probe_ray.tmin = 0.0f;
  probe_ray.dP = differential_make_compact(sd->dP);
  probe_ray.dD = differential_zero_compact();
  probe_ray.time = sd->time;
  Intersection probe_isect;

  /* Phase 1: Find a specular caustic caster along the probe ray. */
  bool found_caster = false;
  bool is_refraction = false;
  float eta = 1.0f;
  float3 caster_verts[3];
  float3 caster_normals[3];

  for (int isect_count = 0; isect_count < 10; isect_count++) {
    const bool hit = scene_intersect(kg, &probe_ray, PATH_RAY_TRANSMIT, &probe_isect);
    if (!hit) {
      break;
    }

    const int object_flags = intersection_get_object_flags(kg, &probe_isect);
    if (object_flags & SD_OBJECT_CAUSTICS_CASTER) {

      /* Must be a triangle primitive. */
      if (!(probe_isect.type & PRIMITIVE_TRIANGLE)) {
        return 0;
      }

      /* Setup shader data on the caster. */
      shader_setup_from_ray(kg, sd_mnee, &probe_ray, &probe_isect);

      /* Must have smooth normals. */
      if (!(sd_mnee->shader & SHADER_SMOOTH_NORMAL)) {
        return 0;
      }

      /* Evaluate shader to get closures. */
      surface_shader_eval<KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW>(
          kg, state, sd_mnee, nullptr, PATH_RAY_DIFFUSE, true);

      /* Find a glossy or refractive microfacet BSDF on the caster. */
      for (int ci = 0; ci < sd_mnee->num_closure; ci++) {
        ccl_private ShaderClosure *bsdf = &sd_mnee->closure[ci];
        if (CLOSURE_IS_BSDF_MICROFACET(bsdf->type)) {
          ccl_private MicrofacetBsdf *mbsdf = (ccl_private MicrofacetBsdf *)bsdf;

          /* Only useful for near-specular surfaces (low roughness). */
          if (fmaxf(mbsdf->alpha_x, mbsdf->alpha_y) > 0.075f) {
            continue;
          }

          is_refraction = CLOSURE_IS_REFRACTION(bsdf->type) || CLOSURE_IS_GLASS(bsdf->type);
          if (is_refraction) {
            eta = (sd_mnee->flag & SD_BACKFACING) ? 1.0f / mbsdf->ior : mbsdf->ior;
          }

          found_caster = true;
          break;
        }
      }

      if (found_caster) {
        /* Load triangle vertices and normals. */
        triangle_vertices_and_normals(kg, sd_mnee->prim, caster_verts, caster_normals);

        /* Apply instance transforms if needed. */
        if (!(sd_mnee->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
          object_position_transform_auto(kg, sd_mnee, &caster_verts[0]);
          object_position_transform_auto(kg, sd_mnee, &caster_verts[1]);
          object_position_transform_auto(kg, sd_mnee, &caster_verts[2]);
          object_normal_transform_auto(kg, sd_mnee, &caster_normals[0]);
          object_normal_transform_auto(kg, sd_mnee, &caster_normals[1]);
          object_normal_transform_auto(kg, sd_mnee, &caster_normals[2]);
        }

        /* Normalize the transformed normals. */
        caster_normals[0] = normalize(caster_normals[0]);
        caster_normals[1] = normalize(caster_normals[1]);
        caster_normals[2] = normalize(caster_normals[2]);

        break;
      }
    }

    /* Continue probing past non-caster intersections. */
    probe_ray.self.object = probe_isect.object;
    probe_ray.self.prim = probe_isect.prim;
    probe_ray.tmin = intersection_t_offset(probe_isect.t);
  }

  if (!found_caster)
    return 0;

  /* Phase 2: Solve the specular polynomial system on the caster triangle.
   * xD = receiver position, xL = light position. */
  SPolySolution solutions[SPOLY_MAX_ROOTS];
  int num_solutions = spoly_solve(sd->P,
                                  ls->P,
                                  caster_verts[0],
                                  caster_verts[1],
                                  caster_verts[2],
                                  caster_normals[0],
                                  caster_normals[1],
                                  caster_normals[2],
                                  is_refraction,
                                  eta,
                                  solutions);

  if (num_solutions == 0)
    return 0;

  /* Phase 3: Pick the best valid solution. */
  float3 best_spec_pos = zero_float3();
  float best_spec_u = 0.0f, best_spec_v = 0.0f;
  float best_score = -1.0f;
  bool found_valid = false;

  for (int si = 0; si < num_solutions; si++) {
    float3 spec_pos = solutions[si].position;
    float u = solutions[si].u;
    float v = solutions[si].v;

    /* Direction from receiver to specular point. */
    float3 to_spec = spec_pos - sd->P;
    float dist_to_spec = len(to_spec);
    if (dist_to_spec < 1e-6f)
      continue;
    to_spec /= dist_to_spec;

    /* Direction from specular point to light. */
    float3 to_light = ls->P - spec_pos;
    float dist_to_light = len(to_light);
    if (dist_to_light < 1e-6f)
      continue;
    to_light /= dist_to_light;

    /* Interpolated normal at the specular point. */
    float w = 1.0f - u - v;
    float3 spec_N = normalize(w * caster_normals[0] + u * caster_normals[1] +
                              v * caster_normals[2]);

    if (!is_refraction) {
      /* Reflection: verify half-vector aligns with normal. */
      float3 wi = -to_spec; /* toward receiver */
      float3 wo = to_light; /* toward light */
      float3 H = normalize(wi + wo);
      float h_dot_n = dot(H, spec_N);
      if (h_dot_n < 0.95f)
        continue;

      /* Hemisphere checks. */
      if (dot(wi, spec_N) < 0.0f || dot(wo, spec_N) < 0.0f)
        continue;
    }
    else {
      /* Refraction: verify Snell's law approximately holds. */
      float cos_i = dot(-to_spec, spec_N);
      if (cos_i < 0.0f) {
        spec_N = -spec_N;
        cos_i = -cos_i;
      }
    }

    /* Score: prefer solutions with larger geometric contribution. */
    float cos_in = fabsf(dot(-to_spec, spec_N));
    float cos_out = fabsf(dot(to_light, spec_N));
    float score = cos_in * cos_out;

    if (score > best_score) {
      best_score = score;
      best_spec_pos = spec_pos;
      best_spec_u = u;
      best_spec_v = v;
      found_valid = true;
    }
  }

  if (!found_valid)
    return 0;

  /* Phase 4: Visibility check - verify line of sight from receiver to specular point. */
  {
    Ray vis_ray;
    vis_ray.P = sd->P;
    float vis_len;
    vis_ray.D = normalize_len(best_spec_pos - sd->P, &vis_len);
    vis_ray.tmin = 0.0f;
    vis_ray.tmax = vis_len;
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
      /* Check that we hit the expected caster object. */
      if (hit_object != sd_mnee->object || fabsf(vis_len - vis_isect.t) > 0.01f) {
        return 0;
      }
    }
  }

  /* Phase 5: Compute path contribution.
   *
   * Path: receiver (sd->P) -> specular_point -> light (ls->P)
   *
   * Contribution = receiver_bsdf(dir_to_spec)
   *              * Fresnel_at_spec
   *              * geometry_factor
   *              * light_eval / ls->pdf
   */

  float dist_to_spec;
  float3 dir_to_spec = normalize_len(best_spec_pos - sd->P, &dist_to_spec);

  /* Evaluate receiver BSDF for the direction toward the specular point. */
  surface_shader_bsdf_eval(kg, state, sd, dir_to_spec, throughput, ls->shader);

  /* Interpolated normal at specular point. */
  float w_bary = 1.0f - best_spec_u - best_spec_v;
  float3 spec_N = normalize(w_bary * caster_normals[0] + best_spec_u * caster_normals[1] +
                            best_spec_v * caster_normals[2]);

  /* Fresnel at specular point. */
  float3 dir_to_light = normalize(ls->P - best_spec_pos);
  float cos_i_spec = dot(-dir_to_spec, spec_N);
  float F;
  if (is_refraction) {
    F = 1.0f - fresnel_dielectric_cos(cos_i_spec, eta);
  }
  else {
    F = fresnel_dielectric_cos(cos_i_spec, 1.5f); /* default IOR for glossy */
  }

  /* Geometry factor: solid angle subtended by specular point at receiver,
   * times the geometric coupling to the light.
   * G = |cos_i_at_spec| / dist_recv_to_spec^2 */
  float cos_o_spec = fabsf(dot(dir_to_light, spec_N));
  float G = fabsf(cos_i_spec) * cos_o_spec / fmaxf(sqr(dist_to_spec), 1e-8f);

  /* Clamp for stability (same approach as MNEE). */
  G = fminf(G, 2.0f);

  bsdf_eval_mul(throughput, F * G);

  /* Update light sample relative to the specular point and evaluate light shader. */
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  /* Save and temporarily adjust bounce info for light shader evaluation. */
  const int transmission_bounce = INTEGRATOR_STATE(state, path, transmission_bounce);
  const int diffuse_bounce = INTEGRATOR_STATE(state, path, diffuse_bounce);
  const int bounce = INTEGRATOR_STATE(state, path, bounce);

  INTEGRATOR_STATE_WRITE(state, path, diffuse_bounce) = diffuse_bounce + 1;
  INTEGRATOR_STATE_WRITE(state, path, bounce) = bounce + 1;

  light_sample_update(kg, ls, best_spec_pos, spec_N, path_flag);

  /* Setup sd_mnee at the specular point for light evaluation and shadow ray.
   * We use shader_setup_from_sample to properly initialize emission_sd. */
  shader_setup_from_sample(kg,
                           sd_mnee,
                           best_spec_pos,
                           spec_N,
                           -dir_to_spec,
                           sd_mnee->shader,
                           sd_mnee->object,
                           sd_mnee->prim,
                           best_spec_u,
                           best_spec_v,
                           dist_to_spec,
                           sd->time,
                           false,
                           false);

  const Spectrum light_eval = light_sample_shader_eval(kg, state, sd_mnee, ls, sd->time);
  bsdf_eval_mul(throughput, light_eval / ls->pdf);

  /* Restore bounce state. */
  INTEGRATOR_STATE_WRITE(state, path, transmission_bounce) = transmission_bounce;
  INTEGRATOR_STATE_WRITE(state, path, diffuse_bounce) = diffuse_bounce;
  INTEGRATOR_STATE_WRITE(state, path, bounce) = bounce;

  return 1;
}

CCL_NAMESPACE_END
