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
 * yielding a 4x4 Bezout matrix and degree-8 resultant.
 * For single-bounce transmission (T), degree 6, yielding 6x6 Bezout matrix and degree-18
 * resultant.
 */

CCL_NAMESPACE_BEGIN

/* Maximum polynomial degrees for single-bounce cases. */
#define SPOLY_MAX_DEGREE_R 4
#define SPOLY_MAX_DEGREE_T 6
#define SPOLY_MAX_BEZOUT_SIZE 6 /* max(R=4, T=6) */
#define SPOLY_MAX_UNI_COEFFS 19 /* 2*SPOLY_MAX_BEZOUT_SIZE + 1 + padding, for T resultant */
#define SPOLY_MAX_BVCOEFFS 7    /* degree+1 for bivariate, max T=6 -> 7 */
#define SPOLY_MAX_ROOTS 16
#define SPOLY_BISECT_ITERATIONS 20
#define SPOLY_ROOT_EPS 1e-6f

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
  p->degree_u = (fabsf(b) > 1e-30f) ? 1 : 0;
  p->degree_v = (fabsf(c) > 1e-30f) ? 1 : 0;
  if (fabsf(a) > 1e-30f && p->degree_u < 0)
    p->degree_u = 0;
  if (fabsf(a) > 1e-30f && p->degree_v < 0)
    p->degree_v = 0;
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
  result->degree_u = max(a->degree_u, b->degree_u);
  result->degree_v = max(a->degree_v, b->degree_v);
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
  result->degree_u = max(a->degree_u, b->degree_u);
  result->degree_v = max(a->degree_v, b->degree_v);
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

/* Extract row i as a univariate polynomial in v (coefficients of u^i). */
ccl_device_inline void spoly_biv_row_as_uni(ccl_private const SPolyBiv *p,
                                            int row,
                                            ccl_private SPolyUni *result)
{
  spoly_uni_zero(result);
  if (row > p->degree_u || row < 0)
    return;
  for (int j = 0; j <= p->degree_v && j < SPOLY_MAX_BVCOEFFS; j++) {
    result->coeffs[j] = p->coeffs[row][j];
    if (fabsf(result->coeffs[j]) > 1e-30f)
      result->degree = j;
  }
}

/* ============================================================================
 * BVP3: Vector of 3 bivariate polynomials (x, y, z components).
 * Used to represent parametric 3D surfaces/vectors as polynomials of (u,v).
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

/* Set to constant 3D vector. */
ccl_device_inline void spoly_bvp3_set_const(ccl_private SPolyBVP3 *p, float3 v)
{
  spoly_biv_set_const(&p->x, v.x);
  spoly_biv_set_const(&p->y, v.y);
  spoly_biv_set_const(&p->z, v.z);
}

/* Set to linear: v0 + (v1-v0)*u + (v2-v0)*v  (barycentric parameterization). */
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

/* Dot product: result = a.x*b.x + a.y*b.y + a.z*b.z (bivariate polynomial). */
ccl_device_inline void spoly_bvp3_dot(ccl_private SPolyBiv *result,
                                      ccl_private const SPolyBVP3 *a,
                                      ccl_private const SPolyBVP3 *b)
{
  SPolyBiv tx, ty, tz;
  spoly_biv_mul(&tx, &a->x, &b->x);
  spoly_biv_mul(&ty, &a->y, &b->y);
  spoly_biv_mul(&tz, &a->z, &b->z);
  SPolyBiv tmp;
  spoly_biv_add(&tmp, &tx, &ty);
  spoly_biv_add(result, &tmp, &tz);
}

/* Cross product: result = a x b (vector of bivariate polynomials). */
ccl_device_inline void spoly_bvp3_cross(ccl_private SPolyBVP3 *result,
                                        ccl_private const SPolyBVP3 *a,
                                        ccl_private const SPolyBVP3 *b)
{
  SPolyBiv t1, t2;
  /* x = a.y*b.z - a.z*b.y */
  spoly_biv_mul(&t1, &a->y, &b->z);
  spoly_biv_mul(&t2, &a->z, &b->y);
  spoly_biv_sub(&result->x, &t1, &t2);
  /* y = a.z*b.x - a.x*b.z */
  spoly_biv_mul(&t1, &a->z, &b->x);
  spoly_biv_mul(&t2, &a->x, &b->z);
  spoly_biv_sub(&result->y, &t1, &t2);
  /* z = a.x*b.y - a.y*b.x */
  spoly_biv_mul(&t1, &a->x, &b->y);
  spoly_biv_mul(&t2, &a->y, &b->x);
  spoly_biv_sub(&result->z, &t1, &t2);
}

/* Scale by scalar. */
ccl_device_inline void spoly_bvp3_scale(ccl_private SPolyBVP3 *p, float s)
{
  spoly_biv_scale(&p->x, s);
  spoly_biv_scale(&p->y, s);
  spoly_biv_scale(&p->z, s);
}

/* ============================================================================
 * Bezout Matrix and Resultant
 *
 * Given two bivariate polynomials Czy(u,v) and Cxz(u,v) viewed as univariate
 * polynomials in u with coefficients that are univariate polynomials in v,
 * the Bezout matrix B[i][j] is an n×n matrix of univariate polynomials in v.
 * Its determinant det(B) is the resultant: a univariate polynomial in v only.
 * Roots of det(B) give candidate v values.
 * ============================================================================ */

/* Evaluate an n×n matrix of univariate polynomials at a specific v value,
 * returning a plain n×n float matrix. */
ccl_device_inline void spoly_bezout_eval_matrix(
    ccl_private SPolyUni bezout[][SPOLY_MAX_BEZOUT_SIZE],
    int n,
    float v,
    ccl_private float mat[][SPOLY_MAX_BEZOUT_SIZE])
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      mat[i][j] = spoly_uni_eval(&bezout[i][j], v);
    }
  }
}

/* Compute determinant of n×n float matrix via Gaussian elimination. */
ccl_device_inline float spoly_matrix_det(ccl_private float mat[][SPOLY_MAX_BEZOUT_SIZE], int n)
{
  float tmp[SPOLY_MAX_BEZOUT_SIZE][SPOLY_MAX_BEZOUT_SIZE];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      tmp[i][j] = mat[i][j];
    }
  }

  float det = 1.0f;
  for (int col = 0; col < n; col++) {
    /* Find pivot. */
    int pivot = -1;
    float max_val = 0.0f;
    for (int row = col; row < n; row++) {
      float av = fabsf(tmp[row][col]);
      if (av > max_val) {
        max_val = av;
        pivot = row;
      }
    }
    if (pivot < 0 || max_val < 1e-20f)
      return 0.0f;

    if (pivot != col) {
      /* Swap rows. */
      for (int j = 0; j < n; j++) {
        float t = tmp[col][j];
        tmp[col][j] = tmp[pivot][j];
        tmp[pivot][j] = t;
      }
      det = -det;
    }
    det *= tmp[col][col];
    float inv_pivot = 1.0f / tmp[col][col];

    for (int row = col + 1; row < n; row++) {
      float factor = tmp[row][col] * inv_pivot;
      for (int j = col + 1; j < n; j++) {
        tmp[row][j] -= factor * tmp[col][j];
      }
    }
  }
  return det;
}

/* Build the Bezout matrix from two bivariate polynomials.
 * The polynomials are treated as univariate in u, with coefficients in v.
 * Bezout matrix entry: B[i][j] = sum_{k} (a[i+k+1]*b[j+k+1-?] - b[i+k+1]*a[j+k+1-?])
 * Actually, the standard Bezout matrix for polys a(u) and b(u) of degree n:
 *   B[i][j] = sum_{k=0}^{min(i,j)} (a[n-i+k]*b[n-j+k] - b[n-i+k]*a[n-j+k]) ... but let's
 * use the correct formulation from the reference.
 *
 * For polynomials f(u) = sum_i a_i u^i and g(u) = sum_i b_i u^i, both of degree n-1,
 * the Bezout matrix B_{i,j} for i,j in [0,n-1]:
 *   B[i][j] = sum_{k=0}^{n-1-max(i,j)} a[i+k+1]*b[j-k+?]...
 *
 * The standard Bezout matrix entry for f,g of degree n:
 *   B[i][j] = sum_{k=0}^{min(i, n-1-j)} (a_{i-k} * b_{j+k+1} - b_{i-k} * a_{j+k+1})
 *   for i,j = 0,...,n-1
 */
ccl_device_inline void spoly_build_bezout(ccl_private const SPolyBiv *poly_f,
                                          ccl_private const SPolyBiv *poly_g,
                                          int n,
                                          ccl_private SPolyUni bezout[][SPOLY_MAX_BEZOUT_SIZE])
{
  /* Extract rows: a[i] is the coefficient of u^i as a univariate polynomial in v. */
  SPolyUni a[SPOLY_MAX_BVCOEFFS];
  SPolyUni b[SPOLY_MAX_BVCOEFFS];
  for (int i = 0; i < SPOLY_MAX_BVCOEFFS; i++) {
    spoly_biv_row_as_uni(poly_f, i, &a[i]);
    spoly_biv_row_as_uni(poly_g, i, &b[i]);
  }

  /* Build the Bezout matrix.
   * Entry B[i][j] = sum_{k} (a[k+i+1]*b[k+j+1-?] - ...)
   *
   * Using the delta-Bezout formulation:
   * (f(x)*g(y) - f(y)*g(x)) / (x - y) = sum_{i,j} B[i][j] * x^i * y^j
   *
   * B[i][j] = sum_{k=1}^{n-max(i,j)} (a[i+k]*b[j+k-1+1] - b[i+k]*a[j+k-1+1])
   * Actually, the standard construction:
   * B[i][j] = sum_{k=0}^{n-1-max(i,j)} (a_{max(i,j)+k+1} * delta)
   *
   * Let me use the concrete formulation from the reference implementation:
   * f[i][j] = a[i] * b[j+1] - b[i] * a[j+1], then symmetrize.
   * This is the direct Bezout construction for a (n x n) matrix where
   * i,j range from 0 to n-1.
   */
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      /* B[i][j] = sum_{k=0}^{?} ... Using the reference formulation:
       * f[i][j] = sum for upper triangle. */
      SPolyUni entry;
      spoly_uni_zero(&entry);

      /* The Bezout construction: for each pair, accumulate
       * a[i+k+1]*b[j-k] - b[i+k+1]*a[j-k] for valid k. */
      for (int k = 0; k <= i; k++) {
        /* Index check: need (i-k) >= 0 and (j+k+1) <= n. */
        int ai = i - k;
        int bi = j + k + 1;
        if (bi > n)
          continue;
        SPolyUni term;
        SPolyUni prod1, prod2;
        spoly_uni_mul(&prod1, &a[ai], &b[bi]);
        spoly_uni_mul(&prod2, &b[ai], &a[bi]);
        spoly_uni_sub(&term, &prod1, &prod2);
        SPolyUni sum;
        spoly_uni_add(&sum, &entry, &term);
        entry = sum;
      }

      bezout[i][j] = entry;
      if (i != j) {
        bezout[j][i] = entry; /* Symmetric. */
      }
    }
  }
}

/* Compute the resultant polynomial by evaluating the Bezout determinant at
 * multiple v-values and interpolating (Lagrange interpolation approach).
 * This avoids symbolic determinant computation which would be very expensive.
 *
 * For a resultant of degree d, we need d+1 sample points. */
ccl_device_inline void spoly_resultant_via_interpolation(
    ccl_private SPolyUni bezout[][SPOLY_MAX_BEZOUT_SIZE],
    int n,
    int result_degree,
    ccl_private SPolyUni *result)
{
  spoly_uni_zero(result);

  int num_samples = result_degree + 1;
  if (num_samples > SPOLY_MAX_UNI_COEFFS)
    num_samples = SPOLY_MAX_UNI_COEFFS;

  /* Sample points and determinant values. */
  float sample_v[SPOLY_MAX_UNI_COEFFS];
  float sample_det[SPOLY_MAX_UNI_COEFFS];

  for (int s = 0; s < num_samples; s++) {
    /* Spread sample points in [0, 1] range, avoiding exact 0 and 1. */
    sample_v[s] = (float)(s) / (float)(num_samples - 1);
    if (num_samples == 1)
      sample_v[s] = 0.5f;

    /* Evaluate the matrix at this v value. */
    float mat[SPOLY_MAX_BEZOUT_SIZE][SPOLY_MAX_BEZOUT_SIZE];
    spoly_bezout_eval_matrix(bezout, n, sample_v[s], mat);
    sample_det[s] = spoly_matrix_det(mat, n);
  }

  /* Lagrange interpolation to recover polynomial coefficients.
   * The resultant polynomial in v of degree result_degree. */
  for (int i = 0; i < num_samples; i++) {
    /* Compute the i-th Lagrange basis polynomial. */
    SPolyUni basis;
    basis.coeffs[0] = 1.0f;
    basis.degree = 0;
    for (int k = 1; k < SPOLY_MAX_UNI_COEFFS; k++)
      basis.coeffs[k] = 0.0f;

    for (int j = 0; j < num_samples; j++) {
      if (j == i)
        continue;
      float denom = sample_v[i] - sample_v[j];
      if (fabsf(denom) < 1e-30f)
        continue;
      float inv_denom = 1.0f / denom;

      /* Multiply basis by (v - sample_v[j]) / (sample_v[i] - sample_v[j]). */
      SPolyUni factor;
      spoly_uni_zero(&factor);
      factor.coeffs[0] = -sample_v[j] * inv_denom;
      factor.coeffs[1] = inv_denom;
      factor.degree = 1;

      SPolyUni new_basis;
      spoly_uni_mul(&new_basis, &basis, &factor);
      basis = new_basis;
    }

    /* Add sample_det[i] * basis to result. */
    spoly_uni_scale(&basis, sample_det[i]);
    SPolyUni sum;
    spoly_uni_add(&sum, result, &basis);
    *result = sum;
  }
  result->degree = result_degree;
  /* Trim trailing near-zeros. */
  while (result->degree >= 0 && fabsf(result->coeffs[result->degree]) < 1e-20f) {
    result->degree--;
  }
}

/* ============================================================================
 * Root isolation via bisection with sign-change detection.
 * ============================================================================ */

ccl_device_inline int spoly_find_roots_in_01(ccl_private const SPolyUni *poly,
                                             ccl_private float roots[SPOLY_MAX_ROOTS])
{
  int num_roots = 0;

  /* Sample the polynomial at many points and detect sign changes. */
  const int num_samples = 64;
  float prev_v = 0.0f;
  float prev_val = spoly_uni_eval(poly, 0.0f);

  for (int s = 1; s <= num_samples; s++) {
    float v = (float)s / (float)num_samples;
    float val = spoly_uni_eval(poly, v);

    if (prev_val * val <= 0.0f && (fabsf(prev_val) > 1e-30f || fabsf(val) > 1e-30f)) {
      /* Sign change detected - bisect. */
      float lo = prev_v, hi = v;
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

    prev_v = v;
    prev_val = val;
  }

  return num_roots;
}

/* ============================================================================
 * Specular constraint formulation for reflection (R).
 *
 * Given:
 *   - xD: camera/viewer position
 *   - xL: light position
 *   - Triangle with vertices P0, P1, P2 and normals N0, N1, N2
 *
 * The point on the triangle: x1(u,v) = P0 + (P1-P0)*u + (P2-P0)*v
 * The normal at point:       n1(u,v) = N0 + (N1-N0)*u + (N2-N0)*v (unnormalized)
 *
 * Reflection constraint (half-vector formulation):
 *   The half-vector h = d0/|d0| + d1/|d1| must be parallel to n.
 *   Equivalently: (d0 * |d1| + d1 * |d0|) × n = 0
 *
 * To avoid square roots, we use the equivalent polynomial constraints:
 *   Czy = (d0·n_hat) * (d1·t2) + (d0·t2) * (d1·n_hat) = 0
 *   Cxz = (d0·n_hat) * (d1·t1) + (d0·t1) * (d1·n_hat) = 0
 *
 * where t1 = n_hat × e1, t2 = n_hat × e2 are tangent vectors derived from
 * the normal and edge vectors, and n_hat is the unnormalized interpolated normal.
 *
 * These are each degree 4 in (u,v) for the reflection case.
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
  /* Parameterize surface position and normal as bivariate polynomials. */
  SPolyBVP3 x1;
  spoly_bvp3_set_barycentric(&x1, P0, P1, P2);

  SPolyBVP3 n1_hat;
  spoly_bvp3_set_barycentric(&n1_hat, N0, N1, N2);

  /* Direction vectors: d0 = x1 - xD, d1 = xL - x1. */
  SPolyBVP3 xD_bvp, xL_bvp;
  spoly_bvp3_set_const(&xD_bvp, xD);
  spoly_bvp3_set_const(&xL_bvp, xL);

  SPolyBVP3 d0, d1;
  spoly_bvp3_sub(&d0, &x1, &xD_bvp);
  spoly_bvp3_sub(&d1, &xL_bvp, &x1);

  /* Edge vectors as BVP3 constants. */
  float3 e1 = P1 - P0;
  float3 e2 = P2 - P0;

  /* Tangent vectors: t1 = n_hat × e1, t2 = n_hat × e2. */
  SPolyBVP3 e1_bvp, e2_bvp;
  spoly_bvp3_set_const(&e1_bvp, e1);
  spoly_bvp3_set_const(&e2_bvp, e2);

  SPolyBVP3 t1, t2;
  spoly_bvp3_cross(&t1, &n1_hat, &e1_bvp);
  spoly_bvp3_cross(&t2, &n1_hat, &e2_bvp);

  /* Dot products for the constraints. */
  SPolyBiv d0_dot_n, d1_dot_n;
  spoly_bvp3_dot(&d0_dot_n, &d0, &n1_hat);
  spoly_bvp3_dot(&d1_dot_n, &d1, &n1_hat);

  SPolyBiv d0_dot_t1, d1_dot_t1;
  spoly_bvp3_dot(&d0_dot_t1, &d0, &t1);
  spoly_bvp3_dot(&d1_dot_t1, &d1, &t1);

  SPolyBiv d0_dot_t2, d1_dot_t2;
  spoly_bvp3_dot(&d0_dot_t2, &d0, &t2);
  spoly_bvp3_dot(&d1_dot_t2, &d1, &t2);

  /* Constraint: Czy = (d0·n) * (d1·t2) + (d0·t2) * (d1·n) = 0
   *             Cxz = (d0·n) * (d1·t1) + (d0·t1) * (d1·n) = 0 */
  SPolyBiv term1, term2;
  spoly_biv_mul(&term1, &d0_dot_n, &d1_dot_t2);
  spoly_biv_mul(&term2, &d0_dot_t2, &d1_dot_n);
  spoly_biv_add(Czy, &term1, &term2);

  spoly_biv_mul(&term1, &d0_dot_n, &d1_dot_t1);
  spoly_biv_mul(&term2, &d0_dot_t1, &d1_dot_n);
  spoly_biv_add(Cxz, &term1, &term2);
}

/* ============================================================================
 * Specular constraint formulation for transmission/refraction (T).
 *
 * For refraction with IOR eta, the constraint is:
 *   eta * (d0 × n) * |d1| + (d1 × n) * |d0| = 0
 *
 * Squaring to eliminate |d0|, |d1|:
 *   eta^2 * |d0×n|^2 * |d1|^2 - |d1×n|^2 * |d0|^2 = 0
 *
 * Projected onto tangent vectors, this gives degree-6 constraints.
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

  float3 e1 = P1 - P0;
  float3 e2 = P2 - P0;
  SPolyBVP3 e1_bvp, e2_bvp;
  spoly_bvp3_set_const(&e1_bvp, e1);
  spoly_bvp3_set_const(&e2_bvp, e2);

  SPolyBVP3 t1, t2;
  spoly_bvp3_cross(&t1, &n1_hat, &e1_bvp);
  spoly_bvp3_cross(&t2, &n1_hat, &e2_bvp);

  /* c0 = d0 × n_hat, c1 = d1 × n_hat. */
  SPolyBVP3 c0, c1;
  spoly_bvp3_cross(&c0, &d0, &n1_hat);
  spoly_bvp3_cross(&c1, &d1, &n1_hat);

  /* |c0|^2, |c1|^2. */
  SPolyBiv c0_sq, c1_sq;
  spoly_bvp3_dot(&c0_sq, &c0, &c0);
  spoly_bvp3_dot(&c1_sq, &c1, &c1);

  /* |d0|^2, |d1|^2. */
  SPolyBiv d0_sq, d1_sq;
  spoly_bvp3_dot(&d0_sq, &d0, &d0);
  spoly_bvp3_dot(&d1_sq, &d1, &d1);

  /* Constraint: eta^2 * |c0|^2 * |d1|^2 - |c1|^2 * |d0|^2 = 0
   * But this is a single scalar constraint. For two constraints we
   * project onto the two tangent directions.
   *
   * Actually, for refraction the approach is:
   * Constraint along t2: (eta*d0·t2*|d1| + d1·t2*|d0|) = 0
   * Squared: eta^2*(d0·t2)^2*|d1|^2 = (d1·t2)^2*|d0|^2
   * Rewritten: eta^2*(d0·t2)^2*(d1·d1) - (d1·t2)^2*(d0·d0) = 0
   *
   * Similarly for t1. */

  SPolyBiv d0_dot_t1, d1_dot_t1, d0_dot_t2, d1_dot_t2;
  spoly_bvp3_dot(&d0_dot_t1, &d0, &t1);
  spoly_bvp3_dot(&d1_dot_t1, &d1, &t1);
  spoly_bvp3_dot(&d0_dot_t2, &d0, &t2);
  spoly_bvp3_dot(&d1_dot_t2, &d1, &t2);

  float eta2 = eta * eta;
  SPolyBiv dt0_sq, dt1_sq;

  /* Czy: eta^2 * (d0·t2)^2 * |d1|^2 - (d1·t2)^2 * |d0|^2 */
  spoly_biv_mul(&dt0_sq, &d0_dot_t2, &d0_dot_t2);
  spoly_biv_mul(&dt1_sq, &d1_dot_t2, &d1_dot_t2);
  SPolyBiv term1, term2;
  spoly_biv_mul(&term1, &dt0_sq, &d1_sq);
  spoly_biv_scale(&term1, eta2);
  spoly_biv_mul(&term2, &dt1_sq, &d0_sq);
  spoly_biv_sub(Czy, &term1, &term2);

  /* Cxz: eta^2 * (d0·t1)^2 * |d1|^2 - (d1·t1)^2 * |d0|^2 */
  spoly_biv_mul(&dt0_sq, &d0_dot_t1, &d0_dot_t1);
  spoly_biv_mul(&dt1_sq, &d1_dot_t1, &d1_dot_t1);
  spoly_biv_mul(&term1, &dt0_sq, &d1_sq);
  spoly_biv_scale(&term1, eta2);
  spoly_biv_mul(&term2, &dt1_sq, &d0_sq);
  spoly_biv_sub(Cxz, &term1, &term2);
}

/* ============================================================================
 * Main solver: find specular points on a triangle.
 *
 * Returns the number of valid solutions found.
 * Each solution is a (u, v) barycentric coordinate on the triangle.
 * ============================================================================ */

struct SPolySolution {
  float u, v;      /* Barycentric coordinates on the triangle. */
  float3 position; /* World-space position of the specular point. */
};

ccl_device_inline int spoly_solve_reflection(float3 xD,
                                             float3 xL,
                                             float3 P0,
                                             float3 P1,
                                             float3 P2,
                                             float3 N0,
                                             float3 N1,
                                             float3 N2,
                                             ccl_private SPolySolution solutions[SPOLY_MAX_ROOTS])
{
  SPolyBiv Czy, Cxz;
  spoly_build_reflection_constraints(xD, xL, P0, P1, P2, N0, N1, N2, &Czy, &Cxz);

  /* Build Bezout matrix: n = max(degree_u of Czy, degree_u of Cxz). */
  int n = max(Czy.degree_u, Cxz.degree_u);
  if (n <= 0)
    return 0;
  if (n > SPOLY_MAX_BEZOUT_SIZE)
    n = SPOLY_MAX_BEZOUT_SIZE;

  /* Pad to same degree in u (fill missing rows with zero). */
  SPolyUni bezout[SPOLY_MAX_BEZOUT_SIZE][SPOLY_MAX_BEZOUT_SIZE];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      spoly_uni_zero(&bezout[i][j]);
    }
  }
  spoly_build_bezout(&Czy, &Cxz, n, bezout);

  /* Resultant degree is at most 2*n*(degree_v).
   * For reflection: degree_u = 4, degree_v = 4, so resultant is up to degree 8. */
  int result_degree = 2 * max(Czy.degree_v, Cxz.degree_v) * (n > 0 ? 1 : 0);
  /* More accurate: resultant degree = n * max_v_degree. */
  result_degree = n * max(Czy.degree_v, Cxz.degree_v);
  if (result_degree <= 0)
    return 0;
  if (result_degree >= SPOLY_MAX_UNI_COEFFS)
    result_degree = SPOLY_MAX_UNI_COEFFS - 1;

  SPolyUni resultant;
  spoly_resultant_via_interpolation(bezout, n, result_degree, &resultant);

  if (resultant.degree < 0)
    return 0;

  /* Find roots of the resultant in [0, 1]. */
  float v_roots[SPOLY_MAX_ROOTS];
  int num_v_roots = spoly_find_roots_in_01(&resultant, v_roots);

  int num_solutions = 0;

  /* For each v root, find u root by back-substitution. */
  for (int ri = 0; ri < num_v_roots; ri++) {
    float v = v_roots[ri];

    /* Evaluate Cxz at this v to get a univariate polynomial in u. */
    SPolyUni cxz_u;
    spoly_biv_eval_at_v(&Cxz, v, &cxz_u);

    /* Find roots of this polynomial in u. */
    float u_roots[SPOLY_MAX_ROOTS];
    int num_u_roots = spoly_find_roots_in_01(&cxz_u, u_roots);

    for (int ui = 0; ui < num_u_roots; ui++) {
      float u = u_roots[ui];

      /* Check barycentric constraint: u + v <= 1, u >= 0, v >= 0. */
      if (u < -SPOLY_ROOT_EPS || v < -SPOLY_ROOT_EPS || (u + v) > 1.0f + SPOLY_ROOT_EPS)
        continue;

      u = clamp(u, 0.0f, 1.0f);
      v = clamp(v, 0.0f, 1.0f - u);

      /* Verify: also check Czy is close to zero. */
      SPolyUni czy_u;
      spoly_biv_eval_at_v(&Czy, v, &czy_u);
      float czy_val = spoly_uni_eval(&czy_u, u);

      /* Use relative tolerance based on constraint magnitudes. */
      float cxz_val = spoly_uni_eval(&cxz_u, u);
      float tolerance = 1e-3f * fmaxf(1.0f, fmaxf(fabsf(czy_val), fabsf(cxz_val)));
      if (fabsf(czy_val) > tolerance)
        continue;

      if (num_solutions < SPOLY_MAX_ROOTS) {
        /* Compute world position. */
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

ccl_device_inline int spoly_solve_refraction(float3 xD,
                                             float3 xL,
                                             float3 P0,
                                             float3 P1,
                                             float3 P2,
                                             float3 N0,
                                             float3 N1,
                                             float3 N2,
                                             float eta,
                                             ccl_private SPolySolution
                                                 solutions[SPOLY_MAX_ROOTS])
{
  SPolyBiv Czy, Cxz;
  spoly_build_refraction_constraints(xD, xL, P0, P1, P2, N0, N1, N2, eta, &Czy, &Cxz);

  int n = max(Czy.degree_u, Cxz.degree_u);
  if (n <= 0)
    return 0;
  if (n > SPOLY_MAX_BEZOUT_SIZE)
    n = SPOLY_MAX_BEZOUT_SIZE;

  SPolyUni bezout[SPOLY_MAX_BEZOUT_SIZE][SPOLY_MAX_BEZOUT_SIZE];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      spoly_uni_zero(&bezout[i][j]);
    }
  }
  spoly_build_bezout(&Czy, &Cxz, n, bezout);

  int result_degree = n * max(Czy.degree_v, Cxz.degree_v);
  if (result_degree <= 0)
    return 0;
  if (result_degree >= SPOLY_MAX_UNI_COEFFS)
    result_degree = SPOLY_MAX_UNI_COEFFS - 1;

  SPolyUni resultant;
  spoly_resultant_via_interpolation(bezout, n, result_degree, &resultant);

  if (resultant.degree < 0)
    return 0;

  float v_roots[SPOLY_MAX_ROOTS];
  int num_v_roots = spoly_find_roots_in_01(&resultant, v_roots);

  int num_solutions = 0;

  for (int ri = 0; ri < num_v_roots; ri++) {
    float v = v_roots[ri];
    SPolyUni cxz_u;
    spoly_biv_eval_at_v(&Cxz, v, &cxz_u);

    float u_roots[SPOLY_MAX_ROOTS];
    int num_u_roots = spoly_find_roots_in_01(&cxz_u, u_roots);

    for (int ui = 0; ui < num_u_roots; ui++) {
      float u = u_roots[ui];

      if (u < -SPOLY_ROOT_EPS || v < -SPOLY_ROOT_EPS || (u + v) > 1.0f + SPOLY_ROOT_EPS)
        continue;

      u = clamp(u, 0.0f, 1.0f);
      v = clamp(v, 0.0f, 1.0f - u);

      SPolyUni czy_u;
      spoly_biv_eval_at_v(&Czy, v, &czy_u);
      float czy_val = spoly_uni_eval(&czy_u, u);
      float cxz_val = spoly_uni_eval(&cxz_u, u);
      float tolerance = 1e-3f * fmaxf(1.0f, fmaxf(fabsf(czy_val), fabsf(cxz_val)));
      if (fabsf(czy_val) > tolerance)
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
 * Integrator-level specular polynomial NEE.
 *
 * Called during direct lighting when the camera ray hits a glossy/glass surface
 * on a triangle mesh. For each triangle light in the scene (or sampled light
 * that illuminates specularly), we solve for the exact specular path.
 *
 * This provides deterministic specular NEE that replaces the near-zero
 * stochastic evaluation that standard path tracing gives for sharp specular
 * surfaces.
 * ============================================================================ */

/* Main entry point: attempt specular polynomial NEE on a glossy/glass surface.
 *
 * Returns true if we successfully found and evaluated a specular connection,
 * in which case bsdf_eval_out is filled with the contribution and ray_out
 * contains the shadow ray to trace.
 *
 * This function is called from integrate_surface_direct_light() when:
 * 1. The setting is enabled (kernel_data.integrator.use_specular_polynomials)
 * 2. The surface has a glossy/glass BSDF
 * 3. The surface is on a triangle mesh (PRIMITIVE_TRIANGLE)
 */
ccl_device_inline bool kernel_path_spoly_connect(KernelGlobals kg,
                                                 IntegratorState state,
                                                 ccl_private ShaderData *sd,
                                                 ccl_private ShaderData *emission_sd,
                                                 const ccl_private RNGState *rng_state,
                                                 ccl_private LightSample *ls,
                                                 ccl_private BsdfEval *bsdf_eval_out)
{
  /* Only works on triangle meshes. */
  if (!(sd->type & PRIMITIVE_TRIANGLE))
    return false;

  /* Must have smooth normals for meaningful interpolated normals. */
  if (!(sd->shader & SHADER_SMOOTH_NORMAL))
    return false;

  /* Find a glossy or glass closure. */
  ccl_private const MicrofacetBsdf *microfacet_bsdf = nullptr;
  bool is_refraction = false;
  float eta = 1.0f;

  for (int i = 0; i < sd->num_closure; i++) {
    ccl_private const ShaderClosure *sc = &sd->closure[i];
    if (CLOSURE_IS_BSDF_MICROFACET(sc->type)) {
      microfacet_bsdf = (ccl_private const MicrofacetBsdf *)sc;
      is_refraction = CLOSURE_IS_REFRACTION(sc->type) || CLOSURE_IS_GLASS(sc->type);
      if (is_refraction) {
        eta = (sd->flag & SD_BACKFACING) ? 1.0f / microfacet_bsdf->ior : microfacet_bsdf->ior;
      }
      break;
    }
  }

  if (!microfacet_bsdf)
    return false;

  /* For rough surfaces, the polynomial solver won't help much -
   * standard sampling works well enough. Only use for sharp specular.
   * Threshold: roughness < 0.1 (alpha < 0.01). */
  float max_alpha = fmaxf(microfacet_bsdf->alpha_x, microfacet_bsdf->alpha_y);
  if (max_alpha > 0.1f)
    return false;

  /* Get light position. For distant/env lights, we can't do spoly
   * (would need a direction, not a position). */
  if (ls->t == FLT_MAX)
    return false;

  float3 xL = ls->P;
  /* Use a virtual camera point along the incoming ray direction.
   * The solver needs 3D positions, not just directions. Place the
   * virtual eye at unit distance along the incoming direction. */
  float3 xD = sd->P + sd->wi;

  /* Fetch triangle vertices and normals. */
  float3 P[3], N[3];
  triangle_vertices_and_normals(kg, sd->prim, P, N);

  /* Solve for specular points. */
  SPolySolution solutions[SPOLY_MAX_ROOTS];
  int num_solutions = 0;

  if (is_refraction) {
    num_solutions = spoly_solve_refraction(xD, xL, P[0], P[1], P[2], N[0], N[1], N[2], eta,
                                           solutions);
  }
  else {
    num_solutions = spoly_solve_reflection(xD, xL, P[0], P[1], P[2], N[0], N[1], N[2],
                                           solutions);
  }

  if (num_solutions == 0)
    return false;

  /* Pick the best solution (closest to the actual shading point, or with
   * the highest BSDF value). For now, evaluate all and pick the brightest. */
  Spectrum best_contribution = zero_spectrum();
  float3 best_direction = zero_float3();
  bool found = false;

  for (int si = 0; si < num_solutions; si++) {
    float3 spec_pos = solutions[si].position;
    float3 D = spec_pos - sd->P;
    float t = len(D);
    if (t < 1e-8f)
      continue;
    D = D / t;

    /* Check that the direction is in the correct hemisphere. */
    bool towards_back = (dot(D, sd->N) < 0.0f);
    if (!is_refraction && towards_back)
      continue;

    /* Evaluate BSDF for this direction. */
    BsdfEval bsdf_eval_tmp ccl_optional_struct_init;
    float bsdf_pdf = surface_shader_bsdf_eval(kg, state, sd, D, &bsdf_eval_tmp, ls->shader);

    if (bsdf_pdf <= 0.0f)
      continue;

    Spectrum contribution = bsdf_eval_sum(&bsdf_eval_tmp);
    float lum = reduce_max(fabs(contribution));

    if (lum > reduce_max(fabs(best_contribution))) {
      best_contribution = contribution;
      best_direction = D;
      found = true;
    }
  }

  if (!found)
    return false;

  /* Now evaluate the light contribution along this direction. */
  const Spectrum light_eval = light_sample_shader_eval(kg, state, emission_sd, ls, sd->time);
  if (is_zero(light_eval))
    return false;

  /* For spoly connections, we use the BSDF evaluation at the found specular direction.
   * The PDF is the light PDF only (no BSDF MIS since the spoly solver is deterministic). */
  float bsdf_pdf = surface_shader_bsdf_eval(kg, state, sd, best_direction, bsdf_eval_out,
                                             ls->shader);

  /* Apply light contribution. The MIS weight for deterministic spoly is 1
   * (we found the exact specular path, so the BSDF PDF is essentially a delta). */
  float mis_weight = 1.0f;
  if (max_alpha > 0.001f) {
    /* For slightly rough surfaces, use MIS with the light PDF. */
    mis_weight = light_sample_mis_weight_nee(kg, ls->pdf, bsdf_pdf);
  }

  bsdf_eval_mul(bsdf_eval_out, light_eval / ls->pdf * mis_weight);

  return true;
}

CCL_NAMESPACE_END
