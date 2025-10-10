/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/integrator/path_state.h"
#include "kernel/integrator/surface_shader.h"

#include "kernel/film/data_passes.h"
#include "kernel/film/denoising_passes.h"
#include "kernel/film/light_passes.h"

#include "kernel/light/sample.h"

#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/triangle.h"

#include "kernel/integrator/mnee.h"

#ifdef WITH_CYCLES_MANIFOLD
#  include "manifold/mpg.h"
#  if !defined(__KERNEL_GPU__)
#    include "manifold/mpg_pgl_summary.h"
#  endif
#endif

#include "kernel/integrator/guiding.h"
#include "kernel/integrator/shadow_linking.h"
#include "kernel/integrator/subsurface.h"
#include "kernel/integrator/volume_stack.h"

#include "kernel/types.h"
#include "util/hash.h"
#include "util/math_intersect.h"

CCL_NAMESPACE_BEGIN

#ifdef WITH_CYCLES_MANIFOLD
ccl_device_inline void surface_write_manifold_direct_light(KernelGlobals kg,
                                                           IntegratorState state,
                                                           Spectrum contribution,
                                                           const int lightgroup,
                                                           ccl_global float *ccl_restrict
                                                               render_buffer)
{
  if (is_zero(contribution)) {
    return;
  }

  film_clamp_light(kg, &contribution, INTEGRATOR_STATE(state, path, bounce));

  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  const int sample = INTEGRATOR_STATE(state, path, sample);

  film_write_combined_pass(kg, path_flag, sample, contribution, buffer);

#ifdef __PASSES__
  if (kernel_data.film.light_pass_flag & PASS_ANY) {
    if (path_flag & PATH_RAY_SHADOW_CATCHER_HIT) {
      return;
    }

    if (lightgroup != LIGHTGROUP_NONE && kernel_data.film.pass_lightgroup != PASS_UNUSED) {
      film_write_pass_spectrum(buffer + kernel_data.film.pass_lightgroup + 3 * lightgroup,
                               contribution);
    }

    Spectrum pass_contribution = contribution;
    int pass_offset = PASS_UNUSED;

    if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
      if (path_flag & PATH_RAY_SURFACE_PASS) {
        const Spectrum diffuse_weight = INTEGRATOR_STATE(state, path, pass_diffuse_weight);
        const Spectrum glossy_weight = INTEGRATOR_STATE(state, path, pass_glossy_weight);

        const int glossy_pass_offset = ((INTEGRATOR_STATE(state, path, bounce) == 1) ?
                                            kernel_data.film.pass_glossy_direct :
                                            kernel_data.film.pass_glossy_indirect);
        if (glossy_pass_offset != PASS_UNUSED) {
          film_write_pass_spectrum(
              buffer + glossy_pass_offset, glossy_weight * contribution);
        }

        const int transmission_pass_offset = ((INTEGRATOR_STATE(state, path, bounce) == 1) ?
                                                  kernel_data.film.pass_transmission_direct :
                                                  kernel_data.film.pass_transmission_indirect);
        if (transmission_pass_offset != PASS_UNUSED) {
          const Spectrum transmission_weight = one_spectrum() - diffuse_weight - glossy_weight;
          film_write_pass_spectrum(buffer + transmission_pass_offset,
                                   transmission_weight * contribution);
        }

        pass_offset = (INTEGRATOR_STATE(state, path, bounce) == 1) ?
                          kernel_data.film.pass_diffuse_direct :
                          kernel_data.film.pass_diffuse_indirect;
        if (pass_offset != PASS_UNUSED) {
          pass_contribution *= diffuse_weight;
        }
      }
      else if (path_flag & PATH_RAY_VOLUME_PASS) {
        pass_offset = (INTEGRATOR_STATE(state, path, bounce) == 1) ?
                          kernel_data.film.pass_volume_direct :
                          kernel_data.film.pass_volume_indirect;
      }
    }

    if (pass_offset != PASS_UNUSED) {
      film_write_pass_spectrum(buffer + pass_offset, pass_contribution);
    }
  }
#endif
}

#  ifdef WITH_CYCLES_DEBUG
ccl_device_inline void manifold_debug_store_average_float3(IntegratorState state,
                                                           ccl_global float *buffer,
                                                           const float3 value)
{
#    ifdef __PASSES__
  const int sample = INTEGRATOR_STATE(state, path, sample);

  if (sample == 0) {
    film_overwrite_pass_float3(buffer, value);
    return;
  }

  const float inv_sample = 1.0f / (float)(sample + 1);
  const float prev_weight = (float)sample * inv_sample;
  const float3 prev = kernel_read_pass_float3(buffer);
  const float3 averaged = (prev * prev_weight) + (value * inv_sample);
  film_overwrite_pass_float3(buffer, averaged);
#    else
  (void)state;
  (void)buffer;
  (void)value;
#    endif
}

ccl_device_inline void manifold_debug_store_average_spectrum(IntegratorState state,
                                                              ccl_global float *buffer,
                                                              const Spectrum value)
{
#    ifdef __PASSES__
  const float3 rgb = spectrum_to_rgb(value);
  manifold_debug_store_average_float3(state, buffer, rgb);
#    else
  (void)state;
  (void)buffer;
  (void)value;
#    endif
}

#  endif

#  ifdef WITH_CYCLES_DEBUG
ccl_device_inline void surface_write_manifold_debug_summary(KernelGlobals kg,
                                                           IntegratorState state,
                                                           const GuideSummary &summary,
                                                           const bool summary_available,
                                                           const bool gate_pass,
                                                           const bool relax_gate_summary,
                                                           const bool relax_or_bootstrap_gate,
                                                           ccl_global float *ccl_restrict
                                                               render_buffer)
{
#    ifdef __PASSES__
  if (kernel_data.film.pass_manifold_summary == PASS_UNUSED &&
      kernel_data.film.pass_manifold_gate == PASS_UNUSED)
  {
    return;
  }

  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  if (path_flag & PATH_RAY_SHADOW_CATCHER_PASS) {
    return;
  }

  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);

  if (kernel_data.film.pass_manifold_summary != PASS_UNUSED) {
    const float3 summary_values = make_float3(
        summary.peak_weight, summary.kappa, summary.rbar);
    manifold_debug_store_average_float3(
        state, buffer + kernel_data.film.pass_manifold_summary, summary_values);
  }

  if (kernel_data.film.pass_manifold_gate != PASS_UNUSED) {
    const float3 gate_values = make_float3(summary_available ? 1.0f : 0.0f,
                                           gate_pass ? 1.0f : 0.0f,
                                           relax_or_bootstrap_gate ? 1.0f : 0.0f);
    film_overwrite_pass_float3(buffer + kernel_data.film.pass_manifold_gate, gate_values);
  }
#    else
  (void)kg;
  (void)state;
  (void)summary;
  (void)summary_available;
  (void)gate_pass;
  (void)relax_gate_summary;
  (void)relax_or_bootstrap_gate;
  (void)render_buffer;
#    endif
}

ccl_device_inline void surface_write_manifold_debug_metrics(KernelGlobals kg,
                                                            IntegratorState state,
                                                            const int attempt_count,
                                                            const bool success,
                                                            const float visibility,
                                                            const float seed_pdf,
                                                            const float bounce_pdf,
                                                            const float light_pdf,
                                                            const float jacobian,
                                                            const float pdf_mpg,
                                                            const float weighted_bsdf_pdf,
                                                            const float weighted_guided_pdf,
                                                            const float weighted_nee_pdf,
                                                            const float mis_denominator,
                                                            const float mis_weight,
                                                            const Spectrum &contribution,
                                                            const bool manifold_pdf_factors_valid,
                                                            const uint32_t gate_mask,
                                                            const int failure_code,
                                                            ccl_global float *ccl_restrict
                                                                render_buffer)
{
#    ifdef __PASSES__
  if (kernel_data.film.pass_manifold_attempt == PASS_UNUSED &&
      kernel_data.film.pass_manifold_pdf_factors == PASS_UNUSED &&
      kernel_data.film.pass_manifold_competing_pdfs == PASS_UNUSED &&
      kernel_data.film.pass_manifold_mis == PASS_UNUSED &&
      kernel_data.film.pass_manifold_contribution == PASS_UNUSED)
  {
    return;
  }

  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
  if (path_flag & PATH_RAY_SHADOW_CATCHER_PASS) {
    return;
  }

  ccl_global float *buffer = film_pass_pixel_render_buffer(kg, state, render_buffer);

  (void)success;
  (void)visibility;

  const int sample = INTEGRATOR_STATE(state, path, sample);
  const float inv_sample = 1.0f / (float)(sample + 1);
  const float prev_weight = (sample > 0) ? (float)sample * inv_sample : 0.0f;

  if (kernel_data.film.pass_manifold_attempt != PASS_UNUSED) {
    const float3 stored_values = make_float3((float)attempt_count,
                                             float(gate_mask),
                                             (float)failure_code);
    film_overwrite_pass_float3(buffer + kernel_data.film.pass_manifold_attempt, stored_values);
  }

  if (kernel_data.film.pass_manifold_pdf_factors != PASS_UNUSED) {
    const float3 factor_values = make_float3(seed_pdf, light_pdf, jacobian);
    /* `light_pdf` is stored in receiver solid angle and already includes the Jacobian. */
    const float invalid_sentinel_offset = 1.0f;
    /* Invalid PDF components are stored as -(abs(value) + offset) so the sign flags the
     * failure while the magnitude retains the previously accumulated average. Downstream tools
     * can detect a failure by checking for negative components and recover the stored magnitude
     * as fabs(component) - invalid_sentinel_offset. */

    const float3 prev_components = (sample > 0) ?
                                       kernel_read_pass_float3(buffer +
                                                               kernel_data.film.pass_manifold_pdf_factors) :
                                       make_float3(0.0f, 0.0f, 0.0f);

    const bool prev_x_valid = (sample > 0) && isfinite_safe(prev_components.x);
    const bool prev_y_valid = (sample > 0) && isfinite_safe(prev_components.y);
    const bool prev_z_valid = (sample > 0) && isfinite_safe(prev_components.z);

    const float prev_x = prev_x_valid ?
                              ((prev_components.x < 0.0f) ?
                                   (-prev_components.x - invalid_sentinel_offset) :
                                   prev_components.x) :
                              0.0f;
    const float prev_y = prev_y_valid ?
                              ((prev_components.y < 0.0f) ?
                                   (-prev_components.y - invalid_sentinel_offset) :
                                   prev_components.y) :
                              0.0f;
    const float prev_z = prev_z_valid ?
                              ((prev_components.z < 0.0f) ?
                                   (-prev_components.z - invalid_sentinel_offset) :
                                   prev_components.z) :
                              0.0f;

    const bool value_x_valid = (isfinite_safe(factor_values.x) && factor_values.x >= 0.0f);
    const bool value_y_valid = (isfinite_safe(factor_values.y) && factor_values.y >= 0.0f);
    const bool value_z_valid = (isfinite_safe(factor_values.z) && factor_values.z >= 0.0f);

    float3 stored_values;

    if (value_x_valid) {
      stored_values.x = prev_x_valid ? (prev_x * prev_weight) + (factor_values.x * inv_sample) :
                                       factor_values.x;
    }
    else {
      stored_values.x = prev_x_valid ?
                           -(fabsf(prev_x) + invalid_sentinel_offset) :
                           -(invalid_sentinel_offset);
    }

    if (value_y_valid) {
      stored_values.y = prev_y_valid ? (prev_y * prev_weight) + (factor_values.y * inv_sample) :
                                       factor_values.y;
    }
    else {
      stored_values.y = prev_y_valid ?
                           -(fabsf(prev_y) + invalid_sentinel_offset) :
                           -(invalid_sentinel_offset);
    }

    if (value_z_valid) {
      stored_values.z = prev_z_valid ? (prev_z * prev_weight) + (factor_values.z * inv_sample) :
                                       factor_values.z;
    }
    else {
      stored_values.z = prev_z_valid ?
                           -(fabsf(prev_z) + invalid_sentinel_offset) :
                           -(invalid_sentinel_offset);
    }

    film_overwrite_pass_float3(buffer + kernel_data.film.pass_manifold_pdf_factors, stored_values);
  }

  if (kernel_data.film.pass_manifold_competing_pdfs != PASS_UNUSED) {
    if (manifold_pdf_factors_valid || sample == 0) {
      float3 competing_values = make_float3(
          weighted_bsdf_pdf, weighted_guided_pdf, weighted_nee_pdf);
      const bool competing_valid = (isfinite_safe(competing_values.x) &&
                                    isfinite_safe(competing_values.y) &&
                                    isfinite_safe(competing_values.z));
      if (!competing_valid) {
        competing_values = make_float3(0.0f, 0.0f, 0.0f);
      }
      manifold_debug_store_average_float3(state,
                                          buffer + kernel_data.film.pass_manifold_competing_pdfs,
                                          competing_values);
    }
  }

  if (kernel_data.film.pass_manifold_mis != PASS_UNUSED) {
    if (manifold_pdf_factors_valid || sample == 0) {
      const bool denominator_valid = (isfinite_safe(mis_denominator) && mis_denominator >= 0.0f);
      const bool bounce_valid = (isfinite_safe(bounce_pdf) && bounce_pdf >= 0.0f);
      /* Store the technique pdf, MIS weight, and the bounce-selection pdf for parity comparisons. */
      float3 mis_values = make_float3(
          pdf_mpg, mis_weight, bounce_valid ? bounce_pdf : 0.0f);
      const bool mis_valid = (isfinite_safe(mis_values.x) &&
                              isfinite_safe(mis_values.y) &&
                              denominator_valid &&
                              bounce_valid);
      if (!mis_valid) {
        mis_values = make_float3(0.0f, 0.0f, 0.0f);
      }
      manifold_debug_store_average_float3(
          state, buffer + kernel_data.film.pass_manifold_mis, mis_values);
    }
  }

  if (kernel_data.film.pass_manifold_contribution != PASS_UNUSED) {
    manifold_debug_store_average_spectrum(
        state, buffer + kernel_data.film.pass_manifold_contribution, contribution);
  }
#    else
  (void)kg;
  (void)state;
  (void)attempt_count;
  (void)success;
  (void)visibility;
  (void)seed_pdf;
  (void)light_pdf;
  (void)jacobian;
  (void)pdf_mpg;
  (void)weighted_bsdf_pdf;
  (void)weighted_guided_pdf;
  (void)weighted_nee_pdf;
  (void)mis_denominator;
  (void)mis_weight;
  (void)contribution;
  (void)manifold_pdf_factors_valid;
  (void)gate_mask;
  (void)failure_code;
  (void)render_buffer;
#    endif
}
#  else
ccl_device_inline void surface_write_manifold_debug_summary(KernelGlobals kg,
                                                           IntegratorState state,
                                                           const GuideSummary &summary,
                                                           const bool summary_available,
                                                           const bool gate_pass,
                                                           const bool relax_gate_summary,
                                                           const bool relax_or_bootstrap_gate,
                                                           ccl_global float *ccl_restrict
                                                               render_buffer)
{
  (void)kg;
  (void)state;
  (void)summary;
  (void)summary_available;
  (void)gate_pass;
  (void)relax_gate_summary;
  (void)relax_or_bootstrap_gate;
  (void)render_buffer;
}

ccl_device_inline void surface_write_manifold_debug_metrics(KernelGlobals kg,
                                                            IntegratorState state,
                                                            const int attempt_count,
                                                            const bool success,
                                                            const float visibility,
                                                            const float seed_pdf,
                                                            const float bounce_pdf,
                                                            const float light_pdf,
                                                            const float jacobian,
                                                            const float pdf_mpg,
                                                            const float weighted_bsdf_pdf,
                                                            const float weighted_guided_pdf,
                                                            const float weighted_nee_pdf,
                                                            const float mis_denominator,
                                                            const float mis_weight,
                                                            const Spectrum &contribution,
                                                            const bool manifold_pdf_factors_valid,
                                                            const uint32_t gate_mask,
                                                            const int failure_code,
                                                            ccl_global float *ccl_restrict
                                                                render_buffer)
{
  (void)kg;
  (void)state;
  (void)attempt_count;
  (void)success;
  (void)visibility;
  (void)seed_pdf;
  (void)bounce_pdf;
  (void)light_pdf;
  (void)jacobian;
  (void)pdf_mpg;
  (void)weighted_bsdf_pdf;
  (void)weighted_guided_pdf;
  (void)weighted_nee_pdf;
  (void)mis_denominator;
  (void)mis_weight;
  (void)contribution;
  (void)manifold_pdf_factors_valid;
  (void)gate_mask;
  (void)failure_code;
  (void)render_buffer;
}
#  endif
#endif

ccl_device_forceinline void integrate_surface_shader_setup(KernelGlobals kg,
                                                           ConstIntegratorState state,
                                                           ccl_private ShaderData *sd)
{
  Intersection isect ccl_optional_struct_init;
  integrator_state_read_isect(state, &isect);

  Ray ray ccl_optional_struct_init;
  integrator_state_read_ray(state, &ray);

  shader_setup_from_ray(kg, sd, &ray, &isect);
}

ccl_device_forceinline float3 integrate_surface_ray_offset(KernelGlobals kg,
                                                           const ccl_private ShaderData *sd,
                                                           const float3 ray_P,
                                                           const float3 ray_D)
{
  /* No ray offset needed for other primitive types. */
  if (!(sd->type & PRIMITIVE_TRIANGLE)) {
    return ray_P;
  }

  /* Self intersection tests already account for the case where a ray hits the
   * same primitive. However precision issues can still cause neighboring
   * triangles to be hit. Here we test if the ray-triangle intersection with
   * the same primitive would miss, implying that a neighboring triangle would
   * be hit instead.
   *
   * This relies on triangle intersection to be watertight, and the object inverse
   * object transform to match the one used by ray intersection exactly.
   *
   * Potential improvements:
   * - It appears this happens when either barycentric coordinates are small,
   *   or dot(sd->Ng, ray_D)  is small. Detect such cases and skip test?
   * - Instead of ray offset, can we tweak P to lie within the triangle?
   */

  /* TODO: Investigate if there are better ray offsetting algorithms for each BVH.
   * Cycles and Custom BVH triangle tests aren't numerically identical, meaning
   * this method isn't ideal for them. */

  float3 verts[3];
  if (sd->type == PRIMITIVE_TRIANGLE) {
    triangle_vertices(kg, sd->prim, verts);
  }
  else {
    kernel_assert(sd->type == PRIMITIVE_MOTION_TRIANGLE);
    motion_triangle_vertices(kg, sd->object, sd->prim, sd->time, verts);
  }

  float3 local_ray_P = ray_P;
  float3 local_ray_D = ray_D;

  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    const Transform itfm = object_get_inverse_transform(kg, sd);
    local_ray_P = transform_point(&itfm, local_ray_P);
    local_ray_D = transform_direction(&itfm, local_ray_D);
  }

  if (ray_triangle_intersect_self(local_ray_P, local_ray_D, verts)) {
    return ray_P;
  }
  return ray_offset(ray_P, sd->Ng);
}

ccl_device_forceinline bool integrate_surface_holdout(KernelGlobals kg,
                                                      ConstIntegratorState state,
                                                      ccl_private ShaderData *sd,
                                                      ccl_global float *ccl_restrict render_buffer)
{
  /* Write holdout transparency to render buffer and stop if fully holdout. */
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  if (((sd->flag & SD_HOLDOUT) || (sd->object_flag & SD_OBJECT_HOLDOUT_MASK)) &&
      (path_flag & PATH_RAY_TRANSPARENT_BACKGROUND))
  {
    const Spectrum holdout_weight = surface_shader_apply_holdout(sd);
    const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput);
    const float transparent = average(holdout_weight * throughput);
    film_write_holdout(kg, state, path_flag, transparent, render_buffer);
    if (isequal(holdout_weight, one_spectrum())) {
      return false;
    }
  }

  return true;
}

ccl_device_forceinline void integrate_surface_emission(KernelGlobals kg,
                                                       IntegratorState state,
                                                       const ccl_private ShaderData *sd,
                                                       ccl_global float *ccl_restrict
                                                           render_buffer)
{
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

#ifdef __LIGHT_LINKING__
  if (!light_link_object_match(kg, light_link_receiver_forward(kg, state), sd->object) &&
      !(path_flag & PATH_RAY_CAMERA))
  {
    return;
  }
#endif

#ifdef __SHADOW_LINKING__
  /* Indirect emission of shadow-linked emissive surfaces is done via shadow rays to dedicated
   * light sources. */
  if (kernel_data.kernel_features & KERNEL_FEATURE_SHADOW_LINKING) {
    if (!(path_flag & PATH_RAY_CAMERA) &&
        kernel_data_fetch(objects, sd->object).shadow_set_membership != LIGHT_LINK_MASK_ALL)
    {
      return;
    }
  }
#endif

  /* Evaluate emissive closure. */
  const Spectrum L = surface_shader_emission(sd);

  const float mis_weight = light_sample_mis_weight_forward_surface(kg, state, path_flag, sd);

  guiding_record_surface_emission(kg, state, L, mis_weight);
  film_write_surface_emission(
      kg, state, L, mis_weight, render_buffer, object_lightgroup(kg, sd->object));
}

ccl_device int integrate_surface_ray_portal(KernelGlobals kg,
                                            IntegratorState state,
                                            ccl_private ShaderData *sd,
                                            const ccl_private ShaderClosure *sc)
{
  const ccl_private RayPortalClosure *pc = (const ccl_private RayPortalClosure *)sc;

  float sum_sample_weight = 0.0f;
  for (int i = 0; i < sd->num_closure; i++) {
    const ccl_private ShaderClosure *sc = &sd->closure[i];

    if (CLOSURE_IS_BSDF_OR_BSSRDF(sc->type)) {
      sum_sample_weight += sc->sample_weight;
    }
  }
  if (sum_sample_weight <= 0.0f) {
    return LABEL_NONE;
  }

  if (len_squared(sd->P - pc->P) > 1e-9f) {
    /* if the ray origin is changed, unset the current object,
     * so we can potentially hit the same polygon again */
    INTEGRATOR_STATE_WRITE(state, isect, object) = OBJECT_NONE;
    INTEGRATOR_STATE_WRITE(state, ray, P) = pc->P;
  }
  else {
    INTEGRATOR_STATE_WRITE(state, ray, P) = integrate_surface_ray_offset(kg, sd, pc->P, pc->D);
  }
  INTEGRATOR_STATE_WRITE(state, ray, D) = pc->D;
  INTEGRATOR_STATE_WRITE(state, ray, tmin) = 0.0f;
  INTEGRATOR_STATE_WRITE(state, ray, tmax) = FLT_MAX;
#ifdef __RAY_DIFFERENTIALS__
  INTEGRATOR_STATE_WRITE(state, ray, dP) = differential_make_compact(sd->dP);
#endif

  const float pick_pdf = pc->sample_weight / sum_sample_weight;
  INTEGRATOR_STATE_WRITE(state, path, throughput) *= pc->weight / pick_pdf;

  const int label = LABEL_TRANSMIT | LABEL_RAY_PORTAL;
  path_state_next(kg, state, label, sd->flag);

  return label;
}

/* Branch off a shadow path and initialize common part of it.
 * THe common is between the surface shading and configuration of a special shadow ray for the
 * shadow linking. */
ccl_device_inline IntegratorShadowState
integrate_direct_light_shadow_init_common(KernelGlobals kg,
                                          IntegratorState state,
                                          const ccl_private Ray *ccl_restrict ray,
                                          const Spectrum bsdf_spectrum,
                                          const int light_group,
                                          const int mnee_vertex_count)
{

  /* Branch off shadow kernel. */
  IntegratorShadowState shadow_state = integrator_shadow_path_init(
      kg, state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW, false);

#ifdef __VOLUME__
  /* Copy volume stack and enter/exit volume. */
  integrator_state_copy_volume_stack_to_shadow(kg, shadow_state, state);
#endif

  /* Write shadow ray and associated state to global memory. */
  integrator_state_write_shadow_ray(shadow_state, ray);
  integrator_state_write_shadow_ray_self(shadow_state, ray);

  /* Copy state from main path to shadow path. */
  const Spectrum unlit_throughput = INTEGRATOR_STATE(state, path, throughput);
  const Spectrum throughput = unlit_throughput * bsdf_spectrum;

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, render_pixel_index) = INTEGRATOR_STATE(
      state, path, render_pixel_index);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_offset) = INTEGRATOR_STATE(
      state, path, rng_offset);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_pixel) = INTEGRATOR_STATE(
      state, path, rng_pixel);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, sample) = INTEGRATOR_STATE(
      state, path, sample);

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transparent_bounce) = INTEGRATOR_STATE(
      state, path, transparent_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, volume_bounds_bounce) = INTEGRATOR_STATE(
      state, path, volume_bounds_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, glossy_bounce) = INTEGRATOR_STATE(
      state, path, glossy_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, throughput) = throughput;

  if ((kernel_data.kernel_features & KERNEL_FEATURE_NODE_PORTAL)) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, portal_bounce) = INTEGRATOR_STATE(
        state, path, portal_bounce);
  }

#ifdef __MNEE__
  if (mnee_vertex_count > 0) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transmission_bounce) =
        INTEGRATOR_STATE(state, path, transmission_bounce) + mnee_vertex_count - 1;
    INTEGRATOR_STATE_WRITE(shadow_state,
                           shadow_path,
                           diffuse_bounce) = INTEGRATOR_STATE(state, path, diffuse_bounce) + 1;
    INTEGRATOR_STATE_WRITE(shadow_state,
                           shadow_path,
                           bounce) = INTEGRATOR_STATE(state, path, bounce) + mnee_vertex_count;
  }
  else
#endif
  {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transmission_bounce) = INTEGRATOR_STATE(
        state, path, transmission_bounce);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, diffuse_bounce) = INTEGRATOR_STATE(
        state, path, diffuse_bounce);
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bounce) = INTEGRATOR_STATE(
        state, path, bounce);
  }

  /* Write Light-group, +1 as light-group is int but we need to encode into a uint8_t. */
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, lightgroup) = light_group + 1;

#if defined(__PATH_GUIDING__)
  if ((kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING)) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, unlit_throughput) = unlit_throughput;
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, path_segment) = INTEGRATOR_STATE(
        state, guiding, path_segment);
    INTEGRATOR_STATE(shadow_state, shadow_path, guiding_mis_weight) = 0.0f;
  }
#endif

  return shadow_state;
}

/* Path tracing: sample point on light and evaluate light shader, then
 * queue shadow ray to be traced. */
template<uint node_feature_mask>
#if defined(__KERNEL_GPU__)
ccl_device_forceinline
#else
/* MSVC has very long compilation time (x20) if we force inline this function */
ccl_device
#endif
    void
    integrate_surface_direct_light(KernelGlobals kg,
                                   IntegratorState state,
                                   ccl_private ShaderData *sd,
                                   const ccl_private RNGState *rng_state)
{
  /* Test if there is a light or BSDF that needs direct light. */
  if (!(kernel_data.integrator.use_direct_light && (sd->flag & SD_BSDF_HAS_EVAL))) {
    return;
  }

  /* Sample position on a light. */
  LightSample ls ccl_optional_struct_init;
  {
    const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
    const uint bounce = INTEGRATOR_STATE(state, path, bounce);
    const float3 rand_light = path_state_rng_3D(kg, rng_state, PRNG_LIGHT);

    if (!light_sample_from_position(kg,
                                    rand_light,
                                    sd->time,
                                    sd->P,
                                    sd->N,
                                    light_link_receiver_nee(kg, sd),
                                    sd->flag,
                                    bounce,
                                    path_flag,
                                    &ls))
    {
      return;
    }
  }

  kernel_assert(ls.pdf != 0.0f);

  const bool is_transmission = dot(ls.D, sd->N) < 0.0f;

  if (ls.prim != PRIM_NONE && ls.prim == sd->prim && ls.object == sd->object) {
    /* Skip self intersection if light direction lies in the same hemisphere as the geometric
     * normal. */
    if (dot(ls.D, is_transmission ? -sd->Ng : sd->Ng) > 0.0f) {
      return;
    }
  }

  /* Evaluate light shader.
   *
   * TODO: can we reuse sd memory? In theory we can move this after
   * integrate_surface_bounce, evaluate the BSDF, and only then evaluate
   * the light shader. This could also move to its own kernel, for
   * non-constant light sources. */
  ShaderDataCausticsStorage emission_sd_storage;
  ccl_private ShaderData *emission_sd = AS_SHADER_DATA(&emission_sd_storage);

  Ray ray ccl_optional_struct_init;
  BsdfEval bsdf_eval ccl_optional_struct_init;

  int mnee_vertex_count = 0;  // NOLINT
#ifdef __MNEE__
  IF_KERNEL_FEATURE(MNEE)
  {
    if (ls.type != LIGHT_TRIANGLE) {
      /* Is this a caustic light? */
      const bool use_caustics = kernel_data_fetch(lights, ls.prim).use_caustics;
      if (use_caustics) {
        /* Are we on a caustic caster? */
        if (is_transmission && (sd->object_flag & SD_OBJECT_CAUSTICS_CASTER)) {
          return;
        }

        /* Are we on a caustic receiver? */
        if (!is_transmission && (sd->object_flag & SD_OBJECT_CAUSTICS_RECEIVER)) {
          mnee_vertex_count = kernel_path_mnee_sample(
              kg, state, sd, emission_sd, rng_state, &ls, &bsdf_eval);
        }
      }
    }
  }
  if (mnee_vertex_count > 0) {
    /* Create shadow ray after successful manifold walk:
     * emission_sd contains the last interface intersection and
     * the light sample ls has been updated */
    light_sample_to_surface_shadow_ray(kg, emission_sd, &ls, &ray);
  }
  else
#endif /* __MNEE__ */
  {
    const Spectrum light_eval = light_sample_shader_eval(kg, state, emission_sd, &ls, sd->time);
    if (is_zero(light_eval)) {
      return;
    }

    /* Evaluate BSDF. */
    const float bsdf_pdf = surface_shader_bsdf_eval(kg, state, sd, ls.D, &bsdf_eval, ls.shader);
    const float mis_weight = light_sample_mis_weight_nee(kg, ls.pdf, bsdf_pdf);
    bsdf_eval_mul(&bsdf_eval, light_eval / ls.pdf * mis_weight);

    /* Path termination. */
    const float terminate = path_state_rng_light_termination(kg, rng_state);
    if (light_sample_terminate(kg, &bsdf_eval, terminate)) {
      return;
    }

    /* Create shadow ray. */
    light_sample_to_surface_shadow_ray(kg, sd, &ls, &ray);
  }

  if (ray.self.object != OBJECT_NONE) {
    ray.P = integrate_surface_ray_offset(kg, sd, ray.P, ray.D);
  }

  /* Branch off shadow kernel. */
  IntegratorShadowState shadow_state = integrate_direct_light_shadow_init_common(
      kg, state, &ray, bsdf_eval_sum(&bsdf_eval), ls.group, mnee_vertex_count);

  if (is_transmission) {
#ifdef __VOLUME__
    volume_stack_enter_exit<true>(kg, shadow_state, sd);
#endif
  }

  uint32_t shadow_flag = INTEGRATOR_STATE(state, path, flag);

  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    PackedSpectrum pass_diffuse_weight;
    PackedSpectrum pass_glossy_weight;

    if (shadow_flag & PATH_RAY_ANY_PASS) {
      /* Indirect bounce, use weights from earlier surface or volume bounce. */
      pass_diffuse_weight = INTEGRATOR_STATE(state, path, pass_diffuse_weight);
      pass_glossy_weight = INTEGRATOR_STATE(state, path, pass_glossy_weight);
    }
    else {
      /* Direct light, use BSDFs at this bounce. */
      shadow_flag |= PATH_RAY_SURFACE_PASS;
      pass_diffuse_weight = PackedSpectrum(bsdf_eval_pass_diffuse_weight(&bsdf_eval));
      pass_glossy_weight = PackedSpectrum(bsdf_eval_pass_glossy_weight(&bsdf_eval));
    }

    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_diffuse_weight) = pass_diffuse_weight;
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, pass_glossy_weight) = pass_glossy_weight;
  }

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = shadow_flag;
}

/* Path tracing: bounce off or through surface with new direction. */
ccl_device_forceinline int integrate_surface_bsdf_bssrdf_bounce(
    KernelGlobals kg,
    IntegratorState state,
    ccl_private ShaderData *sd,
    const ccl_private RNGState *rng_state,
    ccl_global float *ccl_restrict render_buffer)
{
  /* Sample BSDF or BSSRDF. */
  if (!(sd->flag & (SD_BSDF | SD_BSSRDF))) {
    return LABEL_NONE;
  }

  float3 rand_bsdf = path_state_rng_3D(kg, rng_state, PRNG_SURFACE_BSDF);
  const ccl_private ShaderClosure *sc = surface_shader_bsdf_bssrdf_pick(sd, &rand_bsdf);

#ifdef __SUBSURFACE__
  /* BSSRDF closure, we schedule subsurface intersection kernel. */
  if (CLOSURE_IS_BSSRDF(sc->type)) {
    return subsurface_bounce(kg, state, sd, sc);
  }
#endif
  if (CLOSURE_IS_RAY_PORTAL(sc->type)) {
    return integrate_surface_ray_portal(kg, state, sd, sc);
  }

  /* BSDF closure, sample direction. */
#ifdef WITH_CYCLES_MANIFOLD
#  if !defined(__KERNEL_GPU__)
  ccl_attr_maybe_unused GuideSummary manifold_summary = GuideSummary();
  ccl_attr_maybe_unused bool manifold_guiding_ready = false;
  ccl_attr_maybe_unused MpgOptions manifold_options = MpgOptions();
  manifold_options.max_bounces = kernel_data.integrator.manifold_max_bounces;
  manifold_options.max_iters = kernel_data.integrator.manifold_max_iterations;
  manifold_options.gate_w = kernel_data.integrator.manifold_gate_weight;
  manifold_options.gate_kappa = kernel_data.integrator.manifold_gate_kappa;
  ccl_attr_maybe_unused const bool manifold_guiding_enabled =
      (kernel_data.integrator.manifold_guiding_enable != 0);
#    if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
  const bool guiding_features_enabled =
      ((kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING) != 0);
  const bool surface_guiding_active =
      (guiding_features_enabled && INTEGRATOR_STATE(state, guiding, use_surface_guiding));
#    else
  ccl_attr_maybe_unused const bool guiding_features_enabled = false;
  ccl_attr_maybe_unused const bool surface_guiding_active = false;
#    endif

  const int current_sample = INTEGRATOR_STATE(state, path, sample);
  const int current_bounce = INTEGRATOR_STATE(state, path, bounce);
  const int bootstrap_sample_limit = 32;
  const int bootstrap_depth_limit = 2;
  const int bootstrap_extended_limit = 128;
  const bool bootstrap_window = (current_sample < bootstrap_sample_limit) ||
                                (current_bounce < bootstrap_depth_limit &&
                                 current_sample < bootstrap_extended_limit);
  bool relax_gate = false;
  bool relax_gate_summary = false;
  bool bootstrap_gate = false;
  bool summary_available = false;
  bool manifold_gate_pass = false;
  int manifold_attempt_count = 0;
  bool manifold_success = false;
  float manifold_visibility = 0.0f;
  float manifold_seed_pdf = -1.0f;
  float manifold_bounce_pdf = -1.0f;
  float manifold_light_pdf = -1.0f;
  float manifold_abs_jacobian = -1.0f;
  float manifold_pdf = 0.0f;
  bool manifold_pdf_factors_valid = false;
  float manifold_weighted_bsdf_pdf = 0.0f;
  float manifold_weighted_guided_pdf = 0.0f;
  float manifold_weighted_nee_pdf = 0.0f;
  float manifold_mis_denominator = 0.0f;
  float manifold_mis_weight = 0.0f;
  Spectrum manifold_debug_contribution = zero_spectrum();
  uint32_t manifold_gate_mask = MPG_GATE_MASK_NONE;
  int manifold_failure_code = int(MPG_FAILURE_NONE);

#    if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
  if (manifold_guiding_enabled && guiding_features_enabled && kg->opgl_surface_sampling_distribution) {
    const float guiding_seed = INTEGRATOR_STATE(state, guiding, sample_surface_guiding_rand);
    bool guiding_distribution_ready = guiding_surface_init_distribution(kg, sd->P, guiding_seed);

    if (!guiding_distribution_ready) {
      const float guiding_seed = path_state_rng_1D(kg, rng_state, PRNG_SURFACE_BSDF_GUIDING);
      guiding_distribution_ready = guiding_surface_init_distribution(kg, sd->P, guiding_seed);
    }

    if (guiding_distribution_ready) {
      const uint seed = hash_uint3(rng_state->rng_pixel,
                                   uint(rng_state->sample),
                                   rng_state->rng_offset);

      summary_available = pgl_estimate_summary(*kg->opgl_surface_sampling_distribution,
                                               sd->Ng,
                                               seed,
                                               manifold_summary);

      if (summary_available) {
        const bool has_direction_relaxed = (manifold_summary.rbar > 1.0e-4f);
        const bool has_direction_strict = (manifold_summary.rbar > 1.0e-3f);
        const bool meets_gate_thresholds =
            (manifold_summary.peak_weight >= kernel_data.integrator.manifold_gate_weight) &&
            (manifold_summary.kappa >= kernel_data.integrator.manifold_gate_kappa);
        manifold_gate_pass = has_direction_strict && meets_gate_thresholds;

        if (!manifold_gate_pass) {
          if (has_direction_relaxed) {
            relax_gate_summary = true;
          }
          else {
            bootstrap_gate = true;
          }

          /* Previously the relaxed attempt required either a directionless summary or
           * an open bootstrap window. Now we always allow a relaxed attempt whenever
           * a summary exists but fails the strict gate check, matching the Mitsuba
           * reference behavior. */
        }

        if (manifold_gate_pass || relax_gate_summary || bootstrap_gate) {
          manifold_guiding_ready = true;
        }

        if (relax_gate_summary || bootstrap_gate) {
          relax_gate = true;
        }
      }
    }

    if (guiding_distribution_ready && surface_guiding_active) {
      guiding_surface_apply_cosine_product(kg, sd->N);
    }
  }
#    endif

  const bool gate_active_local = (manifold_options.gate_w > 0.0f) ||
                                 (manifold_options.gate_kappa > 0.0f);

  if (manifold_guiding_enabled && !manifold_guiding_ready) {
    if (!summary_available) {
      if (!gate_active_local || bootstrap_window) {
        /* Allow a bootstrap attempt without an OpenPGL summary by mirroring the Mitsuba
         * fallback: relax the gate and flag the bootstrap path so mpg_try_connect() can emit a
         * wide seed around the shading normal. */
        bootstrap_gate = true;
        relax_gate = true;
        manifold_guiding_ready = true;
      }
      else {
        /* Outside the bootstrap window we still record the gate failure for diagnostics. */
        manifold_summary.mean_dir = make_float3(0.0f, 0.0f, 0.0f);
        manifold_summary.peak_weight = 0.0f;
        manifold_summary.kappa = 0.0f;
        manifold_summary.rbar = 0.0f;
        manifold_failure_code = int(MPG_FAILURE_GATE);
      }
    }
  }

  manifold_options.relax_gate = relax_gate;

  if (manifold_guiding_enabled) {
    const bool has_dir_relaxed_local = (manifold_summary.rbar > 1.0e-4f);
    const bool has_dir_strict_local = (manifold_summary.rbar > 1.0e-3f);
    if (gate_active_local) {
      manifold_gate_mask |= MPG_GATE_MASK_ACTIVE;
    }
    if (has_dir_relaxed_local) {
      manifold_gate_mask |= MPG_GATE_MASK_HAS_DIRECTION_RELAXED;
    }
    if (has_dir_strict_local) {
      manifold_gate_mask |= MPG_GATE_MASK_HAS_DIRECTION_STRICT;
    }
    if (manifold_gate_pass || !gate_active_local) {
      manifold_gate_mask |= MPG_GATE_MASK_STRICT_PASS;
    }
    if (relax_gate_summary) {
      manifold_gate_mask |= MPG_GATE_MASK_RELAX_PASS;
    }
    if (bootstrap_gate) {
      manifold_gate_mask |= MPG_GATE_MASK_BOOTSTRAP_PASS;
    }
  }

  if (manifold_guiding_enabled) {
    const bool relax_or_bootstrap_gate = (relax_gate_summary || bootstrap_gate);
    surface_write_manifold_debug_summary(kg,
                                         state,
                                         manifold_summary,
                                         summary_available,
                                         manifold_gate_pass,
                                         relax_gate_summary,
                                         relax_or_bootstrap_gate,
                                         render_buffer);
  }

  if (manifold_guiding_enabled && !manifold_guiding_ready) {
    if (summary_available && !manifold_gate_pass && !relax_gate) {
      manifold_failure_code = int(MPG_FAILURE_GATE);
    }
  }

  if (manifold_guiding_enabled && manifold_guiding_ready) {
    const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);
    const int bounce = INTEGRATOR_STATE(state, path, bounce);
    RNGState manifold_rng_state = *rng_state;
    MpgResult mpg_result = mpg_try_connect(kg,
                                           *sd,
                                           *sc,
                                           manifold_summary,
                                           manifold_options,
                                           path_flag,
                                           bounce,
                                           manifold_rng_state);

    manifold_attempt_count = mpg_result.attempt_count;
    manifold_gate_mask |= mpg_result.gate_mask;
    manifold_failure_code = int(mpg_result.failure_code);
    manifold_success = mpg_result.success;
    manifold_visibility = mpg_result.visibility;

    const bool gate_pass_any_result =
        ((mpg_result.gate_mask & MPG_GATE_MASK_ACTIVE) == 0) ||
        ((mpg_result.gate_mask & (MPG_GATE_MASK_STRICT_PASS | MPG_GATE_MASK_RELAX_PASS |
                                  MPG_GATE_MASK_BOOTSTRAP_PASS)) != 0);
    const bool mpg_failure = (mpg_result.failure_code != MPG_FAILURE_NONE);

    manifold_seed_pdf = -1.0f;
    manifold_bounce_pdf = -1.0f;
    manifold_light_pdf = -1.0f;
    manifold_abs_jacobian = -1.0f;
    manifold_pdf = 0.0f;

    LightSample mpg_light = mpg_result.light;
    const float3 wi_mpg = mpg_result.wi;
    const bool has_valid_wi = !is_zero(wi_mpg);
    const float pdf_mpg_sa = (isfinite_safe(mpg_result.pdf) && mpg_result.pdf > 0.0f) ?
                                 fmaxf(mpg_result.pdf, 1.0e-16f) :
                                 0.0f;
    const float pdf_nee_sa = (isfinite_safe(mpg_result.nee_pdf) && mpg_result.nee_pdf > 0.0f) ?
                                 fmaxf(mpg_result.nee_pdf, 1.0e-16f) :
                                 0.0f;
    const float jacobian_abs = (isfinite_safe(mpg_result.jacobian_total)) ?
                                   fmaxf(fabsf(mpg_result.jacobian_total), 1.0e-16f) :
                                   0.0f;
    const bool seed_pdf_valid = (isfinite_safe(mpg_result.seed_pdf) && mpg_result.seed_pdf > 0.0f);
    const bool light_pdf_valid = (isfinite_safe(mpg_result.light_pdf) && mpg_result.light_pdf > 0.0f);
    const bool bounce_pdf_valid =
        (isfinite_safe(mpg_result.bounce_pdf) && mpg_result.bounce_pdf > 0.0f);

    manifold_pdf_factors_valid = (gate_pass_any_result && !mpg_failure && mpg_result.success &&
                                  seed_pdf_valid && light_pdf_valid && bounce_pdf_valid &&
                                  jacobian_abs > 0.0f &&
                                  pdf_mpg_sa > 0.0f);

    if (manifold_pdf_factors_valid) {
      manifold_seed_pdf = mpg_result.seed_pdf;
      manifold_bounce_pdf = mpg_result.bounce_pdf;
      manifold_light_pdf = mpg_result.light_pdf;
      manifold_abs_jacobian = jacobian_abs;
      manifold_pdf = pdf_mpg_sa;
    }

    if (manifold_guiding_ready && gate_pass_any_result && !mpg_failure) {
      kernel_assert(seed_pdf_valid);
      kernel_assert(light_pdf_valid);
      kernel_assert(bounce_pdf_valid);
      kernel_assert(jacobian_abs > 0.0f);
      kernel_assert(pdf_mpg_sa > 0.0f);
      kernel_assert(pdf_nee_sa > 0.0f);
    }

    const bool mpg_ok =
        (manifold_pdf_factors_valid && has_valid_wi && mpg_result.visibility > 0.0f);

    float pdf_bsdf_sa = 0.0f;
    float pdf_guided_sa = 0.0f;

    if (has_valid_wi) {
      BsdfEval mpg_pdf_eval;
      bsdf_eval_init(&mpg_pdf_eval, zero_spectrum());
      float unguided_pdfs[MAX_CLOSURE];
      pdf_bsdf_sa = surface_shader_bsdf_eval_pdfs(
          kg, sd, wi_mpg, &mpg_pdf_eval, unguided_pdfs, mpg_light.shader);

#      if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
      if (surface_guiding_active)
      {
        const float guiding_sampling_prob = INTEGRATOR_STATE(
            state, guiding, surface_guiding_sampling_prob);
        const float bssrdf_sampling_prob = INTEGRATOR_STATE(state, guiding, bssrdf_sampling_prob);
        const float guiding_pdf = guiding_bsdf_pdf(kg, wi_mpg);
        const float guided_pdf =
            ((isfinite_safe(guiding_pdf) && guiding_pdf > 0.0f) ? guiding_pdf : 0.0f) *
            (1.0f - bssrdf_sampling_prob);

        if (kernel_data.integrator.guiding_directional_sampling_type ==
            GUIDING_DIRECTIONAL_SAMPLING_TYPE_RIS)
        {
          pdf_bsdf_sa *= 0.5f;
          pdf_guided_sa = 0.5f * guided_pdf;
        }
        else {
          pdf_guided_sa = guiding_sampling_prob * guided_pdf;
          pdf_bsdf_sa *= (1.0f - guiding_sampling_prob);
        }
      }
#      endif
    }

    pdf_bsdf_sa = fmaxf(pdf_bsdf_sa, 0.0f);
    pdf_guided_sa = fmaxf(pdf_guided_sa, 0.0f);
    const float pdf_mpg = mpg_ok ? pdf_mpg_sa : 0.0f;

    /* Follow Mitsuba reference behaviour and keep the standard NEE proposal in the
     * MIS denominator even when the manifold solve inserts specular vertices. The
     * comparison must happen in the receiver's solid-angle measure using the
     * pre-MPG NEE PDF at the shading point.
     */
    const float pdf_nee = fmaxf(pdf_nee_sa, 0.0f);

    manifold_weighted_bsdf_pdf = pdf_bsdf_sa;
    manifold_weighted_guided_pdf = pdf_guided_sa;
    manifold_weighted_nee_pdf = pdf_nee;
    if (pdf_mpg > 0.0f) {
      manifold_pdf = pdf_mpg;
    }

    const float mis_denominator = pdf_bsdf_sa + pdf_guided_sa + pdf_nee + pdf_mpg;
    float mis_weight = 0.0f;
    if (pdf_mpg > 0.0f &&
        mis_denominator > 0.0f &&
        isfinite_safe(mis_denominator)) {
      mis_weight = pdf_mpg / mis_denominator;

      if (isfinite_safe(mis_weight)) {
        const float expected_weight = pdf_mpg / mis_denominator;
        const float diff = fabsf(expected_weight - mis_weight);
        const float tolerance = fmaxf(1.0e-6f, fabsf(expected_weight) * 1.0e-5f);
        kernel_assert(diff <= tolerance);
      }
    }

    manifold_mis_denominator = mis_denominator;
    manifold_mis_weight = mis_weight;

    if (mpg_ok && !is_zero(mpg_result.spec_weight))
    {
      ShaderDataCausticsStorage mpg_emission_sd_storage;
      ccl_private ShaderData *mpg_emission_sd = AS_SHADER_DATA(&mpg_emission_sd_storage);

      /* Mirror the temporary bounce adjustment in mnee_path_contribution so Light Path nodes
       * observe the MPG specular chain with the correct history. */
      const int diffuse_bounce = INTEGRATOR_STATE(state, path, diffuse_bounce);
      const int transmission_bounce = INTEGRATOR_STATE(state, path, transmission_bounce);
      const int bounce = INTEGRATOR_STATE(state, path, bounce);
      const int specular_count = (mpg_result.specular_vertex_count > 0) ?
                                     mpg_result.specular_vertex_count :
                                     0;
      int refractive_count = 0;
      for (int i = 0; i < specular_count; ++i) {
        if (mpg_result.specular_vertices[i].is_refraction) {
          refractive_count++;
        }
      }

      INTEGRATOR_STATE_WRITE(state, path, diffuse_bounce) = diffuse_bounce + 1;
      INTEGRATOR_STATE_WRITE(state, path, transmission_bounce) = transmission_bounce +
                                                                 refractive_count;
      INTEGRATOR_STATE_WRITE(state, path, bounce) = bounce + specular_count;

      Spectrum light_eval = light_sample_shader_eval(kg, state, mpg_emission_sd, &mpg_light, sd->time);

      INTEGRATOR_STATE_WRITE(state, path, diffuse_bounce) = diffuse_bounce;
      INTEGRATOR_STATE_WRITE(state, path, transmission_bounce) = transmission_bounce;
      INTEGRATOR_STATE_WRITE(state, path, bounce) = bounce;

      if (!is_zero(light_eval)) {
        BsdfEval mpg_bsdf_eval;
        const float mpg_bsdf_pdf = surface_shader_bsdf_eval(
            kg, state, sd, wi_mpg, &mpg_bsdf_eval, mpg_light.shader);

        if (mpg_bsdf_pdf > 0.0f) {
          bsdf_eval_mul(&mpg_bsdf_eval, mpg_result.spec_weight);

          if (!bsdf_eval_is_zero(&mpg_bsdf_eval) && mis_weight > 0.0f) {
            const float visibility_weight = (mpg_result.visibility > 0.0f && pdf_mpg > 0.0f) ? (mpg_result.visibility * (mis_weight / pdf_mpg)) : 0.0f;
            if (visibility_weight > 0.0f && isfinite_safe(visibility_weight)) {
              bsdf_eval_mul(&mpg_bsdf_eval, visibility_weight);

              Spectrum mpg_contribution =
                  INTEGRATOR_STATE(state, path, throughput) * bsdf_eval_sum(&mpg_bsdf_eval) * light_eval;

              manifold_debug_contribution = mpg_contribution;

              surface_write_manifold_direct_light(kg,
                                                  state,
                                                  mpg_contribution,
                                                  mpg_light.group,
                                                  render_buffer);
#      if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 1
              if (manifold_options.relax_gate) {
                guiding_record_manifold_direct_light(kg, state, mpg_contribution, mis_weight);
              }
#      endif
            }
          }
        }
      }
    }

    if (!mpg_result.success &&
        manifold_abs_jacobian < 0.0f &&
        mpg_result.failure_code != MPG_FAILURE_NONE)
    {
      manifold_abs_jacobian = -float(mpg_result.failure_code);
    }
  }

  if (manifold_guiding_enabled) {
    surface_write_manifold_debug_metrics(kg,
                                         state,
                                         manifold_attempt_count,
                                         manifold_success,
                                         manifold_visibility,
                                         manifold_seed_pdf,
                                         manifold_bounce_pdf,
                                         manifold_light_pdf,
                                         manifold_abs_jacobian,
                                         manifold_pdf,
                                         manifold_weighted_bsdf_pdf,
                                         manifold_weighted_guided_pdf,
                                         manifold_weighted_nee_pdf,
                                         manifold_mis_denominator,
                                         manifold_mis_weight,
                                         manifold_debug_contribution,
                                         manifold_pdf_factors_valid,
                                         manifold_gate_mask,
                                         manifold_failure_code,
                                         render_buffer);
  }
#  endif
#endif

  float bsdf_pdf = 0.0f;
  float unguided_bsdf_pdf = 0.0f;
  BsdfEval bsdf_eval ccl_optional_struct_init;
  float3 bsdf_wo ccl_optional_struct_init;
  int label;

  float2 bsdf_sampled_roughness = make_float2(1.0f, 1.0f);
  float bsdf_eta = 1.0f;
  float mis_pdf = 1.0f;

#if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
  if (kernel_data.integrator.use_surface_guiding &&
      (kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING))
  {
    label = surface_shader_bsdf_guided_sample_closure(kg,
                                                      state,
                                                      sd,
                                                      sc,
                                                      rand_bsdf,
                                                      &bsdf_eval,
                                                      &bsdf_wo,
                                                      &bsdf_pdf,
                                                      &mis_pdf,
                                                      &unguided_bsdf_pdf,
                                                      &bsdf_sampled_roughness,
                                                      &bsdf_eta,
                                                      rng_state);

    if (bsdf_pdf == 0.0f || bsdf_eval_is_zero(&bsdf_eval)) {
      return LABEL_NONE;
    }

    INTEGRATOR_STATE_WRITE(state, path, unguided_throughput) *= bsdf_pdf / unguided_bsdf_pdf;
  }
  else
#endif
  {
    label = surface_shader_bsdf_sample_closure(kg,
                                               sd,
                                               sc,
                                               rand_bsdf,
                                               &bsdf_eval,
                                               &bsdf_wo,
                                               &bsdf_pdf,
                                               &bsdf_sampled_roughness,
                                               &bsdf_eta);

    if (bsdf_pdf == 0.0f || bsdf_eval_is_zero(&bsdf_eval)) {
      return LABEL_NONE;
    }
    mis_pdf = bsdf_pdf;
    unguided_bsdf_pdf = bsdf_pdf;
  }

  if (label & LABEL_TRANSPARENT) {
    /* Only need to modify start distance for transparent. */
    INTEGRATOR_STATE_WRITE(state, ray, tmin) = intersection_t_offset(sd->ray_length);
  }
  else {
    /* Setup ray with changed origin and direction. */
    const float3 D = normalize(bsdf_wo);
    INTEGRATOR_STATE_WRITE(state, ray, P) = integrate_surface_ray_offset(kg, sd, sd->P, D);
    INTEGRATOR_STATE_WRITE(state, ray, D) = D;
    INTEGRATOR_STATE_WRITE(state, ray, tmin) = 0.0f;
    INTEGRATOR_STATE_WRITE(state, ray, tmax) = FLT_MAX;
#ifdef __RAY_DIFFERENTIALS__
    INTEGRATOR_STATE_WRITE(state, ray, dP) = differential_make_compact(sd->dP);
#endif
  }

  /* Update throughput. */
  const Spectrum bsdf_weight = bsdf_eval_sum(&bsdf_eval) / bsdf_pdf;
  INTEGRATOR_STATE_WRITE(state, path, throughput) *= bsdf_weight;

  if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_PASSES) {
    if (INTEGRATOR_STATE(state, path, bounce) == 0) {
      INTEGRATOR_STATE_WRITE(state, path, pass_diffuse_weight) = bsdf_eval_pass_diffuse_weight(
          &bsdf_eval);
      INTEGRATOR_STATE_WRITE(state, path, pass_glossy_weight) = bsdf_eval_pass_glossy_weight(
          &bsdf_eval);
    }
  }

  /* Update path state */
  if (!(label & LABEL_TRANSPARENT)) {
    INTEGRATOR_STATE_WRITE(state, path, mis_ray_pdf) = mis_pdf;
    INTEGRATOR_STATE_WRITE(state, path, mis_origin_n) = sd->N;
    INTEGRATOR_STATE_WRITE(state, path, min_ray_pdf) = fminf(
        unguided_bsdf_pdf, INTEGRATOR_STATE(state, path, min_ray_pdf));

#ifdef __LIGHT_LINKING__
    if (kernel_data.kernel_features & KERNEL_FEATURE_LIGHT_LINKING) {
      INTEGRATOR_STATE_WRITE(state, path, mis_ray_object) = sd->object;
    }
#endif
  }

  path_state_next(kg, state, label, sd->flag);

  guiding_record_surface_bounce(kg,
                                state,
                                bsdf_weight,
                                bsdf_pdf,
                                sd->N,
                                normalize(bsdf_wo),
                                bsdf_sampled_roughness,
                                bsdf_eta);

  return label;
}

#ifdef __VOLUME__
ccl_device_forceinline int integrate_surface_volume_only_bounce(IntegratorState state,
                                                                ccl_private ShaderData *sd)
{
  if (!path_state_volume_next(state)) {
    return LABEL_NONE;
  }

  /* Only modify start distance. */
  INTEGRATOR_STATE_WRITE(state, ray, tmin) = intersection_t_offset(sd->ray_length);

  return LABEL_TRANSMIT | LABEL_TRANSPARENT;
}
#endif

ccl_device_forceinline bool integrate_surface_terminate(IntegratorState state,
                                                        const uint32_t path_flag)
{
  const float continuation_probability = (path_flag & PATH_RAY_TERMINATE_ON_NEXT_SURFACE) ?
                                             0.0f :
                                             INTEGRATOR_STATE(
                                                 state, path, continuation_probability);
  if (continuation_probability == 0.0f) {
    return true;
  }
  if (continuation_probability != 1.0f) {
    INTEGRATOR_STATE_WRITE(state, path, throughput) /= continuation_probability;
  }

  return false;
}

#if defined(__AO__)
ccl_device_forceinline void integrate_surface_ao(KernelGlobals kg,
                                                 IntegratorState state,
                                                 const ccl_private ShaderData *ccl_restrict sd,
                                                 const ccl_private RNGState *ccl_restrict
                                                     rng_state)
{
  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  if (!(kernel_data.kernel_features & KERNEL_FEATURE_AO_ADDITIVE) &&
      !(path_flag & PATH_RAY_CAMERA))
  {
    return;
  }

  /* Skip AO for paths that were split off for shadow catchers to avoid double-counting. */
  if (path_flag & PATH_RAY_SHADOW_CATCHER_PASS) {
    return;
  }

  const float2 rand_bsdf = path_state_rng_2D(kg, rng_state, PRNG_SURFACE_BSDF);

  float3 ao_N;
  const Spectrum ao_weight = surface_shader_ao(
      sd, kernel_data.integrator.ao_additive_factor, &ao_N);

  float3 ao_D;
  float ao_pdf;
  sample_cos_hemisphere(ao_N, rand_bsdf, &ao_D, &ao_pdf);

  bool skip_self = true;

  Ray ray ccl_optional_struct_init;
  ray.P = shadow_ray_offset(kg, sd, ao_D, &skip_self);
  ray.D = ao_D;
  if (skip_self) {
    ray.P = integrate_surface_ray_offset(kg, sd, ray.P, ray.D);
  }
  ray.tmin = 0.0f;
  ray.tmax = kernel_data.integrator.ao_bounces_distance;
  ray.time = sd->time;
  ray.self.object = (skip_self) ? sd->object : OBJECT_NONE;
  ray.self.prim = (skip_self) ? sd->prim : PRIM_NONE;
  ray.self.light_object = OBJECT_NONE;
  ray.self.light_prim = PRIM_NONE;
  ray.dP = differential_zero_compact();
  ray.dD = differential_zero_compact();

  /* Branch off shadow kernel. */
  IntegratorShadowState shadow_state = integrator_shadow_path_init(
      kg, state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW, true);

#  ifdef __VOLUME__
  /* Copy volume stack and enter/exit volume. */
  integrator_state_copy_volume_stack_to_shadow(kg, shadow_state, state);
#  endif

  /* Write shadow ray and associated state to global memory. */
  integrator_state_write_shadow_ray(shadow_state, &ray);
  integrator_state_write_shadow_ray_self(shadow_state, &ray);

  /* Copy state from main path to shadow path. */
  const uint16_t bounce = INTEGRATOR_STATE(state, path, bounce);
  const uint16_t transparent_bounce = INTEGRATOR_STATE(state, path, transparent_bounce);
  const uint32_t shadow_flag = INTEGRATOR_STATE(state, path, flag) | PATH_RAY_SHADOW_FOR_AO;
  const Spectrum throughput = INTEGRATOR_STATE(state, path, throughput) * surface_shader_alpha(sd);

  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, render_pixel_index) = INTEGRATOR_STATE(
      state, path, render_pixel_index);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_offset) = INTEGRATOR_STATE(
      state, path, rng_offset);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, rng_pixel) = INTEGRATOR_STATE(
      state, path, rng_pixel);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, sample) = INTEGRATOR_STATE(
      state, path, sample);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, flag) = shadow_flag;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, bounce) = bounce;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, transparent_bounce) = transparent_bounce;
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, volume_bounds_bounce) = INTEGRATOR_STATE(
      state, path, volume_bounds_bounce);
  INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, throughput) = throughput;

  if (kernel_data.kernel_features & KERNEL_FEATURE_AO_ADDITIVE) {
    INTEGRATOR_STATE_WRITE(shadow_state, shadow_path, unshadowed_throughput) = ao_weight;
  }
}
#endif /* defined(__AO__) */

template<uint node_feature_mask>
ccl_device int integrate_surface(KernelGlobals kg,
                                 IntegratorState state,
                                 ccl_global float *ccl_restrict render_buffer)

{
  PROFILING_INIT_FOR_SHADER(kg, PROFILING_SHADE_SURFACE_SETUP);

  /* Setup shader data. */
  ShaderData sd;
  integrate_surface_shader_setup(kg, state, &sd);
  PROFILING_SHADER(sd.object, sd.shader);

  int continue_path_label = 0;

  const uint32_t path_flag = INTEGRATOR_STATE(state, path, flag);

  /* Skip most work for volume bounding surface. */
#ifdef __VOLUME__
  if (!(sd.flag & SD_HAS_ONLY_VOLUME)) {
#endif
    guiding_record_surface_segment(kg, state, &sd);

#ifdef __SUBSURFACE__
    /* Can skip shader evaluation for BSSRDF exit point without bump mapping. */
    if (!(path_flag & PATH_RAY_SUBSURFACE) || ((sd.flag & SD_HAS_BSSRDF_BUMP)))
#endif
    {
      /* Evaluate shader. */
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_EVAL);
      surface_shader_eval<node_feature_mask>(kg, state, &sd, render_buffer, path_flag);

      /* Initialize additional RNG for BSDFs. */
      if (sd.flag & SD_BSDF_NEEDS_LCG) {
        sd.lcg_state = lcg_state_init(INTEGRATOR_STATE(state, path, rng_pixel),
                                      INTEGRATOR_STATE(state, path, rng_offset),
                                      INTEGRATOR_STATE(state, path, sample),
                                      0xb4bc3953);
      }
    }

#ifdef __SUBSURFACE__
    if (path_flag & PATH_RAY_SUBSURFACE) {
      /* When coming from inside subsurface scattering, setup a diffuse
       * closure to perform lighting at the exit point. */
      subsurface_shader_data_setup(kg, &sd);
      INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_SUBSURFACE;
    }
    else
#endif
    {
      /* Filter closures. */
      surface_shader_prepare_closures(kg, state, &sd, path_flag);

      /* Evaluate holdout. */
      if (!integrate_surface_holdout(kg, state, &sd, render_buffer)) {
        return LABEL_NONE;
      }

      /* Write emission. */
      if (sd.flag & SD_EMISSION) {
        integrate_surface_emission(kg, state, &sd, render_buffer);
      }

      /* Perform path termination. Most paths have already been terminated in
       * the intersect_closest kernel, this is just for emission and for dividing
       * throughput by the probability at the right moment.
       *
       * Also ensure we don't do it twice for SSS at both the entry and exit point. */
      if (integrate_surface_terminate(state, path_flag)) {
        return LABEL_NONE;
      }

      /* Write render passes. */
#ifdef __PASSES__
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_PASSES);
      film_write_data_passes(kg, state, &sd, render_buffer);
#endif

#ifdef __DENOISING_FEATURES__
      film_write_denoising_features_surface(kg, state, &sd, render_buffer);
#endif
    }

    /* Load random number state. */
    RNGState rng_state;
    path_state_rng_load(state, &rng_state);

#if defined(__PATH_GUIDING__) && PATH_GUIDING_LEVEL >= 4
    if (kernel_data.kernel_features & KERNEL_FEATURE_PATH_GUIDING) {
      surface_shader_prepare_guiding(kg, state, &sd, &rng_state);
      guiding_write_debug_passes(kg, state, &sd, render_buffer);
    }
#endif
    /* Direct light. */
    PROFILING_EVENT(PROFILING_SHADE_SURFACE_DIRECT_LIGHT);
    integrate_surface_direct_light<node_feature_mask>(kg, state, &sd, &rng_state);

#if defined(__AO__)
    /* Ambient occlusion pass. */
    if (kernel_data.kernel_features & KERNEL_FEATURE_AO) {
      PROFILING_EVENT(PROFILING_SHADE_SURFACE_AO);
      integrate_surface_ao(kg, state, &sd, &rng_state);
    }
#endif

    PROFILING_EVENT(PROFILING_SHADE_SURFACE_INDIRECT_LIGHT);
    continue_path_label = integrate_surface_bsdf_bssrdf_bounce(
        kg, state, &sd, &rng_state, render_buffer);
#ifdef __VOLUME__
  }
  else {
    if (integrate_surface_terminate(state, path_flag)) {
      return LABEL_NONE;
    }

    PROFILING_EVENT(PROFILING_SHADE_SURFACE_INDIRECT_LIGHT);
    continue_path_label = integrate_surface_volume_only_bounce(state, &sd);
  }

  if (continue_path_label & LABEL_TRANSMIT) {
    /* Enter/Exit volume. */
    volume_stack_enter_exit<false>(kg, state, &sd);
  }
#endif

  return continue_path_label;
}

template<DeviceKernel current_kernel>
ccl_device_forceinline void integrator_shade_surface_next_kernel(IntegratorState state)
{
  if (INTEGRATOR_STATE(state, path, flag) & PATH_RAY_SUBSURFACE) {
    integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE);
  }
  else {
    kernel_assert(INTEGRATOR_STATE(state, ray, tmax) != 0.0f);
    integrator_path_next(state, current_kernel, DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
  }
}

template<uint node_feature_mask = KERNEL_FEATURE_NODE_MASK_SURFACE & ~KERNEL_FEATURE_NODE_RAYTRACE,
         DeviceKernel current_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE>
ccl_device_forceinline void integrator_shade_surface(KernelGlobals kg,
                                                     IntegratorState state,
                                                     ccl_global float *ccl_restrict render_buffer)
{
  const int continue_path_label = integrate_surface<node_feature_mask>(kg, state, render_buffer);
  if (continue_path_label == LABEL_NONE) {
    integrator_path_terminate(kg, state, render_buffer, current_kernel);
    return;
  }

#ifdef __SHADOW_LINKING__
  /* No need to cast shadow linking rays at a transparent bounce: the lights will be accumulated
   * via the main path in this case. BSSRDF bounces continue with intersect_subsurface. */
  if ((continue_path_label & (LABEL_TRANSPARENT | LABEL_SUBSURFACE_SCATTER)) == 0) {
    if (shadow_linking_schedule_intersection_kernel<current_kernel>(kg, state)) {
      return;
    }
  }
#endif

  integrator_shade_surface_next_kernel<current_kernel>(state);
}

ccl_device_forceinline void integrator_shade_surface_raytrace(
    KernelGlobals kg, IntegratorState state, ccl_global float *ccl_restrict render_buffer)
{
  integrator_shade_surface<KERNEL_FEATURE_NODE_MASK_SURFACE,
                           DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE>(
      kg, state, render_buffer);
}

ccl_device_forceinline void integrator_shade_surface_mnee(
    KernelGlobals kg, IntegratorState state, ccl_global float *ccl_restrict render_buffer)
{
#ifdef __MNEE__
  integrator_shade_surface<(KERNEL_FEATURE_NODE_MASK_SURFACE & ~KERNEL_FEATURE_NODE_RAYTRACE) |
                               KERNEL_FEATURE_MNEE,
                           DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_MNEE>(kg, state, render_buffer);
#endif
}

CCL_NAMESPACE_END
