/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/object.h"

#include "device/device.h"
#include "scene/camera.h"
#include "scene/curves.h"
#include "scene/hair.h"
#include "scene/integrator.h"
#include "scene/light.h"
#include "scene/mesh.h"
#include "scene/particles.h"
#include "scene/pointcloud.h"
#include "scene/scene.h"
#include "scene/stats.h"
#include "scene/volume.h"

#include "util/log.h"
#include "util/map.h"
#include "util/math.h"

#include <algorithm>
#include "util/murmurhash.h"
#include "util/progress.h"
#include "util/set.h"
#include "util/tbb.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

/* Global state of object transform update. */

struct UpdateObjectTransformState {
  /* Global state used by device_update_object_transform().
   * Common for both threaded and non-threaded update.
   */

  /* Type of the motion required by the scene settings. */
  Scene::MotionType need_motion;

  /* Mapping from particle system to a index in packed particle array.
   * Only used for read.
   */
  map<ParticleSystem *, int> particle_offset;

  /* Motion offsets for each object. */
  array<uint> motion_offset;

  /* Packed object arrays. Those will be filled in. */
  uint *object_flag;
  uint *object_visibility;
  KernelObject *objects;
  Transform *object_motion_pass;
  DecomposedTransform *object_motion;

  /* Flags which will be synchronized to Integrator. */
  bool have_motion;
  bool have_curves;
  bool have_points;
  bool have_volumes;

  /* ** Scheduling queue. ** */
  Scene *scene;

  /* First unused object index in the queue. */
  int queue_start_object;
};

/* Object */

NODE_DEFINE(Object)
{
  NodeType *type = NodeType::add("object", create);

  SOCKET_NODE(geometry, "Geometry", Geometry::get_node_base_type());
  SOCKET_TRANSFORM(tfm, "Transform", transform_identity());
  SOCKET_UINT(visibility, "Visibility", ~0);
  SOCKET_COLOR(color, "Color", zero_float3());
  SOCKET_FLOAT(alpha, "Alpha", 0.0f);
  SOCKET_UINT(random_id, "Random ID", 0);
  SOCKET_INT(pass_id, "Pass ID", 0);
  SOCKET_BOOLEAN(use_holdout, "Use Holdout", false);
  SOCKET_BOOLEAN(hide_on_missing_motion, "Hide on Missing Motion", false);
  SOCKET_POINT(dupli_generated, "Dupli Generated", zero_float3());
  SOCKET_POINT2(dupli_uv, "Dupli UV", zero_float2());
  SOCKET_TRANSFORM_ARRAY(motion, "Motion", array<Transform>());
  SOCKET_FLOAT(shadow_terminator_shading_offset, "Shadow Terminator Shading Offset", 0.0f);
  SOCKET_FLOAT(shadow_terminator_geometry_offset, "Shadow Terminator Geometry Offset", 0.1f);
  SOCKET_STRING(asset_name, "Asset Name", ustring());

  SOCKET_BOOLEAN(is_shadow_catcher, "Shadow Catcher", false);

  SOCKET_BOOLEAN(is_caustics_caster, "Cast Shadow Caustics", false);
  SOCKET_BOOLEAN(is_caustics_receiver, "Receive Shadow Caustics", false);

  SOCKET_BOOLEAN(is_bake_target, "Bake Target", false);

  SOCKET_NODE(particle_system, "Particle System", ParticleSystem::get_node_type());
  SOCKET_INT(particle_index, "Particle Index", 0);

  SOCKET_FLOAT(ao_distance, "AO Distance", 0.0f);

  SOCKET_STRING(lightgroup, "Light Group", ustring());
  SOCKET_UINT(receiver_light_set, "Light Set Index", 0);
  SOCKET_UINT64(light_set_membership, "Light Set Membership", LIGHT_LINK_MASK_ALL);
  SOCKET_UINT(blocker_shadow_set, "Shadow Set Index", 0);
  SOCKET_UINT64(shadow_set_membership, "Shadow Set Membership", LIGHT_LINK_MASK_ALL);

  return type;
}

Object::Object() : Node(get_node_type())
{
  particle_system = nullptr;
  particle_index = 0;
  attr_map_offset = 0;
  bounds = BoundBox::empty;
  intersects_volume = false;
}

Object::~Object() = default;

void Object::update_motion()
{
  if (!use_motion()) {
    return;
  }

  bool have_motion = false;

  for (size_t i = 0; i < motion.size(); i++) {
    if (motion[i] == transform_empty()) {
      if (hide_on_missing_motion) {
        /* Hide objects that have no valid previous or next
         * transform, for example particle that stop existing. It
         * would be better to handle this in the kernel and make
         * objects invisible outside certain motion steps. */
        tfm = transform_empty();
        motion.clear();
        return;
      }
      /* Otherwise just copy center motion. */
      motion[i] = tfm;
    }

    /* Test if any of the transforms are actually different. */
    have_motion = have_motion || motion[i] != tfm;
  }

  /* Clear motion array if there is no actual motion. */
  if (!have_motion) {
    motion.clear();
  }
}

void Object::compute_bounds(bool motion_blur)
{
  const BoundBox mbounds = geometry->bounds;

  if (motion_blur && use_motion()) {
    array<DecomposedTransform> decomp(motion.size());
    transform_motion_decompose(decomp.data(), motion.data(), motion.size());

    bounds = BoundBox::empty;

    /* TODO: this is really terrible. according to PBRT there is a better
     * way to find this iteratively, but did not find implementation yet
     * or try to implement myself */
    for (float t = 0.0f; t < 1.0f; t += (1.0f / 128.0f)) {
      Transform ttfm;

      transform_motion_array_interpolate(&ttfm, decomp.data(), motion.size(), t);
      bounds.grow(mbounds.transformed(&ttfm));
    }
  }
  else {
    /* No motion blur case. */
    if (geometry->transform_applied) {
      bounds = mbounds;
    }
    else {
      bounds = mbounds.transformed(&tfm);
    }
  }
}

void Object::apply_transform(bool apply_to_motion)
{
  if (!geometry || tfm == transform_identity()) {
    return;
  }

  geometry->apply_transform(tfm, apply_to_motion);

  /* we keep normals pointing in same direction on negative scale, notify
   * geometry about this in it (re)calculates normals */
  if (transform_negative_scale(tfm)) {
    geometry->transform_negative_scaled = true;
  }

  if (bounds.valid()) {
    geometry->compute_bounds();
    compute_bounds(false);
  }

  /* tfm is not reset to identity, all code that uses it needs to check the
   * transform_applied boolean */
}

void Object::tag_update(Scene *scene)
{
  uint32_t flag = ObjectManager::UPDATE_NONE;

  if (is_modified()) {
    flag |= ObjectManager::OBJECT_MODIFIED;

    if (use_holdout_is_modified()) {
      flag |= ObjectManager::HOLDOUT_MODIFIED;
    }

    if (is_shadow_catcher_is_modified()) {
      scene->tag_shadow_catcher_modified();
      flag |= ObjectManager::VISIBILITY_MODIFIED;
    }
  }

  if (geometry) {
    if (tfm_is_modified() || motion_is_modified()) {
      flag |= ObjectManager::TRANSFORM_MODIFIED;
      if (geometry->has_volume) {
        scene->volume_manager->tag_update(this, flag);
      }
    }

    if (visibility_is_modified()) {
      flag |= ObjectManager::VISIBILITY_MODIFIED;
    }

    for (Node *node : geometry->get_used_shaders()) {
      Shader *shader = static_cast<Shader *>(node);
      if (shader->emission_sampling != EMISSION_SAMPLING_NONE) {
        scene->light_manager->tag_update(scene, LightManager::EMISSIVE_MESH_MODIFIED);
      }
    }
  }

  scene->camera->need_flags_update = true;
  scene->object_manager->tag_update(scene, flag);
}

bool Object::use_motion() const
{
  return (motion.size() > 1);
}

float Object::motion_time(const int step) const
{
  return (use_motion()) ? 2.0f * step / (motion.size() - 1) - 1.0f : 0.0f;
}

int Object::motion_step(const float time) const
{
  if (use_motion()) {
    for (size_t step = 0; step < motion.size(); step++) {
      if (time == motion_time(step)) {
        return step;
      }
    }
  }

  return -1;
}

bool Object::is_traceable() const
{
  /* Not supported for lights yet. */
  if (geometry->is_light()) {
    return false;
  }
  /* Mesh itself can be empty,can skip all such objects. */
  if (!bounds.valid() || bounds.size() == zero_float3()) {
    return false;
  }
  /* TODO(sergey): Check for mesh vertices/curves. visibility flags. */
  return true;
}

uint Object::visibility_for_tracing() const
{
  return SHADOW_CATCHER_OBJECT_VISIBILITY(is_shadow_catcher, visibility & PATH_RAY_ALL_VISIBILITY);
}

int Object::get_device_index() const
{
  return index;
}

bool Object::usable_as_light() const
{
  Geometry *geom = get_geometry();
  if (!geom->is_mesh() && !geom->is_volume()) {
    return false;
  }
  /* Skip non-traceable objects. */
  if (!is_traceable()) {
    return false;
  }
  /* Skip if we are not visible for BSDFs. */
  if (!(get_visibility() &
        (PATH_RAY_DIFFUSE | PATH_RAY_GLOSSY | PATH_RAY_TRANSMIT | PATH_RAY_VOLUME_SCATTER)))
  {
    return false;
  }
  /* Skip if we have no emission shaders. */
  /* TODO(sergey): Ideally we want to avoid such duplicated loop, since it'll
   * iterate all geometry shaders twice (when counting and when calculating
   * triangle area.
   */
  for (Node *node : geom->get_used_shaders()) {
    Shader *shader = static_cast<Shader *>(node);
    if (shader->emission_sampling != EMISSION_SAMPLING_NONE) {
      return true;
    }
  }
  return false;
}

bool Object::has_light_linking() const
{
  if (get_receiver_light_set()) {
    return true;
  }

  if (get_light_set_membership() != LIGHT_LINK_MASK_ALL) {
    return true;
  }

  return false;
}

bool Object::has_shadow_linking() const
{
  if (get_blocker_shadow_set()) {
    return true;
  }

  if (get_shadow_set_membership() != LIGHT_LINK_MASK_ALL) {
    return true;
  }

  return false;
}

/* Object Manager */

ObjectManager::ObjectManager()
{
  update_flags = UPDATE_ALL;
  need_flags_update = true;
}

ObjectManager::~ObjectManager() = default;

static float object_volume_density(const Transform &tfm, Geometry *geom)
{
  if (geom->is_volume()) {
    /* Volume density automatically adjust to object scale. */
    if (static_cast<Volume *>(geom)->get_object_space()) {
      const float3 unit = normalize(one_float3());
      return 1.0f / len(transform_direction(&tfm, unit));
    }
  }

  return 1.0f;
}

static int object_num_motion_verts(Geometry *geom)
{
  return (geom->is_mesh() || geom->is_volume()) ? static_cast<Mesh *>(geom)->get_verts().size() :
         geom->is_hair()       ? static_cast<Hair *>(geom)->get_curve_keys().size() :
         geom->is_pointcloud() ? static_cast<PointCloud *>(geom)->num_points() :
                                 0;
}

void ObjectManager::device_update_object_transform(UpdateObjectTransformState *state,
                                                   Object *ob,
                                                   bool update_all,
                                                   const Scene *scene)
{
  KernelObject &kobject = state->objects[ob->index];
  Transform *object_motion_pass = state->object_motion_pass;

  Geometry *geom = ob->geometry;
  uint flag = 0;

  /* Compute transformations. */
  const Transform tfm = ob->tfm;
  const Transform itfm = transform_inverse(tfm);

  const float3 color = ob->color;
  const float pass_id = ob->pass_id;
  const float random_number = (float)ob->random_id * (1.0f / (float)0xFFFFFFFF);
  const int particle_index = (ob->particle_system) ?
                                 ob->particle_index + state->particle_offset[ob->particle_system] :
                                 0;

  kobject.tfm = tfm;
  kobject.itfm = itfm;
  kobject.volume_density = object_volume_density(tfm, geom);
  kobject.color[0] = color.x;
  kobject.color[1] = color.y;
  kobject.color[2] = color.z;
  kobject.alpha = ob->alpha;
  kobject.pass_id = pass_id;
  kobject.random_number = random_number;
  kobject.particle_index = particle_index;
  kobject.motion_offset = 0;
  kobject.ao_distance = ob->ao_distance;
  kobject.receiver_light_set = ob->receiver_light_set >= LIGHT_LINK_SET_MAX ?
                                   0 :
                                   ob->receiver_light_set;
  kobject.light_set_membership = ob->light_set_membership;
  kobject.blocker_shadow_set = ob->blocker_shadow_set >= LIGHT_LINK_SET_MAX ?
                                   0 :
                                   ob->blocker_shadow_set;
  kobject.shadow_set_membership = ob->shadow_set_membership;

  if (geom->get_use_motion_blur()) {
    state->have_motion = true;
  }

  if (transform_negative_scale(tfm)) {
    flag |= SD_OBJECT_NEGATIVE_SCALE;
  }

  /* TODO: why not check hair? */
  if (geom->is_pointcloud()) {
    if (geom->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION)) {
      flag |= SD_OBJECT_HAS_VERTEX_MOTION;
    }
  }
  else if (geom->is_mesh()) {
    Mesh *mesh = static_cast<Mesh *>(geom);
    if (mesh->attributes.find(ATTR_STD_MOTION_VERTEX_POSITION) ||
        (mesh->get_subdivision_type() != Mesh::SUBDIVISION_NONE &&
         mesh->subd_attributes.find(ATTR_STD_MOTION_VERTEX_POSITION)))
    {
      flag |= SD_OBJECT_HAS_VERTEX_MOTION;
    }
  }
  else if (geom->is_volume()) {
    Volume *volume = static_cast<Volume *>(geom);
    if (volume->attributes.find(ATTR_STD_VOLUME_VELOCITY) && volume->get_velocity_scale() != 0.0f)
    {
      flag |= SD_OBJECT_HAS_VOLUME_MOTION;
      kobject.velocity_scale = volume->get_velocity_scale();
    }
  }

  if (state->need_motion == Scene::MOTION_PASS) {
    /* Clear motion array if there is no actual motion. */
    ob->update_motion();

    /* Compute motion transforms. */
    Transform tfm_pre;
    Transform tfm_post;
    if (ob->use_motion()) {
      tfm_pre = ob->motion[0];
      tfm_post = ob->motion[ob->motion.size() - 1];
    }
    else {
      tfm_pre = tfm;
      tfm_post = tfm;
    }

    /* Motion transformations, is world/object space depending if mesh
     * comes with deformed position in object space, or if we transform
     * the shading point in world space. */
    if (!(flag & SD_OBJECT_HAS_VERTEX_MOTION)) {
      tfm_pre = tfm_pre * itfm;
      tfm_post = tfm_post * itfm;
    }

    const int motion_pass_offset = ob->index * OBJECT_MOTION_PASS_SIZE;
    object_motion_pass[motion_pass_offset + 0] = tfm_pre;
    object_motion_pass[motion_pass_offset + 1] = tfm_post;
  }
  else if (state->need_motion == Scene::MOTION_BLUR) {
    if (ob->use_motion()) {
      kobject.motion_offset = state->motion_offset[ob->index];

      /* Decompose transforms for interpolation. */
      if (ob->tfm_is_modified() || ob->motion_is_modified() || update_all) {
        DecomposedTransform *decomp = state->object_motion + kobject.motion_offset;
        transform_motion_decompose(decomp, ob->motion.data(), ob->motion.size());
      }

      flag |= SD_OBJECT_MOTION;
      state->have_motion = true;
    }
  }

  /* Dupli object coords and motion info. */
  kobject.dupli_generated[0] = ob->dupli_generated[0];
  kobject.dupli_generated[1] = ob->dupli_generated[1];
  kobject.dupli_generated[2] = ob->dupli_generated[2];
  kobject.dupli_uv[0] = ob->dupli_uv[0];
  kobject.dupli_uv[1] = ob->dupli_uv[1];
  kobject.num_geom_steps = (geom->get_motion_steps() - 1) / 2;
  kobject.num_tfm_steps = ob->motion.size();
  kobject.numverts = object_num_motion_verts(geom);
  kobject.attribute_map_offset = 0;

  if (ob->asset_name_is_modified() || update_all) {
    const uint32_t hash_name = util_murmur_hash3(ob->name.c_str(), ob->name.length(), 0);
    const uint32_t hash_asset = util_murmur_hash3(
        ob->asset_name.c_str(), ob->asset_name.length(), 0);
    kobject.cryptomatte_object = util_hash_to_float(hash_name);
    kobject.cryptomatte_asset = util_hash_to_float(hash_asset);
  }

  kobject.shadow_terminator_shading_offset = 1.0f /
                                             (1.0f - 0.5f * ob->shadow_terminator_shading_offset);
  kobject.shadow_terminator_geometry_offset = ob->shadow_terminator_geometry_offset;

  kobject.visibility = ob->visibility_for_tracing();
  kobject.primitive_type = geom->primitive_type();

  /* Object shadow caustics flag */
  if (ob->is_caustics_caster) {
    flag |= SD_OBJECT_CAUSTICS_CASTER;
  }
  if (ob->is_caustics_receiver) {
    flag |= SD_OBJECT_CAUSTICS_RECEIVER;
  }

  /* Object flag. */
  if (ob->use_holdout) {
    flag |= SD_OBJECT_HOLDOUT_MASK;
  }
  state->object_flag[ob->index] = flag;

  /* Have curves. */
  if (geom->is_hair()) {
    state->have_curves = true;
  }
  if (geom->is_pointcloud()) {
    state->have_points = true;
  }
  if (geom->is_volume()) {
    state->have_volumes = true;
  }

  /* Light group. */
  auto it = scene->lightgroups.find(ob->lightgroup);
  if (it != scene->lightgroups.end()) {
    kobject.lightgroup = it->second;
  }
  else {
    kobject.lightgroup = LIGHTGROUP_NONE;
  }
}

void ObjectManager::device_update_prim_offsets(Device *device, DeviceScene *dscene, Scene *scene)
{
  if (!scene->integrator->get_use_light_tree()) {
    const BVHLayoutMask layout_mask = device->get_bvh_layout_mask(dscene->data.kernel_features);
    if (layout_mask != BVH_LAYOUT_METAL && layout_mask != BVH_LAYOUT_MULTI_METAL &&
        layout_mask != BVH_LAYOUT_MULTI_METAL_EMBREE && layout_mask != BVH_LAYOUT_HIPRT &&
        layout_mask != BVH_LAYOUT_MULTI_HIPRT && layout_mask != BVH_LAYOUT_MULTI_HIPRT_EMBREE)
    {
      return;
    }
  }

  /* On MetalRT, primitive / curve segment offsets can't be baked at BVH build time. Intersection
   * handlers need to apply the offset manually. */
  uint *object_prim_offset = dscene->object_prim_offset.alloc(scene->objects.size());
  for (Object *ob : scene->objects) {
    uint32_t prim_offset = 0;
    if (Geometry *const geom = ob->geometry) {
      if (geom->is_hair()) {
        prim_offset = ((Hair *const)geom)->curve_segment_offset;
      }
      else {
        prim_offset = geom->prim_offset;
      }
    }
    const uint obj_index = ob->get_device_index();
    object_prim_offset[obj_index] = prim_offset;
  }

  dscene->object_prim_offset.copy_to_device();
  dscene->object_prim_offset.clear_modified();
}

/* ======================================================================
 * Specular Polynomials: 4-ary tree building for hierarchical pruning.
 *
 * Each caustic caster mesh gets a 4-ary spatial tree built via double-median
 * splits. The tree enables efficient interval-arithmetic pruning on the GPU,
 * reducing solver calls from O(N) to O(log4(N)) per pixel.
 *
 * Tree node layout in flat float4 array (4 float4s per node):
 *   [i*4+0] = (pos_min.x, pos_min.y, pos_min.z, pos_max.x)
 *   [i*4+1] = (pos_max.y, pos_max.z, nor_min.x, nor_min.y)
 *   [i*4+2] = (nor_min.z, nor_max.x, nor_max.y, nor_max.z)
 *   [i*4+3] = (int_as_float(child_offset), int_as_float(num_children),
 *              int_as_float(triangle_prim), pos_area)
 *
 * Leaf nodes: num_children=0, triangle_prim >= 0, pos_area = -1
 * Internal nodes: num_children=1..4, triangle_prim = -1, pos_area >= 0
 * Children of a node are contiguous: child_offset .. child_offset+num_children-1
 * ====================================================================== */

namespace {

struct SPolyTriData {
  float3 centroid;
  float3 verts[3];
  float3 normals[3];
  int global_prim;
};

struct SPolyBuildNode {
  float3 pos_min, pos_max;
  float3 nor_min, nor_max;
  int triangle_prim; /* global prim for leaf, -1 for internal */
  float pos_area;    /* AABB surface area for internal, -1 for leaf */
  int children[4];   /* indices into build_nodes, -1 if unused */
  int num_children;
};

static int spoly_largest_axis(float3 extent)
{
  if (extent.x >= extent.y && extent.x >= extent.z)
    return 0;
  if (extent.y >= extent.z)
    return 1;
  return 2;
}

static float spoly_centroid_component(const SPolyTriData &tri, int axis)
{
  return (&tri.centroid.x)[axis];
}

static void spoly_compute_bounds(SPolyBuildNode &node,
                                 const vector<SPolyTriData> &tris,
                                 int start,
                                 int count)
{
  node.pos_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  node.pos_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  node.nor_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  node.nor_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  for (int i = start; i < start + count; i++) {
    for (int v = 0; v < 3; v++) {
      node.pos_min = min(node.pos_min, tris[i].verts[v]);
      node.pos_max = max(node.pos_max, tris[i].verts[v]);
      node.nor_min = min(node.nor_min, tris[i].normals[v]);
      node.nor_max = max(node.nor_max, tris[i].normals[v]);
    }
  }

  float3 ext = node.pos_max - node.pos_min;
  node.pos_area = 2.0f * (ext.x * ext.y + ext.y * ext.z + ext.z * ext.x);
}

/* Recursively build a 4-ary tree via double-median splits.
 * Returns node index in the build_nodes array. */
static int spoly_build_tree(vector<SPolyBuildNode> &nodes,
                            vector<SPolyTriData> &tris,
                            int start,
                            int count)
{
  if (count <= 0)
    return -1;

  int node_idx = (int)nodes.size();
  nodes.emplace_back();

  SPolyBuildNode &node = nodes[node_idx];
  node.triangle_prim = -1;
  node.num_children = 0;
  for (int i = 0; i < 4; i++)
    node.children[i] = -1;

  if (count <= 4) {
    /* Base case: create internal node with up to 4 leaf children. */
    spoly_compute_bounds(node, tris, start, count);

    for (int i = 0; i < count; i++) {
      int leaf_idx = (int)nodes.size();
      nodes.emplace_back();
      SPolyBuildNode &leaf = nodes[leaf_idx];

      const SPolyTriData &tri = tris[start + i];
      leaf.pos_min = min(min(tri.verts[0], tri.verts[1]), tri.verts[2]);
      leaf.pos_max = max(max(tri.verts[0], tri.verts[1]), tri.verts[2]);
      leaf.nor_min = min(min(tri.normals[0], tri.normals[1]), tri.normals[2]);
      leaf.nor_max = max(max(tri.normals[0], tri.normals[1]), tri.normals[2]);
      leaf.triangle_prim = tri.global_prim;
      leaf.pos_area = -1.0f;
      leaf.num_children = 0;
      for (int j = 0; j < 4; j++)
        leaf.children[j] = -1;

      /* Note: nodes[node_idx] may have been invalidated by emplace_back,
       * so we access through node_idx at the end. */
    }

    /* Re-access after all emplace_backs. */
    nodes[node_idx].num_children = count;
    /* Children are at node_idx+1 .. node_idx+count. */
    for (int i = 0; i < count; i++) {
      nodes[node_idx].children[i] = node_idx + 1 + i;
    }

    return node_idx;
  }

  /* Recursive case: double-median split into 4 groups. */

  /* Find centroid AABB and largest axis. */
  float3 cmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 cmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  for (int i = start; i < start + count; i++) {
    cmin = min(cmin, tris[i].centroid);
    cmax = max(cmax, tris[i].centroid);
  }
  int axis0 = spoly_largest_axis(cmax - cmin);

  /* Split at median along primary axis. */
  int mid = start + count / 2;
  std::nth_element(
      tris.begin() + start,
      tris.begin() + mid,
      tris.begin() + start + count,
      [axis0](const SPolyTriData &a, const SPolyTriData &b) {
        return spoly_centroid_component(a, axis0) < spoly_centroid_component(b, axis0);
      });

  /* Split left half [start, mid) along its largest axis. */
  float3 lcmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 lcmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  for (int i = start; i < mid; i++) {
    lcmin = min(lcmin, tris[i].centroid);
    lcmax = max(lcmax, tris[i].centroid);
  }
  int axis1 = spoly_largest_axis(lcmax - lcmin);
  int left_mid = start + (mid - start) / 2;
  std::nth_element(
      tris.begin() + start,
      tris.begin() + left_mid,
      tris.begin() + mid,
      [axis1](const SPolyTriData &a, const SPolyTriData &b) {
        return spoly_centroid_component(a, axis1) < spoly_centroid_component(b, axis1);
      });

  /* Split right half [mid, start+count) along its largest axis. */
  float3 rcmin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 rcmax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  for (int i = mid; i < start + count; i++) {
    rcmin = min(rcmin, tris[i].centroid);
    rcmax = max(rcmax, tris[i].centroid);
  }
  int axis2 = spoly_largest_axis(rcmax - rcmin);
  int right_mid = mid + (start + count - mid) / 2;
  std::nth_element(
      tris.begin() + mid,
      tris.begin() + right_mid,
      tris.begin() + start + count,
      [axis2](const SPolyTriData &a, const SPolyTriData &b) {
        return spoly_centroid_component(a, axis2) < spoly_centroid_component(b, axis2);
      });

  /* 4 groups: [start,left_mid), [left_mid,mid), [mid,right_mid), [right_mid,start+count) */
  int g_start[4] = {start, left_mid, mid, right_mid};
  int g_count[4] = {left_mid - start,
                    mid - left_mid,
                    right_mid - mid,
                    start + count - right_mid};

  /* Compute bounds for this node from all triangles. */
  spoly_compute_bounds(nodes[node_idx], tris, start, count);
  nodes[node_idx].triangle_prim = -1;

  /* Recursively build children. */
  int child_count = 0;
  for (int g = 0; g < 4; g++) {
    if (g_count[g] > 0) {
      int child_idx = spoly_build_tree(nodes, tris, g_start[g], g_count[g]);
      nodes[node_idx].children[child_count] = child_idx;
      child_count++;
    }
  }
  nodes[node_idx].num_children = child_count;

  return node_idx;
}

/* Serialize the pointer-based build tree into a flat float4 array using BFS.
 * BFS ensures that children of each node are contiguous in the output.
 * Returns the number of nodes written.
 * flat_offset is the starting index in the global flat array for this tree. */
static int spoly_flatten_tree(const vector<SPolyBuildNode> &build_nodes,
                              int root_idx,
                              vector<float4> &flat,
                              int flat_offset)
{
  if (root_idx < 0)
    return 0;

  /* BFS: map from build_node index to flat node index. */
  vector<int> bfs_order;
  bfs_order.reserve(build_nodes.size());
  bfs_order.push_back(root_idx);

  size_t head = 0;
  while (head < bfs_order.size()) {
    int bi = bfs_order[head++];
    const SPolyBuildNode &bn = build_nodes[bi];
    for (int c = 0; c < bn.num_children; c++) {
      if (bn.children[c] >= 0) {
        bfs_order.push_back(bn.children[c]);
      }
    }
  }

  /* Build mapping from build index to flat index. */
  map<int, int> build_to_flat;
  for (size_t i = 0; i < bfs_order.size(); i++) {
    build_to_flat[bfs_order[i]] = flat_offset + (int)i;
  }

  /* Write nodes in BFS order. Each node = 4 float4s. */
  int num_nodes = (int)bfs_order.size();
  size_t base = flat.size();
  flat.resize(base + num_nodes * 4);

  for (size_t i = 0; i < bfs_order.size(); i++) {
    const SPolyBuildNode &bn = build_nodes[bfs_order[i]];
    size_t idx = base + i * 4;

    /* Find flat child offset (first child in BFS order). */
    int child_flat_offset = 0;
    if (bn.num_children > 0 && bn.children[0] >= 0) {
      child_flat_offset = build_to_flat[bn.children[0]];
    }

    flat[idx + 0] = make_float4(
        bn.pos_min.x, bn.pos_min.y, bn.pos_min.z, bn.pos_max.x);
    flat[idx + 1] = make_float4(
        bn.pos_max.y, bn.pos_max.z, bn.nor_min.x, bn.nor_min.y);
    flat[idx + 2] = make_float4(
        bn.nor_min.z, bn.nor_max.x, bn.nor_max.y, bn.nor_max.z);
    flat[idx + 3] = make_float4(
        __int_as_float(child_flat_offset),
        __int_as_float(bn.num_children),
        __int_as_float(bn.triangle_prim),
        bn.pos_area);
  }

  return num_nodes;
}

} /* anonymous namespace */

void ObjectManager::device_update_spoly_casters(DeviceScene *dscene, Scene *scene)
{
  /* Build 4-ary trees for all caustic caster meshes and upload to GPU.
   * Each caster gets its own tree stored contiguously in spoly_tree_nodes. */
  vector<uint> caster_indices;
  vector<uint> caster_tree_offsets;
  vector<float4> all_tree_nodes;

  int total_node_offset = 0;

  for (Object *ob : scene->objects) {
    if (!ob->get_is_caustics_caster()) {
      continue;
    }
    Geometry *geom = ob->geometry;
    if (!geom || geom->is_hair() || geom->is_pointcloud() || geom->is_volume()) {
      continue;
    }
    Mesh *mesh = static_cast<Mesh *>(geom);
    const size_t num_tris = mesh->num_triangles();
    if (num_tris == 0) {
      continue;
    }

    /* Get object transform. */
    const Transform tfm = ob->get_tfm();
    const Transform itfm = transform_inverse(tfm);
    const bool do_transform = !geom->transform_applied;

    /* Get vertex normals. */
    Attribute *attr_vN = mesh->attributes.find(ATTR_STD_VERTEX_NORMAL);
    if (attr_vN == nullptr) {
      continue;
    }
    const float3 *vN = attr_vN->data_float3();
    const float3 *verts_data = mesh->verts.data();
    const int *tri_data = mesh->triangles.data();
    const int prim_offset = (int)geom->prim_offset;

    /* Collect triangle data in world space. */
    vector<SPolyTriData> tri_infos(num_tris);
    for (size_t ti = 0; ti < num_tris; ti++) {
      SPolyTriData &td = tri_infos[ti];
      const int v0 = tri_data[ti * 3 + 0];
      const int v1 = tri_data[ti * 3 + 1];
      const int v2 = tri_data[ti * 3 + 2];

      td.verts[0] = verts_data[v0];
      td.verts[1] = verts_data[v1];
      td.verts[2] = verts_data[v2];
      td.normals[0] = vN[v0];
      td.normals[1] = vN[v1];
      td.normals[2] = vN[v2];

      if (do_transform) {
        for (int v = 0; v < 3; v++) {
          td.verts[v] = transform_point(&tfm, td.verts[v]);
          td.normals[v] = safe_normalize(transform_direction_transposed(&itfm, td.normals[v]));
        }
      }
      else {
        for (int v = 0; v < 3; v++) {
          td.normals[v] = safe_normalize(td.normals[v]);
        }
      }

      td.centroid = (td.verts[0] + td.verts[1] + td.verts[2]) * (1.0f / 3.0f);
      td.global_prim = prim_offset + (int)ti;
    }

    /* Build 4-ary tree. */
    vector<SPolyBuildNode> build_nodes;
    build_nodes.reserve(num_tris * 2);
    int root = spoly_build_tree(build_nodes, tri_infos, 0, (int)num_tris);

    /* Record caster info. */
    caster_indices.push_back(ob->get_device_index());
    caster_tree_offsets.push_back((uint)total_node_offset);

    /* Flatten and append to global array. */
    int num_flat_nodes = spoly_flatten_tree(build_nodes, root, all_tree_nodes, total_node_offset);
    total_node_offset += num_flat_nodes;
  }

  const size_t num_casters = caster_indices.size();
  dscene->data.integrator.num_spoly_caster_objects = (int)num_casters;

  if (num_casters > 0) {
    uint *idx = dscene->spoly_caster_object_index.alloc(num_casters);
    uint *off = dscene->spoly_caster_tree_offset.alloc(num_casters);
    for (size_t i = 0; i < num_casters; i++) {
      idx[i] = caster_indices[i];
      off[i] = caster_tree_offsets[i];
    }
    dscene->spoly_caster_object_index.copy_to_device();
    dscene->spoly_caster_tree_offset.copy_to_device();

    /* Upload tree nodes. */
    const size_t num_float4s = all_tree_nodes.size();
    if (num_float4s > 0) {
      float4 *nodes_data = dscene->spoly_tree_nodes.alloc(num_float4s);
      memcpy(nodes_data, all_tree_nodes.data(), num_float4s * sizeof(float4));
      dscene->spoly_tree_nodes.copy_to_device();
    }
  }
  else {
    dscene->spoly_caster_object_index.free();
    dscene->spoly_caster_tree_offset.free();
    dscene->spoly_tree_nodes.free();
  }
}

void ObjectManager::device_update_transforms(DeviceScene *dscene, Scene *scene, Progress &progress)
{
  UpdateObjectTransformState state;
  state.need_motion = scene->need_motion();
  state.have_motion = false;
  state.have_curves = false;
  state.have_points = false;
  state.have_volumes = false;
  state.scene = scene;
  state.queue_start_object = 0;

  state.objects = dscene->objects.alloc(scene->objects.size());
  state.object_flag = dscene->object_flag.alloc(scene->objects.size());
  state.object_motion = nullptr;
  state.object_motion_pass = nullptr;

  if (state.need_motion == Scene::MOTION_PASS) {
    state.object_motion_pass = dscene->object_motion_pass.alloc(OBJECT_MOTION_PASS_SIZE *
                                                                scene->objects.size());
  }
  else if (state.need_motion == Scene::MOTION_BLUR) {
    /* Set object offsets into global object motion array. */
    uint *motion_offsets = state.motion_offset.resize(scene->objects.size());
    uint motion_offset = 0;

    for (Object *ob : scene->objects) {
      *motion_offsets = motion_offset;
      motion_offsets++;

      /* Clear motion array if there is no actual motion. */
      ob->update_motion();
      motion_offset += ob->motion.size();
    }

    state.object_motion = dscene->object_motion.alloc(motion_offset);
  }

  /* Particle system device offsets
   * 0 is dummy particle, index starts at 1.
   */
  int numparticles = 1;
  for (ParticleSystem *psys : scene->particle_systems) {
    state.particle_offset[psys] = numparticles;
    numparticles += psys->particles.size();
  }

  /* as all the arrays are the same size, checking only dscene.objects is sufficient */
  const bool update_all = dscene->objects.need_realloc();

  /* Parallel object update, with grain size to avoid too much threading overhead
   * for individual objects. */
  static const int OBJECTS_PER_TASK = 32;
  parallel_for(blocked_range<size_t>(0, scene->objects.size(), OBJECTS_PER_TASK),
               [&](const blocked_range<size_t> &r) {
                 for (size_t i = r.begin(); i != r.end(); i++) {
                   Object *ob = state.scene->objects[i];
                   device_update_object_transform(&state, ob, update_all, scene);
                 }
               });

  if (progress.get_cancel()) {
    return;
  }

  dscene->objects.copy_to_device_if_modified();
  if (state.need_motion == Scene::MOTION_PASS) {
    dscene->object_motion_pass.copy_to_device();
  }
  else if (state.need_motion == Scene::MOTION_BLUR) {
    dscene->object_motion.copy_to_device();
  }

  dscene->data.bvh.have_motion = state.have_motion;
  dscene->data.bvh.have_curves = state.have_curves;
  dscene->data.bvh.have_points = state.have_points;
  dscene->data.bvh.have_volumes = state.have_volumes;

  dscene->objects.clear_modified();
  dscene->object_motion_pass.clear_modified();
  dscene->object_motion.clear_modified();
}

void ObjectManager::device_update(Device *device,
                                  DeviceScene *dscene,
                                  Scene *scene,
                                  Progress &progress)
{
  if (!need_update()) {
    return;
  }

  if (update_flags & (OBJECT_ADDED | OBJECT_REMOVED)) {
    dscene->objects.tag_realloc();
    dscene->object_motion_pass.tag_realloc();
    dscene->object_motion.tag_realloc();
    dscene->object_flag.tag_realloc();

    /* If objects are added to the scene or deleted, the object indices might change, so we need to
     * update the root indices of the volume octrees. */
    scene->volume_manager->tag_update_indices();
  }

  if (update_flags & HOLDOUT_MODIFIED) {
    dscene->object_flag.tag_modified();
  }

  if (update_flags & PARTICLE_MODIFIED) {
    dscene->objects.tag_modified();
  }

  LOG_INFO << "Total " << scene->objects.size() << " objects.";

  device_free(device, dscene, false);

  if (scene->objects.empty()) {
    return;
  }

  {
    /* Assign object IDs. */
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->object.times.add_entry({"device_update (assign index)", time});
      }
    });

    int index = 0;
    for (Object *object : scene->objects) {
      object->index = index++;

      /* this is a bit too broad, however a bigger refactor might be needed to properly separate
       * update each type of data (transform, flags, etc.) */
      if (object->is_modified()) {
        dscene->objects.tag_modified();
        dscene->object_motion_pass.tag_modified();
        dscene->object_motion.tag_modified();
        dscene->object_flag.tag_modified();
      }

      /* Update world object index. */
      if (!object->get_geometry()->is_light()) {
        continue;
      }

      const Light *light = static_cast<const Light *>(object->get_geometry());
      if (light->get_light_type() == LIGHT_BACKGROUND) {
        dscene->data.background.object_index = object->index;
      }
    }
  }

  {
    /* set object transform matrices, before applying static transforms */
    const scoped_callback_timer timer([scene](double time) {
      if (scene->update_stats) {
        scene->update_stats->object.times.add_entry(
            {"device_update (copy objects to device)", time});
      }
    });

    progress.set_status("Updating Objects", "Copying Transformations to device");
    device_update_transforms(dscene, scene, progress);
  }

  /* Build specular polynomial caustic caster list. */
  device_update_spoly_casters(dscene, scene);

  for (Object *object : scene->objects) {
    object->clear_modified();
  }
}

void ObjectManager::device_update_flags(Device * /*unused*/,
                                        DeviceScene *dscene,
                                        Scene *scene,
                                        Progress & /*progress*/,
                                        bool bounds_valid)
{
  if (!need_update() && !need_flags_update) {
    return;
  }

  const scoped_callback_timer timer([scene](double time) {
    if (scene->update_stats) {
      scene->update_stats->object.times.add_entry({"device_update_flags", time});
    }
  });

  if (bounds_valid) {
    /* Object flags and calculations related to volume depend on proper bounds calculated, which
     * might not be available yet when object flags are updated for displacement or hair
     * transparency calculation. In this case do not clear the need_flags_update, so that these
     * values which depend on bounds are re-calculated when the device_update process comes back
     * here from the "Updating Objects Flags" stage. */
    update_flags = UPDATE_NONE;
    need_flags_update = false;
  }

  if (scene->objects.empty()) {
    return;
  }

  /* Object info flag. */
  uint *object_flag = dscene->object_flag.data();

  /* Object volume intersection. */
  vector<Object *> volume_objects;
  bool has_volume_objects = false;
  for (Object *object : scene->objects) {
    if (object->geometry->has_volume) {
      if (bounds_valid) {
        volume_objects.push_back(object);
      }
      has_volume_objects = true;
    }
  }

  for (Object *object : scene->objects) {
    if (object->geometry->has_volume) {
      object_flag[object->index] |= SD_OBJECT_HAS_VOLUME;
      object_flag[object->index] &= ~SD_OBJECT_HAS_VOLUME_ATTRIBUTES;

      for (const Attribute &attr : object->geometry->attributes.attributes) {
        if (attr.element == ATTR_ELEMENT_VOXEL) {
          object_flag[object->index] |= SD_OBJECT_HAS_VOLUME_ATTRIBUTES;
        }
      }
    }
    else {
      object_flag[object->index] &= ~(SD_OBJECT_HAS_VOLUME | SD_OBJECT_HAS_VOLUME_ATTRIBUTES);
    }

    if (object->is_shadow_catcher) {
      object_flag[object->index] |= SD_OBJECT_SHADOW_CATCHER;
    }
    else {
      object_flag[object->index] &= ~SD_OBJECT_SHADOW_CATCHER;
    }

    if (bounds_valid) {
      object->intersects_volume = false;
      for (Object *volume_object : volume_objects) {
        if (object == volume_object) {
          continue;
        }
        if (object->bounds.intersects(volume_object->bounds)) {
          object_flag[object->index] |= SD_OBJECT_INTERSECTS_VOLUME;
          object->intersects_volume = true;
          break;
        }
      }
    }
    else if (has_volume_objects) {
      /* Not really valid, but can't make more reliable in the case
       * of bounds not being up to date.
       */
      object_flag[object->index] |= SD_OBJECT_INTERSECTS_VOLUME;
    }
  }

  /* Copy object flag. */
  dscene->object_flag.copy_to_device();

  dscene->object_flag.clear_modified();
}

void ObjectManager::device_update_geom_offsets(Device * /*unused*/,
                                               DeviceScene *dscene,
                                               Scene *scene)
{
  if (dscene->objects.size() == 0) {
    return;
  }

  KernelObject *kobjects = dscene->objects.data();

  bool update = false;

  for (Object *object : scene->objects) {
    Geometry *geom = object->geometry;

    size_t attr_map_offset = object->attr_map_offset;

    /* An object attribute map cannot have a zero offset because mesh maps come first. */
    if (attr_map_offset == 0) {
      attr_map_offset = geom->attr_map_offset;
    }

    KernelObject &kobject = kobjects[object->index];

    if (kobject.attribute_map_offset != attr_map_offset) {
      kobject.attribute_map_offset = attr_map_offset;
      update = true;
    }

    const int numverts = object_num_motion_verts(geom);
    if (kobject.numverts != numverts) {
      kobject.numverts = numverts;
      update = true;
    }
  }

  if (update) {
    dscene->objects.copy_to_device();
  }
}

void ObjectManager::device_free(Device * /*unused*/, DeviceScene *dscene, bool force_free)
{
  dscene->objects.free_if_need_realloc(force_free);
  dscene->object_motion_pass.free_if_need_realloc(force_free);
  dscene->object_motion.free_if_need_realloc(force_free);
  dscene->object_flag.free_if_need_realloc(force_free);
  dscene->object_prim_offset.free_if_need_realloc(force_free);
  dscene->spoly_caster_object_index.free_if_need_realloc(force_free);
  dscene->spoly_caster_tree_offset.free_if_need_realloc(force_free);
  dscene->spoly_tree_nodes.free_if_need_realloc(force_free);
}

void ObjectManager::apply_static_transforms(DeviceScene *dscene, Scene *scene, Progress &progress)
{
  /* todo: normals and displacement should be done before applying transform! */
  /* todo: create objects/geometry in right order! */

  /* counter geometry users */
  map<Geometry *, int> geometry_users;
  const Scene::MotionType need_motion = scene->need_motion();
  const bool motion_blur = need_motion == Scene::MOTION_BLUR;
  const bool apply_to_motion = need_motion != Scene::MOTION_PASS;
  int i = 0;

  for (Object *object : scene->objects) {
    const map<Geometry *, int>::iterator it = geometry_users.find(object->geometry);

    if (it == geometry_users.end()) {
      geometry_users[object->geometry] = 1;
    }
    else {
      it->second++;
    }
  }

  if (progress.get_cancel()) {
    return;
  }

  uint *object_flag = dscene->object_flag.data();

  /* apply transforms for objects with single user geometry */
  for (Object *object : scene->objects) {
    /* Annoying feedback loop here: we can't use is_instanced() because
     * it'll use uninitialized transform_applied flag.
     *
     * Could be solved by moving reference counter to Geometry.
     */
    Geometry *geom = object->geometry;
    bool apply = (geometry_users[geom] == 1) && !geom->has_surface_bssrdf &&
                 !geom->has_true_displacement();

    if (geom->is_mesh()) {
      Mesh *mesh = static_cast<Mesh *>(geom);
      apply = apply && mesh->get_subdivision_type() == Mesh::SUBDIVISION_NONE;
    }
    else if (geom->is_hair() || geom->is_pointcloud()) {
      /* Can't apply non-uniform scale to curves and points, this can't be
       * represented by control points and radius alone. */
      float scale;
      apply = apply && transform_uniform_scale(object->tfm, scale);
    }

    if (apply) {
      if (!(motion_blur && object->use_motion())) {
        if (!geom->transform_applied) {
          object->apply_transform(apply_to_motion);
          geom->transform_applied = true;

          if (progress.get_cancel()) {
            return;
          }
        }

        object_flag[i] |= SD_OBJECT_TRANSFORM_APPLIED;
      }
    }

    i++;
  }
}

void ObjectManager::tag_update(Scene *scene, const uint32_t flag)
{
  update_flags |= flag;

  /* avoid infinite loops if the geometry manager tagged us for an update */
  if ((flag & GEOMETRY_MANAGER) == 0) {
    uint32_t geometry_flag = GeometryManager::OBJECT_MANAGER;

    /* Also notify in case added or removed objects were instances, as no Geometry might have been
     * added or removed, but the BVH still needs to updated. */
    if ((flag & (OBJECT_ADDED | OBJECT_REMOVED)) != 0) {
      geometry_flag |= (GeometryManager::GEOMETRY_ADDED | GeometryManager::GEOMETRY_REMOVED);
    }

    if ((flag & TRANSFORM_MODIFIED) != 0) {
      geometry_flag |= GeometryManager::TRANSFORM_MODIFIED;
    }

    if ((flag & VISIBILITY_MODIFIED) != 0) {
      geometry_flag |= GeometryManager::VISIBILITY_MODIFIED;
    }

    scene->geometry_manager->tag_update(scene, geometry_flag);
  }

  scene->light_manager->tag_update(scene, LightManager::OBJECT_MANAGER);

  /* Integrator's shadow catcher settings depends on object visibility settings. */
  if (flag & (OBJECT_ADDED | OBJECT_REMOVED | OBJECT_MODIFIED)) {
    scene->integrator->tag_update(scene, Integrator::OBJECT_MANAGER);
  }
}

bool ObjectManager::need_update() const
{
  return update_flags != UPDATE_NONE;
}

string ObjectManager::get_cryptomatte_objects(Scene *scene)
{
  string manifest = "{";

  unordered_set<ustring> objects;
  for (Object *object : scene->objects) {
    if (objects.count(object->name)) {
      continue;
    }
    objects.insert(object->name);
    const uint32_t hash_name = util_murmur_hash3(object->name.c_str(), object->name.length(), 0);
    manifest += string_printf("\"%s\":\"%08x\",", object->name.c_str(), hash_name);
  }
  manifest[manifest.size() - 1] = '}';
  return manifest;
}

string ObjectManager::get_cryptomatte_assets(Scene *scene)
{
  string manifest = "{";
  unordered_set<ustring> assets;
  for (Object *ob : scene->objects) {
    if (assets.count(ob->asset_name)) {
      continue;
    }
    assets.insert(ob->asset_name);
    const uint32_t hash_asset = util_murmur_hash3(
        ob->asset_name.c_str(), ob->asset_name.length(), 0);
    manifest += string_printf("\"%s\":\"%08x\",", ob->asset_name.c_str(), hash_asset);
  }
  manifest[manifest.size() - 1] = '}';
  return manifest;
}

CCL_NAMESPACE_END
