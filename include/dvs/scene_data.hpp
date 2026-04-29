#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "dvs/scene_types.hpp"

namespace dvs {

bool load_scene_bundle(
    const std::filesystem::path& bundle_dir,
    SceneBundle<Vec3, Uint3, Mat3>* out_bundle,
    std::string* error_message);

std::vector<Triangle<Vec3>> build_triangles_from_mesh(const MeshData<Vec3, Uint3>& mesh);

void build_bvh(
    const std::vector<Triangle<Vec3>>& triangles,
    std::vector<BVHNode<Vec3>>* out_nodes,
    std::vector<Triangle<Vec3>>* out_ordered_triangles,
    int leaf_size = 8);

}  // namespace dvs
