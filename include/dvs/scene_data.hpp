#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "dvs/scene_types.hpp"

namespace dvs {

bool load_scene_bundle(
    const std::filesystem::path& bundle_dir,
    SceneBundle* out_bundle,
    std::string* error_message);

std::vector<Triangle> build_triangles_from_mesh(const MeshData& mesh);

void build_bvh(
    const std::vector<Triangle>& triangles,
    std::vector<BVHNode>* out_nodes,
    std::vector<Triangle>* out_ordered_triangles,
    int leaf_size = 8);

}  // namespace dvs
