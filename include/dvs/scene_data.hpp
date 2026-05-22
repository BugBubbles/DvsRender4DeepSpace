#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "dvs/math_types.hpp"
#include "dvs/scene_types.hpp"

namespace dvs {

bool load_scene_bundle(const std::filesystem::path &bundle_dir,
                       SceneBundle<Vec3> &out_bundle,
                       std::vector<FrameData<Vec3, Mat3>> &out_frames,
                       std::string *error_message);

template <typename F>
bool extract_required(const std::unordered_map<std::string, std::string> &map,
                      const std::string &section_name, const std::string &key,
                      F &destination, std::string &err,
                      bool (*convert)(const std::string &, F &)) {
  auto it = map.find(key);
  if (it == map.end()) {
    err = "Missing key '" + key + "' in section '" + section_name + "'";
    return false;
  }
  if (!convert(it->second, destination)) {
    err = "Invalid value for key '" + key + "' in section '" + section_name +
          "': '" + it->second + "'";
    return false;
  }
  return true;
};

std::vector<Triangle<Vec3>>
build_triangles_from_mesh(const MeshData<Vec3, Uint3> &mesh);

void build_bvh(const std::vector<Triangle<Vec3>> &triangles,
               std::vector<BVHNode<Vec3>> *out_nodes,
               std::vector<Triangle<Vec3>> *out_ordered_triangles,
               int leaf_size = 8);

} // namespace dvs
