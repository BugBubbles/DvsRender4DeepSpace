#include "dvs/scene_data.hpp"
#include "dvs/scene_types.hpp"
#include "helper_math.hpp"
#include "tools/build_from_raster.hpp"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector_types.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dvs {

constexpr int kFrameStrideFloats = 24;

std::string trim(const std::string &s) {
  std::size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
    ++b;
  }
  std::size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
    --e;
  }
  return s.substr(b, e - b);
}

bool read_binary_f32(const std::filesystem::path &path, std::vector<float> *out,
                     std::string *err) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    if (err != nullptr) {
      *err = "failed to open " + path.string();
    }
    return false;
  }
  fin.seekg(0, std::ios::end);
  const std::streamoff size = fin.tellg();
  fin.seekg(0, std::ios::beg);
  if (size < 0 || size % static_cast<std::streamoff>(sizeof(float)) != 0) {
    if (err != nullptr) {
      *err = "invalid f32 binary size for " + path.string();
    }
    return false;
  }
  out->resize(static_cast<std::size_t>(
      size / static_cast<std::streamoff>(sizeof(float))));
  fin.read(reinterpret_cast<char *>(out->data()), size);
  if (!fin) {
    if (err != nullptr) {
      *err = "failed to read " + path.string();
    }
    return false;
  }
  return true;
}

bool read_binary_u32(const std::filesystem::path &path,
                     std::vector<std::uint32_t> *out, std::string *err) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    if (err != nullptr) {
      *err = "failed to open " + path.string();
    }
    return false;
  }
  fin.seekg(0, std::ios::end);
  const std::streamoff size = fin.tellg();
  fin.seekg(0, std::ios::beg);
  if (size < 0 ||
      size % static_cast<std::streamoff>(sizeof(std::uint32_t)) != 0) {
    if (err != nullptr) {
      *err = "invalid u32 binary size for " + path.string();
    }
    return false;
  }
  out->resize(static_cast<std::size_t>(
      size / static_cast<std::streamoff>(sizeof(std::uint32_t))));
  fin.read(reinterpret_cast<char *>(out->data()), size);
  if (!fin) {
    if (err != nullptr) {
      *err = "failed to read " + path.string();
    }
    return false;
  }
  return true;
}

bool load_mesh_data(const std::filesystem::path &dir,
                    MeshData<float3, uint3> &out, std::string *err) {
  std::vector<float> vertices_raw;
  std::vector<float> normals_raw;
  std::vector<std::uint32_t> indices_raw;

  if (!read_binary_f32(dir / "mesh_vertices.bin", &vertices_raw, err)) {
    return false;
  }
  if (!read_binary_f32(dir / "mesh_normals.bin", &normals_raw, err)) {
    return false;
  }
  if (!read_binary_u32(dir / "mesh_indices.bin", &indices_raw, err)) {
    return false;
  }

  if (vertices_raw.size() % 3 != 0 || normals_raw.size() % 3 != 0 ||
      indices_raw.size() % 3 != 0) {
    if (err != nullptr) {
      *err = "mesh binaries must be Nx3";
    }
    return false;
  }

  out.vertices.resize(vertices_raw.size() / 3);
  out.normals.resize(normals_raw.size() / 3);
  out.indices.resize(indices_raw.size() / 3);

#pragma omp parallel for
  for (std::ptrdiff_t i = 0;
       i < static_cast<std::ptrdiff_t>(out.vertices.size()); ++i) {
    out.vertices[i] = {
        vertices_raw[static_cast<std::size_t>(i) * 3 + 0],
        vertices_raw[static_cast<std::size_t>(i) * 3 + 1],
        vertices_raw[static_cast<std::size_t>(i) * 3 + 2],
    };
  }
#pragma omp parallel for
  for (std::ptrdiff_t i = 0;
       i < static_cast<std::ptrdiff_t>(out.normals.size()); ++i) {
    out.normals[i] = {
        normals_raw[static_cast<std::size_t>(i) * 3 + 0],
        normals_raw[static_cast<std::size_t>(i) * 3 + 1],
        normals_raw[static_cast<std::size_t>(i) * 3 + 2],
    };
  }
#pragma omp parallel for
  for (std::ptrdiff_t i = 0;
       i < static_cast<std::ptrdiff_t>(out.indices.size()); ++i) {
    out.indices[i] = {
        indices_raw[static_cast<std::size_t>(i) * 3 + 0],
        indices_raw[static_cast<std::size_t>(i) * 3 + 1],
        indices_raw[static_cast<std::size_t>(i) * 3 + 2],
    };
  }

  return true;
}

bool load_frames_from_binary(const std::filesystem::path &dir,
                             std::vector<FrameData<Vec3, Mat3>> &out,
                             std::string *err) {
  std::vector<float> raw;
  if (!read_binary_f32(dir / "frames.bin", &raw, err)) {
    return false;
  }
  if (raw.size() % kFrameStrideFloats != 0) {
    if (err != nullptr) {
      *err = "frames.bin has invalid stride";
    }
    return false;
  }

  const std::size_t frame_count = raw.size() / kFrameStrideFloats;
  out.resize(frame_count);
  Mat4 camera_world;
#pragma omp parallel for
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(frame_count);
       ++i) {
    const std::size_t o = i * kFrameStrideFloats;
    FrameData<Vec3, Mat3> frame;
    frame.timestamp = raw[o + 0];
    frame.camera_origin = Vec3(raw[o + 1], raw[o + 2], raw[o + 3]);
    for (int j = 0; j < 16; ++j) {
      camera_world(j / 4, j % 4) = raw[o + 4 + j];
    }
    frame.camera_world = camera_world.block<3, 3>(0, 0);
    frame.light_dir = Vec3(raw[o + 20], raw[o + 21], raw[o + 22]);
    frame.light_intensity = raw[o + 23];
    out[i] = frame;
  }
  return true;
}

void build_triangles_from_mesh(const MeshData<float3, uint3> &mesh,
                               std::vector<Triangle<float3>> &out) {
  std::vector<Triangle<float3>> tmp(mesh.indices.size());
  std::vector<std::uint8_t> valid(mesh.indices.size(),
                                  static_cast<std::uint8_t>(0));

#pragma omp parallel for
  for (std::ptrdiff_t i = 0;
       i < static_cast<std::ptrdiff_t>(mesh.indices.size()); ++i) {
    const uint3 tri = mesh.indices[static_cast<std::size_t>(i)];
    if (tri.x >= mesh.vertices.size() || tri.y >= mesh.vertices.size() ||
        tri.z >= mesh.vertices.size()) {
      continue;
    }
    const float3 v0 = mesh.vertices[static_cast<std::size_t>(tri.x)];
    const float3 v1 = mesh.vertices[static_cast<std::size_t>(tri.y)];
    const float3 v2 = mesh.vertices[static_cast<std::size_t>(tri.z)];
    const float3 face_n = normalize(cross(v1 - v0, v2 - v0));

    const float3 n0 = (tri.x < mesh.normals.size())
                          ? mesh.normals[static_cast<std::size_t>(tri.x)]
                          : face_n;
    const float3 n1 = (tri.y < mesh.normals.size())
                          ? mesh.normals[static_cast<std::size_t>(tri.y)]
                          : face_n;
    const float3 n2 = (tri.z < mesh.normals.size())
                          ? mesh.normals[static_cast<std::size_t>(tri.z)]
                          : face_n;

    tmp[static_cast<std::size_t>(i)] = Triangle<float3>{v0, v1, v2, n0, n1, n2};
    valid[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(1);
  }

  out.reserve(mesh.indices.size());
  for (std::size_t i = 0; i < tmp.size(); ++i) {
    if (valid[i] == 0) {
      continue;
    }
    out.push_back(tmp[i]);
  }
}

bool load_scene_bundle(const std::filesystem::path &bundle_dir,
                       SceneBundle<float3> &out_bundle,
                       std::vector<FrameData<Vec3, Mat3>> &frames,
                       std::string *error_message) {
  MeshData<float3, uint3> mesh;
  if (!load_mesh_data(bundle_dir, mesh, error_message)) {
    return false;
  }

  std::cout << "Loaded mesh with " << mesh.vertices.size() << " vertices, "
            << mesh.normals.size() << " normals, and " << mesh.indices.size()
            << " triangles" << std::endl;

  if (!load_frames_from_binary(bundle_dir, frames, error_message)) {
    return false;
  }
  if (frames.empty()) {
    if (error_message != nullptr) {
      *error_message = "frames.bin is empty";
    }
    return false;
  }

  build_triangles_from_mesh(mesh, out_bundle.bvh_triangles);
  out_bundle.bvh_nodes.clear();

  if (out_bundle.bvh_triangles.empty()) {
    if (error_message != nullptr) {
      *error_message = "mesh is empty after triangulation";
    }
    return false;
  }

  return true;
}

bool load_frames_from_text(const std::filesystem::path &path,
                           std::vector<FrameData<Vec3, Mat3>> &out,
                           std::string *err) {
  std::vector<float> raw;
  if (!std::filesystem::exists(path)) {
    if (err != nullptr) {
      *err = "file does not exist: " + path.string();
    }
    return false;
  }
  std::ifstream fin(path);
  if (!fin) {
    if (err != nullptr) {
      *err = "failed to open " + path.string();
    }
    return false;
  }
  Mat4 camera_world;
  float qx, qy, qz, qw;

  // 读取文件，存储格式是：frame_idx x y z qx qy qz qw light_dir_x light_dir_y
  // light_dir_z light_intensity
  std::string line;
  while (std::getline(fin, line)) {
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    std::istringstream iss(line);
    FrameData<Vec3, Mat3> frame;
    if (!(iss >> frame.timestamp >> frame.camera_origin.x() >>
          frame.camera_origin.y() >> frame.camera_origin.z() >> qx >> qy >>
          qz >> qw >> frame.light_dir.x() >> frame.light_dir.y() >>
          frame.light_dir.z() >> frame.light_intensity)) {
      if (err != nullptr) {
        *err = "failed to parse camera orientation in line: " + line;
      }
      return false;
    }
    // 将四元数转换为旋转矩阵，使用Eigen的API
    Eigen::Quaternionf q(qw, qx, qy, qz);
    frame.camera_world = q.toRotationMatrix();
    out.push_back(frame);
  }
  return true;
}
bool load_scene_bundle(const std::filesystem::path &dem_path,
                       const std::filesystem::path &albedo_path,
                       SceneBundle<float3> &out_bundle,
                       std::string *error_message) {
  std::vector<float> albedos;
  if (!tools::load_from_raster(dem_path.string(), albedo_path.string(),
                               out_bundle.bvh_triangles, albedos)) {
    return false;
  }

  out_bundle.bvh_nodes.clear();

  if (out_bundle.bvh_triangles.empty()) {
    if (error_message != nullptr) {
      *error_message = "mesh is empty after triangulation";
    }
    return false;
  }

  return true;
}

bool load_scene_bundle(const std::filesystem::path &dem_path,
                       const std::filesystem::path &albedo_path,
                       const std::filesystem::path &traj_path,
                       SceneBundle<float3> &out_bundle,
                       std::vector<FrameData<Vec3, Mat3>> &frames,
                       std::string *error_message) {
  std::vector<Triangle<float3>> triangles;
  std::vector<float> albedos;
  if (!tools::load_from_raster(dem_path.string(), albedo_path.string(),
                               triangles, albedos)) {
    return false;
  }

  if (!load_frames_from_text(traj_path, frames, error_message)) {
    return false;
  }
  if (frames.empty()) {
    if (error_message != nullptr) {
      *error_message = "frames.bin is empty";
    }
    return false;
  }
  out_bundle.bvh_triangles = std::move(triangles);
  out_bundle.bvh_nodes.clear();

  if (out_bundle.bvh_triangles.empty()) {
    if (error_message != nullptr) {
      *error_message = "mesh is empty after triangulation";
    }
    return false;
  }

  return true;
}
} // namespace dvs
