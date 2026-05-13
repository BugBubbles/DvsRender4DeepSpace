#include "dvs/hapke.hpp"
#include "dvs/raycast.hpp"
#include <fstream>
#include <iostream>

namespace dvs {

GpuRaycaster::GpuRaycaster(const SceneBundle<Vec3, Uint3, Mat3> &scene,
                           const int root_index,
                           const HapkeParams &hapke_params)
    : root_index_(root_index), hapke_params_(hapke_params), fx_(scene.fx),
      fy_(scene.fy), cx_(scene.cx), cy_(scene.cy), width_(scene.width),
      height_(scene.height) {
  const auto &nodes = scene.bvh_nodes;
  const auto &triangles = scene.bvh_triangles;

  node_count_ = static_cast<int>(nodes.size());
  tri_count_ = static_cast<int>(triangles.size());
  ray_count_ = width_ * height_;
  if (node_count_ <= 0 || tri_count_ <= 0) {
    throw std::invalid_argument(
        "prebuilt BVH requires non-empty nodes and triangles");
  }

  std::cout << "BVH root_index_ = " << root_index_
            << ", root count = " << nodes[root_index_].count
            << ", left = " << nodes[root_index_].left
            << ", right = " << nodes[root_index_].right << std::endl;
  validate_bvh(nodes, root_index_);
  std::cout << "BVH validated: " << node_count_ << " nodes, " << tri_count_
            << " triangles." << std::endl;
  // 将Vec3格式的nodes转为float3格式的nodes
  std::vector<BVHNode<float3>> float_nodes(node_count_);
  std::vector<Triangle<float3>> float_tris(tri_count_);
  for (int i = 0; i < tri_count_; ++i) {
    const auto &src_tri = triangles[static_cast<size_t>(i)];
    auto &dst_tri = float_tris[static_cast<size_t>(i)];
    dst_tri.v0 = make_float3(src_tri.v0.x(), src_tri.v0.y(), src_tri.v0.z());
    dst_tri.v1 = make_float3(src_tri.v1.x(), src_tri.v1.y(), src_tri.v1.z());
    dst_tri.v2 = make_float3(src_tri.v2.x(), src_tri.v2.y(), src_tri.v2.z());
    dst_tri.n0 = make_float3(src_tri.n0.x(), src_tri.n0.y(), src_tri.n0.z());
    dst_tri.n1 = make_float3(src_tri.n1.x(), src_tri.n1.y(), src_tri.n1.z());
    dst_tri.n2 = make_float3(src_tri.n2.x(), src_tri.n2.y(), src_tri.n2.z());
  }
  for (int i = 0; i < node_count_; ++i) {
    const auto &src_node = nodes[static_cast<size_t>(i)];
    auto &dst_node = float_nodes[static_cast<size_t>(i)];
    dst_node.box.min = make_float3(src_node.box.min.x(), src_node.box.min.y(),
                                   src_node.box.min.z());
    dst_node.box.max = make_float3(src_node.box.max.x(), src_node.box.max.y(),
                                   src_node.box.max.z());
    dst_node.left = src_node.left;
    dst_node.right = src_node.right;
    dst_node.start = src_node.start;
    dst_node.count = src_node.count;
  }
  CUDA_CHECK(cudaMalloc(&d_nodes_, sizeof(BVHNode<float3>) *
                                       static_cast<size_t>(node_count_)));
  CUDA_CHECK(cudaMalloc(&d_tris_, sizeof(Triangle<float3>) *
                                      static_cast<size_t>(tri_count_)));
  CUDA_CHECK(
      cudaMemcpy(d_nodes_, float_nodes.data(),
                 sizeof(BVHNode<float3>) * static_cast<size_t>(node_count_),
                 cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_tris_, float_tris.data(),
                 sizeof(Triangle<float3>) * static_cast<size_t>(tri_count_),
                 cudaMemcpyHostToDevice));

  // 预分配足够的内存用于存储每一帧的光线、交点结果和光强度
  CUDA_CHECK(cudaMalloc(&d_rays_, sizeof(float3) * ray_count_));
  CUDA_CHECK(cudaMalloc(&d_hits_, sizeof(Intersection<float3>) * ray_count_));
  CUDA_CHECK(cudaMalloc(&d_lights_, sizeof(float) * ray_count_));
}
void GpuRaycaster::validate_bvh(const std::vector<BVHNode<Vec3>> &nodes,
                                int root) const {
  if (root < 0 || root >= static_cast<int>(nodes.size())) {
    throw std::runtime_error("BVH root index out of range");
  }

  std::vector<bool> visited(nodes.size(), false);
  std::vector<int> stack;
  stack.push_back(root);

  int visited_count = 0;
  int leaf_count = 0;
  int internal_count = 0;

  while (!stack.empty()) {
    int idx = stack.back();
    stack.pop_back();

    if (idx < 0 || idx >= static_cast<int>(nodes.size())) {
      throw std::runtime_error("BVH invalid node index");
    }

    if (visited[idx]) {
      throw std::runtime_error("BVH cycle detected");
    }

    visited[idx] = true;
    visited_count++;

    const auto &node = nodes[idx];

    if (node.count > 0) {
      leaf_count++;

      if (node.left != -1 || node.right != -1) {
        throw std::runtime_error("BVH leaf has non-negative children");
      }

      continue;
    }

    internal_count++;

    if (node.left < 0 || node.left >= static_cast<int>(nodes.size()) ||
        node.right < 0 || node.right >= static_cast<int>(nodes.size())) {
      throw std::runtime_error("BVH child index out of range");
    }

    stack.push_back(node.right);
    stack.push_back(node.left);
  }

  std::cout << "BVH validation visited nodes: " << visited_count
            << ", leaves: " << leaf_count << ", internals: " << internal_count
            << std::endl;
}
void GpuRaycaster::render_main(const FrameData<Vec3, Mat3> &frame,
                               std::vector<float> &light_frame) {
  auto err = cudaGetLastError();
  float3 ray_origin =
      make_float3(frame.camera_origin.x(), frame.camera_origin.y(),
                  frame.camera_origin.z());
  float *d_camera_world;
  CUDA_CHECK(cudaMalloc(&d_camera_world, sizeof(float) * 9));
  CUDA_CHECK(cudaMemcpy(d_camera_world, frame.camera_world.data(),
                        sizeof(float) * 9, cudaMemcpyHostToDevice));
  // 生成视线
  dim3 block_size(16, 16);
  dim3 grid_size((width_ + block_size.x - 1) / block_size.x,
                 (height_ + block_size.y - 1) / block_size.y);
  view_direction_batch_kernel<<<grid_size, block_size>>>(
      width_, height_, fx_, fy_, cx_, cy_, d_camera_world, ray_origin, d_rays_);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Error in [Ray Casting]: " << cudaGetErrorString(err) << "\n";
  // 进行批量求交，暴力枚举法
  // intersect_brute_batch_kernel<<<grid_size, block_size>>>(
  //     width, height, tri_count_, node_count_, d_nodes_, d_tris_, d_rays_,
  //     d_hits_);
  const int threads = 128; // 或 256
  const int blocks = (ray_count_ + threads - 1) / threads;
  batch_intersect_kernel<<<blocks, threads>>>(
      d_nodes_, root_index_, d_tris_, d_rays_, ray_origin, ray_count_, d_hits_);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Error in [Intersecting]: " << cudaGetErrorString(err) << "\n";
  // 检查求交结果
  std::vector<Intersection<float3>> hits(static_cast<std::size_t>(ray_count_));
  CUDA_CHECK(cudaMemcpy(hits.data(), d_hits_,
                        sizeof(Intersection<float3>) * ray_count_,
                        cudaMemcpyDeviceToHost));
  cudaFree(d_camera_world);
  // 计算光照，使用Hapke模型
  float3 light_dir = make_float3(frame.light_dir.x(), frame.light_dir.y(),
                                 frame.light_dir.z());


  float light_intensity = frame.light_intensity;
  batch_render_kernel<<<blocks, threads>>>(
      d_hits_, d_rays_, ray_origin, light_dir, light_intensity, hapke_params_,
      ray_count_, d_lights_);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Error in [Rendering]: " << cudaGetErrorString(err) << "\n";
  // 将结果从GPU复制回CPU
  light_frame.resize(static_cast<std::size_t>(ray_count_));
  CUDA_CHECK(cudaMemcpy(light_frame.data(), d_lights_,
                        sizeof(float) * ray_count_, cudaMemcpyDeviceToHost));
  // 输出调试信息
  // std::ofstream ofs("debug_output.txt");
  // for (int y = 0; y < height_; ++y) {
  //   for (int x = 0; x < width_; ++x) {
  //     int idx = y * width_ + x;
  //     ofs << "Pixel (" << x << "," << y << "): light=" << light_frame[idx]
  //         << std::endl;
  //     ofs.flush();
  //   }
  // }
  // ofs.close();
}
} // namespace dvs