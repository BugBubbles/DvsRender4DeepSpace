#include "dvs/hapke.hpp"
#include "dvs/raycast.hpp"
#include <fstream>
#include <iostream>

namespace dvs {

GpuRaycaster::GpuRaycaster(const std::vector<BVHNode<Vec3>> &nodes,
                           const std::vector<Triangle<Vec3>> &triangles,
                           const int root_index)
    : root_index_(root_index) {
  node_count_ = static_cast<int>(nodes.size());
  tri_count_ = static_cast<int>(triangles.size());
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
void GpuRaycaster::intersect(const SceneBundle<Vec3, Uint3, Mat3> &scene,
                             int frame_idx,
                             std::vector<Intersection<Vec3>> &hits) {
  int width = scene.width;
  int height = scene.height;
  float fx = scene.fx;
  float fy = scene.fy;
  float cx = scene.cx;
  float cy = scene.cy;
  auto err = cudaGetLastError();

  auto &frame = scene.frames[frame_idx];
  float3 origin = make_float3(frame.camera_origin.x(), frame.camera_origin.y(),
                              frame.camera_origin.z());
  float *d_camera_world;
  CUDA_CHECK(cudaMalloc(&d_camera_world, sizeof(float) * 9));
  CUDA_CHECK(cudaMemcpy(d_camera_world, frame.camera_world.data(),
                        sizeof(float) * 9, cudaMemcpyHostToDevice));

  ensure_ray_capacity(static_cast<std::size_t>(width * height));
  // 生成视线
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  view_direction_batch_kernel<<<grid_size, block_size>>>(
      width, height, fx, fy, cx, cy, d_camera_world, origin, d_rays_);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Kernel error = " << cudaGetErrorString(err) << "\n";
  // 进行批量求交
  // intersect_brute_batch_kernel<<<grid_size, block_size>>>(
  //     width, height, tri_count_, node_count_, d_nodes_, d_tris_, d_rays_,
  //     d_hits_);
  const int ray_count = width * height;
  const int threads = 128; // 或 256
  const int blocks = (ray_count + threads - 1) / threads;
  batch_intersect_kernel<<<blocks, threads>>>(d_nodes_, root_index_, d_tris_,
                                              d_rays_, width * height, d_hits_);
  // 测试单测，只计算256，256这个像素的交点，验证intersect_triangle的正确性
  // Ray<float3> test_ray;
  // test_ray.origin = make_float3(21.7763, -32185.2109, 7938.9810);
  // test_ray.direction = make_float3(-0.0104, 0.1778, -0.9840);

  // BVHNode<float3> *test_nodes = new BVHNode<float3>[node_count_];
  // Triangle<float3> *test_tris = new Triangle<float3>[tri_count_];
  // CUDA_CHECK(
  //     cudaMemcpy(test_nodes, d_nodes_,
  //                sizeof(BVHNode<float3>) * static_cast<size_t>(node_count_),
  //                cudaMemcpyDeviceToHost));
  // CUDA_CHECK(
  //     cudaMemcpy(test_tris, d_tris_,
  //                sizeof(Triangle<float3>) * static_cast<size_t>(tri_count_),
  //                cudaMemcpyDeviceToHost));

  // Intersection<float3> hit = intersect_single_ray_cpu(
  //     test_nodes, root_index_, test_tris, test_ray, node_count_);
  // std::cout << "CPU single ray test: hit=" << hit.hit << " t=" << hit.t
  //           << " tri_idx=" << hit.triangle_index << " point=(" << hit.point.x
  //           << "," << hit.point.y << "," << hit.point.z << ") normal=("
  //           << hit.normal.x << "," << hit.normal.y << "," << hit.normal.z
  //           << ")\n";
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Kernel error = " << cudaGetErrorString(err) << "\n";
  // 检查求交结果
  hits.resize(static_cast<std::size_t>(width * height));
  CUDA_CHECK(cudaMemcpy(hits.data(), d_hits_,
                        sizeof(Intersection<float3>) * width * height,
                        cudaMemcpyDeviceToHost));
  cudaFree(d_camera_world);
  // 将所有的光线、交点等信息打印出来，验证是否正确
  std::ofstream ofs("debug_output.txt");
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = y * width + x;
      ofs << "Pixel (" << x << "," << y << "): "
          << "hit=" << hits[idx].hit << " t=" << hits[idx].t
          << " tri_idx=" << hits[idx].triangle_index << " point=("
          << hits[idx].point.x() << "," << hits[idx].point.y() << ","
          << hits[idx].point.z() << ")"
          << " normal=(" << hits[idx].normal.x() << "," << hits[idx].normal.y()
          << "," << hits[idx].normal.z() << ")" << std::endl;

      ofs.flush();
    }
  }
}
} // namespace dvs