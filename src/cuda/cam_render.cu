#include "bvh/build.h"
#include "dvs/hapke.hpp"
#include "dvs/raycast.hpp"
#include "dvs/scene_types.hpp"
#include <ctime>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

namespace dvs {

__global__ void view_direction_batch_kernel(
    const int width, const int height, const int delta_x, const int delta_y,
    const int supersample_factor, const float fx, const float fy,
    const float cx, const float cy, const float *camera_world,
    const float3 origin, float3 *out_rays, curandState *rand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  int idx = (y * width + x);

  // 计算超采样偏移
  // curandState local_state = [idx];
  float offset_x =
      (delta_x + curand_uniform(rand_states + idx)) / supersample_factor;
  float offset_y =
      (delta_y + curand_uniform(rand_states + idx)) / supersample_factor;
  // rand_states[idx] = local_state;

  // 计算像素中心的视线方向
  float px = (x + offset_x - cx - 0.5) / fx;
  float py = (y + offset_y - cy - 0.5) / fy;
  float3 dir_camera = make_float3(px, py, -1.0f); // 摄像机空间的视线方向
  // 转换到世界空间
  float3 dir_world = make_float3(
      camera_world[0] * dir_camera.x + camera_world[3] * dir_camera.y +
          camera_world[6] * dir_camera.z,
      camera_world[1] * dir_camera.x + camera_world[4] * dir_camera.y +
          camera_world[7] * dir_camera.z,
      camera_world[2] * dir_camera.x + camera_world[5] * dir_camera.y +
          camera_world[8] * dir_camera.z);

  dir_world = normalize(dir_world);
  out_rays[idx] = dir_world;
}
/**
 * @brief 将当前超采样结果累加到总的光强缓冲中
 * @param image 当前超采样的光强结果
 * @param output 累加的光强结果
 */
__global__ void batch_image_sum_kernel(const int ray_count, const float *image,
                                       float *output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ray_count) {
    return;
  }
  output[idx] += image[idx];
}
/**
 * @brief 计算平均光强
 * @param image 累加的光强结果
 * @param count 超采样的总次数（即supersample_factor_的平方
 */
__global__ void batch_image_mean_kernel(const int ray_count, const int count,
                                        float *image) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ray_count) {
    return;
  }
  image[idx] /= count;
}
__global__ void batch_blenderize_kernel(float *image, const int ray_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ray_count) {
    return;
  }
  // 简单的blenderize处理：对光强进行gamma校正
  float intensity = image[idx];
  float gamma = 2.2f; // 常用的gamma值
  image[idx] = powf(intensity, 1.0f / gamma);
}

CamRayCaster::CamRayCaster(const SceneBundle<float3> &scene,
                           const Config &config, const int root_index)
    : root_index_(root_index), hapke_config_(config.hapke),
      fx_(config.camera.fx), fy_(config.camera.fy), cx_(config.camera.cx),
      cy_(config.camera.cy), width_(config.camera.width),
      height_(config.camera.height), scene_config_(config.scene),
      supersample_factor_(config.scene.supersample_factor) {
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
  BVHBuilder::validate(nodes, root_index_);
  std::cout << "BVH validated: " << node_count_ << " nodes, " << tri_count_
            << " triangles." << std::endl;
  CUDA_CHECK(cudaMalloc(&d_nodes_, sizeof(BVHNode<float3>) *
                                       static_cast<size_t>(node_count_)),
             "Failed to allocate device memory for BVH nodes");
  CUDA_CHECK(cudaMalloc(&d_tris_, sizeof(Triangle<float3>) *
                                      static_cast<size_t>(tri_count_)),
             "Failed to allocate device memory for triangles");
  CUDA_CHECK(
      cudaMemcpy(d_nodes_, nodes.data(),
                 sizeof(BVHNode<float3>) * static_cast<size_t>(node_count_),
                 cudaMemcpyHostToDevice),
      "Failed to copy BVH data from host to device");
  CUDA_CHECK(
      cudaMemcpy(d_tris_, triangles.data(),
                 sizeof(Triangle<float3>) * static_cast<size_t>(tri_count_),
                 cudaMemcpyHostToDevice),
      "Failed to copy BVH data from host to device");

  // 预分配足够的内存用于存储每一帧的光线、交点结果和光强度
  CUDA_CHECK(cudaMalloc(&d_rays_, sizeof(float3) * ray_count_),
             "Failed to allocate device memory for rays");
  CUDA_CHECK(cudaMalloc(&d_hits_, sizeof(Intersection<float3>) * ray_count_),
             "Failed to allocate device memory for intersection results");
  CUDA_CHECK(cudaMalloc(&d_lights_, sizeof(float) * ray_count_),
             "Failed to allocate device memory for light intensity buffer");
  CUDA_CHECK(
      cudaMalloc(&d_lights_step_, sizeof(float) * ray_count_),
      "Failed to allocate device memory for light intensity step buffer");
  CUDA_CHECK(cudaMalloc(&d_depth_map_, sizeof(float) * ray_count_),
             "Failed to allocate device memory for depth map");
  CUDA_CHECK(cudaMalloc(&d_depth_map_step_, sizeof(float) * ray_count_),
             "Failed to allocate device memory for depth map step buffer");
  CUDA_CHECK(cudaMalloc(&d_rand_states_, sizeof(curandState) * ray_count_),
             "Failed to allocate device memory for random states");
  // 初始化随机状态
  const int threads = 128;
  const int blocks = (ray_count_ + threads - 1) / threads;
  unsigned seed = static_cast<unsigned>(std::time(nullptr));
  init_rand_states<<<blocks, threads>>>(d_rand_states_, ray_count_, seed);
}

void CamRayCaster::render_main(const float3 &camera_origin,
                               const float *camera_world,
                               const float3 &light_dir,
                               const float light_intensity,
                               std::vector<float> &light_frame,
                               std::vector<float> &depth_map) {
#ifdef DEBUG
  auto err = cudaGetLastError();
#endif
  float *d_camera_world;
  CUDA_CHECK(cudaMalloc(&d_camera_world, sizeof(float) * 9),
             "Failed to allocate device memory for camera world matrix");
  CUDA_CHECK(cudaMemcpy(d_camera_world, camera_world, sizeof(float) * 9,
                        cudaMemcpyHostToDevice),
             "Failed to copy camera world matrix from host to device");
  CUDA_CHECK(cudaMemset(d_lights_, 0, sizeof(float) * ray_count_),
             "Failed to clear light intensity buffer on device");
  // 生成视线
  const dim3 block_size(16, 16);
  const dim3 grid_size((width_ + block_size.x - 1) / block_size.x,
                       (height_ + block_size.y - 1) / block_size.y);
  const int threads = 128; // 或 256
  const int blocks = (ray_count_ + threads - 1) / threads;

  // 执行supersample_factor_倍的超采样，每个像素生成supersample_factor_^2条视线
  for (int sx = 0; sx < supersample_factor_; ++sx) {
    for (int sy = 0; sy < supersample_factor_; ++sy) {
      // 生成超采样的视线
      view_direction_batch_kernel<<<grid_size, block_size>>>(
          width_, height_, sx, sy, supersample_factor_, fx_, fy_, cx_, cy_,
          d_camera_world, camera_origin, d_rays_, d_rand_states_);
#ifdef DEBUG
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      std::cout << "Error in [Ray Casting]: " << cudaGetErrorString(err)
                << "\n";
#endif
      batch_intersect_kernel<<<blocks, threads>>>(
          d_nodes_, root_index_, d_tris_, d_rays_, camera_origin, ray_count_,
          d_hits_, d_depth_map_step_);
#ifdef DEBUG
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      std::cout << "Error in [Intersecting]: " << cudaGetErrorString(err)
                << "\n";
#endif
      // 计算光照，使用Hapke模型
      batch_render_kernel<<<blocks, threads>>>(
          d_hits_, d_tris_, d_rays_, camera_origin, light_dir, light_intensity,
          hapke_config_, ray_count_, d_lights_step_);
#ifdef DEBUG
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      std::cout << "Error in [Rendering]: " << cudaGetErrorString(err) << "\n";
#endif
      // 将当前超采样结果累加到总的光强缓冲中
      batch_image_sum_kernel<<<blocks, threads>>>(ray_count_, d_lights_step_,
                                                  d_lights_);
      batch_image_sum_kernel<<<blocks, threads>>>(ray_count_, d_depth_map_step_,
                                                  d_depth_map_);
    }
  }
  // 计算平均光强
  batch_image_mean_kernel<<<blocks, threads>>>(
      ray_count_, supersample_factor_ * supersample_factor_, d_lights_);
  batch_image_mean_kernel<<<blocks, threads>>>(
      ray_count_, supersample_factor_ * supersample_factor_, d_depth_map_);
  // 将结果从GPU复制回CPU
  light_frame.resize(static_cast<std::size_t>(ray_count_));
  depth_map.resize(static_cast<std::size_t>(ray_count_));
  CUDA_CHECK(cudaMemcpy(light_frame.data(), d_lights_,
                        sizeof(float) * ray_count_, cudaMemcpyDeviceToHost),
             "Failed to copy light frame from device to host");
  CUDA_CHECK(cudaMemcpy(depth_map.data(), d_depth_map_,
                        sizeof(float) * ray_count_, cudaMemcpyDeviceToHost),
             "Failed to copy depth map from device to host");
#ifdef DEBUG
  // 输出调试信息
  std::ofstream ofs("debug_output.txt");
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int idx = y * width_ + x;
      ofs << "Pixel (" << x << "," << y << "): light=" << light_frame[idx]
          << std::endl;
      ofs.flush();
    }
  }
  ofs.close();
#endif
  cudaFree(d_camera_world);
}

void CamBlurRayCaster::render_main(const float3 &camera_origin,
                                   const float *camera_world,
                                   const float3 &light_dir,
                                   const float light_intensity,
                                   std::vector<float> &light_frame) {
#ifdef DEBUG
  auto err = cudaGetLastError();
#endif
  float *d_camera_world;
  CUDA_CHECK(cudaMalloc(&d_camera_world, sizeof(float) * 9),
             "Failed to allocate device memory for camera world matrix");
  CUDA_CHECK(cudaMemcpy(d_camera_world, camera_world, sizeof(float) * 9,
                        cudaMemcpyHostToDevice),
             "Failed to copy camera world matrix from host to device");
  CUDA_CHECK(cudaMemset(d_lights_, 0, sizeof(float) * ray_count_),
             "Failed to clear light intensity buffer on device");
  // 生成视线
  const dim3 block_size(16, 16);
  const dim3 grid_size((width_ + block_size.x - 1) / block_size.x,
                       (height_ + block_size.y - 1) / block_size.y);
  const int threads = 128; // 或 256
  const int blocks = (ray_count_ + threads - 1) / threads;

  // 执行supersample_factor_倍的超采样，每个像素生成supersample_factor_^2条视线
  for (int sx = 0; sx < supersample_factor_; ++sx) {
    for (int sy = 0; sy < supersample_factor_; ++sy) {
      // 生成超采样的视线
      view_direction_batch_kernel<<<grid_size, block_size>>>(
          width_, height_, sx, sy, supersample_factor_, fx_, fy_, cx_, cy_,
          d_camera_world, camera_origin, d_rays_, d_rand_states_);
#ifdef DEBUG
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      std::cout << "Error in [Ray Casting]: " << cudaGetErrorString(err)
                << "\n";
#endif
      batch_intersect_kernel<<<blocks, threads>>>(
          d_nodes_, root_index_, d_tris_, d_rays_, camera_origin, ray_count_,
          d_hits_, d_depth_map_step_);
#ifdef DEBUG
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      std::cout << "Error in [Intersecting]: " << cudaGetErrorString(err)
                << "\n";
#endif
      // 计算光照，使用Hapke模型
      batch_render_kernel<<<blocks, threads>>>(
          d_hits_, d_tris_, d_rays_, camera_origin, light_dir, light_intensity,
          hapke_config_, ray_count_, d_lights_step_);
#ifdef DEBUG
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      std::cout << "Error in [Rendering]: " << cudaGetErrorString(err) << "\n";
#endif
      // 将当前超采样结果累加到总的光强缓冲中
      batch_image_sum_kernel<<<blocks, threads>>>(ray_count_, d_lights_step_,
                                                  d_lights_);
      batch_image_sum_kernel<<<blocks, threads>>>(ray_count_, d_depth_map_step_,
                                                  d_depth_map_);
    }
  }
  // 计算平均光强
  batch_image_mean_kernel<<<blocks, threads>>>(
      ray_count_, supersample_factor_ * supersample_factor_, d_lights_);
  // 将结果从GPU复制回CPU
  light_frame.resize(static_cast<std::size_t>(ray_count_));
  CUDA_CHECK(cudaMemcpy(light_frame.data(), d_lights_,
                        sizeof(float) * ray_count_, cudaMemcpyDeviceToHost),
             "Failed to copy light frame from device to host");
#ifdef DEBUG
  // 输出调试信息
  std::ofstream ofs("debug_output.txt");
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int idx = y * width_ + x;
      ofs << "Pixel (" << x << "," << y << "): light=" << light_frame[idx]
          << std::endl;
      ofs.flush();
    }
  }
  ofs.close();
#endif
  cudaFree(d_camera_world);
}
} // namespace dvs