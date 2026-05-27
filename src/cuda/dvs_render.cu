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

__global__ void threshold_map_init_kernel(const int width, const int height,
                                          const float threshold,
                                          const float noise_stddev,
                                          float *threshold_map) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int idx = y * width + x;

  curandState state;
  curand_init(RAND_SEED, idx, 0, &state); // 固定种子确保每次运行一致
  // 使用均匀分布生成噪声
  float noise = curand_uniform(&state) * 2.0f * noise_stddev -
                noise_stddev; // [-noise_stddev, +noise_stddev]

  threshold_map[idx] = threshold * (1.0f + noise); // 添加噪声后的阈值
}

DvsRayCaster::DvsRayCaster(const SceneBundle<float3, uint3> &scene,
                           const Config &config, const int root_index)
    : root_index_(root_index), hapke_config_(config.hapke),
      fx_(config.camera.fx), fy_(config.camera.fy), cx_(config.camera.cx),
      cy_(config.camera.cy), width_(config.camera.width),
      height_(config.camera.height), physic_config_(config.physic),
      scene_config_(config.scene),
      supersample_factor_(config.scene.supersample_factor) {
  const auto &nodes = scene.bvh_nodes;
  const auto &triangles = scene.bvh_triangles;
  const int ss = supersample_factor_ * supersample_factor_;
  use_albedo_ = scene.use_albedo;
  if (use_albedo_) {
    std::cout << "Using albedo in rendering.\n";
  } else {
    std::cout << "Not using albedo in rendering.\n";
  }

  node_count_ = static_cast<int>(nodes.size());
  tri_count_ = static_cast<int>(triangles.size());
  ver_count_ = static_cast<int>(scene.vertices.size());
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
  CUDA_CHECK(cudaMalloc(&d_vertices_,
                        sizeof(float3) * static_cast<size_t>(ver_count_)),
             "Failed to allocate device memory for vertices");
  CUDA_CHECK(
      cudaMalloc(&d_normals_, sizeof(float3) * static_cast<size_t>(ver_count_)),
      "Failed to allocate device memory for normals");
  CUDA_CHECK(
      cudaMalloc(&d_indices_, sizeof(uint3) * static_cast<size_t>(tri_count_)),
      "Failed to allocate device memory for indices");
  if (use_albedo_) {
    CUDA_CHECK(cudaMalloc(&d_albedos_,
                          sizeof(float) * static_cast<size_t>(ver_count_)),
               "Failed to allocate device memory for albedos");
  }
  CUDA_CHECK(
      cudaMemcpy(d_nodes_, nodes.data(),
                 sizeof(BVHNode<float3>) * static_cast<size_t>(node_count_),
                 cudaMemcpyHostToDevice),
      "Failed to copy BVH data from host to device");
  CUDA_CHECK(cudaMemcpy(d_vertices_, scene.vertices.data(),
                        sizeof(float3) * static_cast<size_t>(ver_count_),
                        cudaMemcpyHostToDevice),
             "Failed to copy vertex data from host to device");
  CUDA_CHECK(cudaMemcpy(d_normals_, scene.normals.data(),
                        sizeof(float3) * static_cast<size_t>(ver_count_),
                        cudaMemcpyHostToDevice),
             "Failed to copy normal data from host to device");

  CUDA_CHECK(cudaMemcpy(d_indices_, scene.bvh_triangles.data(),
                        sizeof(uint3) * static_cast<size_t>(tri_count_),
                        cudaMemcpyHostToDevice),
             "Failed to copy index data from host to device");
  if (use_albedo_) {
    CUDA_CHECK(cudaMemcpy(d_albedos_, scene.albedos.data(),
                          sizeof(float) * static_cast<size_t>(ver_count_),
                          cudaMemcpyHostToDevice),
               "Failed to copy albedo data from host to device");
  }
  // 预分配足够的内存用于存储每一帧的光线、交点结果和光强度
  CUDA_CHECK(cudaMalloc(&d_rays_, sizeof(float3) * ray_count_ * ss),
             "Failed to allocate device memory for rays");
  CUDA_CHECK(
      cudaMalloc(&d_hits_, sizeof(Intersection<float3>) * ray_count_ * ss),
      "Failed to allocate device memory for intersection results");
  CUDA_CHECK(cudaMalloc(&d_lights_, sizeof(float) * ray_count_),
             "Failed to allocate device memory for light intensity buffer");
  CUDA_CHECK(
      cudaMalloc(&d_lights_step_, sizeof(float) * ray_count_ * ss),
      "Failed to allocate device memory for light intensity step buffer");
  CUDA_CHECK(cudaMalloc(&d_last_log_intensity_, sizeof(float) * ray_count_),
             "Failed to allocate device memory for last log intensity buffer");
  CUDA_CHECK(
      cudaMemset(d_last_log_intensity_, 0.0f, sizeof(float) * ray_count_),
      "Failed to initialize last log intensity buffer");
  CUDA_CHECK(cudaMalloc(&d_depth_map_, sizeof(float) * ray_count_),
             "Failed to allocate device memory for depth map");
  CUDA_CHECK(cudaMalloc(&d_depth_map_step_, sizeof(float) * ray_count_ * ss),
             "Failed to allocate device memory for depth map step buffer");
  CUDA_CHECK(cudaMalloc(&d_events_, sizeof(Event) * ray_count_),
             "Failed to allocate device memory for events");
  CUDA_CHECK(cudaMalloc(&d_event_count_, sizeof(int)),
             "Failed to allocate device memory for event count");
  // 初始化随机状态
  const int threads = 128;
  int blocks = (ray_count_ + threads - 1) / threads;
  unsigned seed = static_cast<unsigned>(std::time(nullptr));
#ifdef ENABLE_PHYSIC_EFFECTS
  CUDA_CHECK(cudaMalloc(&d_refractory_map_, sizeof(float) * ray_count_),
             "Failed to allocate device memory for refractory map");
  CUDA_CHECK(cudaMemset(d_refractory_map_, 0.0f, sizeof(float) * ray_count_),
             "Failed to initialize refractory map");
  CUDA_CHECK(cudaMalloc(&d_threshold_pos_map_, sizeof(float) * ray_count_),
             "Failed to allocate device memory for positive threshold map");
  CUDA_CHECK(cudaMalloc(&d_threshold_neg_map_, sizeof(float) * ray_count_),
             "Failed to allocate device memory for negative threshold map");
  // 阈值采用高斯核函数填充，均值为设定的阈值，标准差服从于physic_config_.threshold_noise
  // (单位为百分比)
  dim3 block_size(16, 16);
  dim3 grid_size((width_ + block_size.x - 1) / block_size.x,
                 (height_ + block_size.y - 1) / block_size.y);
  threshold_map_init_kernel<<<grid_size, block_size>>>(
      width_, height_, physic_config_.threshold_pos,
      physic_config_.threshold_noise, d_threshold_pos_map_);
  threshold_map_init_kernel<<<grid_size, block_size>>>(
      width_, height_, physic_config_.threshold_neg,
      physic_config_.threshold_noise, d_threshold_neg_map_);
#ifdef ENABLE_HIGH_PASS_FILTER
  CUDA_CHECK(cudaMalloc(&d_last_voltage_, sizeof(float) * ray_count_),
             "Failed to allocate device memory for refractory map");
  CUDA_CHECK(cudaMemset(d_last_voltage_, 0.0f, sizeof(float) * ray_count_),
             "Failed to initialize refractory map");
  if (physic_config_.hpf_time_constant <= 1e-8) {
    alpha_ = 1.0f;
  } else {
    alpha_ = std::exp(-config.scene.time_resolution /
                      physic_config_.hpf_time_constant);
  }
  CUDA_CHECK(cudaMalloc(&d_rand_states_, sizeof(curandState) * ray_count_),
             "Failed to allocate device memory for random states");

  init_rand_states<<<blocks, threads>>>(d_rand_states_, ray_count_, seed);
#ifdef ENABLE_AER_DELAY
  CUDA_CHECK(cudaMalloc(&d_event_count_row_, sizeof(int) * height_),
             "Failed to allocate device memory for event count per row");
  CUDA_CHECK(cudaMalloc(&d_delay_row_, sizeof(float) * height_),
             "Failed to initialize device delay row buffer");
  h_event_count_row_.resize(static_cast<std::size_t>(height_));
  h_delay_row_.resize(static_cast<std::size_t>(height_));
  last_row_index_ = -1;

  unit_row_delay_ = 1 / physic_config_.readout_bandwidth;
#endif
#endif
#endif
  CUDA_CHECK(
      cudaMalloc(&d_super_rand_states_, sizeof(curandState) * ray_count_ * ss),
      "Failed to allocate device memory for random states");
  // 初始化随机状态
  blocks = (ray_count_ * ss + threads - 1) / threads;
  init_rand_states<<<blocks, threads>>>(d_super_rand_states_, ray_count_ * ss,
                                        seed);
}

void DvsRayCaster::initialize(const float frame_intensity) {
  // 初始化last_log_intensity为log(frame_intensity)，确保第一帧能正确生成事件
  float initial_log_intensity = std::log(frame_intensity);
  thrust::device_ptr<float> ptr(d_last_log_intensity_);
  thrust::fill(ptr, ptr + ray_count_, initial_log_intensity);
}

void DvsRayCaster::render_main(const float3 &camera_origin,
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
  // 生成视线
  // dim3 block_size(16, 16);
  // dim3 grid_size((width_ + block_size.x - 1) / block_size.x,
  //                (height_ + block_size.y - 1) / block_size.y);
  dim3 block_size(16, 16, 4); // 每个块处理16x16像素和4次超采样
  dim3 grid_size(
      (width_ + block_size.x - 1) / block_size.x,
      (height_ + block_size.y - 1) / block_size.y,
      (supersample_factor_ * supersample_factor_ + block_size.z - 1) /
          block_size.z);
  view_direction_batch_kernel<<<grid_size, block_size>>>(
      width_, height_, fx_, fy_, cx_, cy_, d_camera_world, camera_origin,
      supersample_factor_, d_super_rand_states_, d_rays_);
  // cudaDeviceSynchronize();
  // err = cudaGetLastError();
  // std::cout << "Error in [Ray Casting]: " << cudaGetErrorString(err) << "\n";
  // 进行批量求交，暴力枚举法
  // intersect_brute_batch_kernel<<<grid_size, block_size>>>(
  //     width, height, tri_count_, node_count_, d_nodes_, d_tris_, d_rays_,
  //     d_hits_);
  const int ss = supersample_factor_ * supersample_factor_;
  const int threads = 128; // 或 256
  const int blocks = (ray_count_ * ss + threads - 1) / threads;

  batch_intersect_kernel<<<blocks, threads>>>(
      d_nodes_, root_index_, d_vertices_, d_normals_, d_indices_, d_rays_,
      camera_origin, ray_count_ * ss, d_hits_, d_depth_map_step_);
#ifdef DEBUG
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Error in [Intersecting]: " << cudaGetErrorString(err) << "\n";
#endif
  cudaFree(d_camera_world);
  // 计算光照，使用Hapke模型
  if (use_albedo_) {
    batch_render_kernel<<<blocks, threads>>>(
        d_hits_, d_nodes_, root_index_, d_vertices_, d_normals_, d_indices_,
        d_albedos_, d_rays_, camera_origin, light_dir, light_intensity,
        hapke_config_, ray_count_ * ss, d_lights_step_);
  } else {
    batch_render_kernel<<<blocks, threads>>>(
        d_hits_, d_nodes_, root_index_, d_vertices_, d_normals_, d_indices_,
        d_rays_, camera_origin, light_dir, light_intensity, hapke_config_,
        ray_count_ * ss, d_lights_step_);
  }
#ifdef DEBUG
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Error in [Rendering]: " << cudaGetErrorString(err) << "\n";
#endif
  dim3 pixel_block_size(16, 16);
  dim3 pixel_grid_size((width_ + pixel_block_size.x - 1) / pixel_block_size.x,
                       (height_ + pixel_block_size.y - 1) / pixel_block_size.y);
  // 归约超采样结果至每个像素
  batch_supersample_fuse_kernel<<<pixel_grid_size, pixel_block_size>>>(
      width_, height_, supersample_factor_, d_lights_step_, d_depth_map_step_,
      d_lights_, d_depth_map_);
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
}

void DvsRayCaster::render_main(const float3 &camera_origin,
                               const float *camera_world,
                               const float3 &light_dir,
                               const float light_intensity,
                               const float current_time,
                               std::vector<Event> &events) {
#ifdef DEBUG
  auto err = cudaGetLastError();
#endif
  float *d_camera_world;
  CUDA_CHECK(cudaMalloc(&d_camera_world, sizeof(float) * 9),
             "Failed to allocate device memory for camera world matrix");
  CUDA_CHECK(cudaMemcpy(d_camera_world, camera_world, sizeof(float) * 9,
                        cudaMemcpyHostToDevice),
             "Failed to copy camera world matrix from host to device");
  // 生成视线
  // dim3 block_size(16, 16);
  // dim3 grid_size((width_ + block_size.x - 1) / block_size.x,
  //                (height_ + block_size.y - 1) / block_size.y);
  dim3 block_size(16, 16, 4); // 每个块处理16x16像素和4次超采样
  dim3 grid_size(
      (width_ + block_size.x - 1) / block_size.x,
      (height_ + block_size.y - 1) / block_size.y,
      (supersample_factor_ * supersample_factor_ + block_size.z - 1) /
          block_size.z);
  view_direction_batch_kernel<<<grid_size, block_size>>>(
      width_, height_, fx_, fy_, cx_, cy_, d_camera_world, camera_origin,
      supersample_factor_, d_super_rand_states_, d_rays_);
  cudaFree(d_camera_world);
#ifdef DEBUG
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Error in [Ray Casting]: " << cudaGetErrorString(err) << "\n";
#endif
  // 进行批量求交，暴力枚举法
  // intersect_brute_batch_kernel<<<grid_size, block_size>>>(
  //     width, height, tri_count_, node_count_, d_nodes_, d_tris_, d_rays_,
  //     d_hits_);
  const int ss = supersample_factor_ * supersample_factor_;
  const int threads = 128; // 或 256
  int blocks = (ray_count_ * ss + threads - 1) / threads;
  batch_intersect_kernel<<<blocks, threads>>>(
      d_nodes_, root_index_, d_vertices_, d_normals_, d_indices_, d_rays_,
      camera_origin, ray_count_ * ss, d_hits_, d_depth_map_step_);
#ifdef DEBUG
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Error in [Intersecting]: " << cudaGetErrorString(err) << "\n";
#endif

  // 计算光照，使用Hapke模型
  if (use_albedo_) {
    batch_render_kernel<<<blocks, threads>>>(
        d_hits_, d_nodes_, root_index_, d_vertices_, d_normals_, d_indices_,
        d_albedos_, d_rays_, camera_origin, light_dir, light_intensity,
        hapke_config_, ray_count_ * ss, d_lights_step_);
  } else {
    batch_render_kernel<<<blocks, threads>>>(
        d_hits_, d_nodes_, root_index_, d_vertices_, d_normals_, d_indices_,
        d_rays_, camera_origin, light_dir, light_intensity, hapke_config_,
        ray_count_ * ss, d_lights_step_);
  }
  dim3 pixel_block_size(16, 16);
  dim3 pixel_grid_size((width_ + pixel_block_size.x - 1) / pixel_block_size.x,
                       (height_ + pixel_block_size.y - 1) / pixel_block_size.y);
  // 归约超采样结果至每个像素
  batch_supersample_fuse_kernel<<<pixel_grid_size, pixel_block_size>>>(
      width_, height_, supersample_factor_, d_lights_step_, d_depth_map_step_,
      d_lights_, d_depth_map_);
#ifdef DEBUG
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Error in [Rendering]: " << cudaGetErrorString(err) << "\n";
#endif
  // 在GPU上生成事件
  CUDA_CHECK(cudaMemset(d_event_count_, 0, sizeof(int)),
             "Failed to initialize event count");

#ifdef ENABLE_PHYSIC_EFFECTS
#ifdef ENABLE_HIGH_PASS_FILTER
#ifdef ENABLE_AER_DELAY
  CUDA_CHECK(cudaMemset(d_event_count_row_, 0, sizeof(int) * height_),
             "Failed to initialize event count per row");
  batch_generate_events_kernel<<<pixel_grid_size, pixel_block_size>>>(
      current_time, d_lights_, width_, height_, d_threshold_pos_map_,
      d_threshold_neg_map_, physic_config_.dark_current,
      physic_config_.refractory_time, alpha_, physic_config_.shot_intensity,
      d_depth_map_, d_refractory_map_, d_last_log_intensity_, d_last_voltage_,
      d_events_, d_event_count_row_, d_event_count_, d_rand_states_);

  CUDA_CHECK(cudaMemcpy(h_event_count_row_.data(), d_event_count_row_,
                        sizeof(int) * height_, cudaMemcpyDeviceToHost),
             "Failed to copy event count per row from device to host");
  row_delay_buffer(height_, h_event_count_row_.data(), unit_row_delay_,
                   physic_config_.readout_latency, h_delay_row_.data(),
                   &last_row_index_);

  CUDA_CHECK(cudaMemcpy(d_delay_row_, h_delay_row_.data(),
                        sizeof(float) * height_, cudaMemcpyHostToDevice),
             "Failed to copy updated delay row from host to device");
  blocks = (ray_count_ + threads - 1) / threads;
  batch_aer_kernel<<<blocks, threads>>>(d_event_count_, d_events_,
                                        d_delay_row_);
#else
  // 考虑噪声模型与高通模型的事件仿真生成器
  batch_generate_events_kernel<<<pixel_grid_size, pixel_block_size>>>(
      current_time, d_lights_, width_, height_, d_threshold_pos_map_,
      d_threshold_neg_map_, physic_config_.dark_current,
      physic_config_.refractory_time, alpha_, physic_config_.shot_intensity,
      d_depth_map_, d_refractory_map_, d_last_log_intensity_, d_last_voltage_,
      d_events_, d_event_count_, d_rand_states_);
#endif
#else
  // 考虑噪声模型的事件仿真生成器
  batch_generate_events_kernel<<<pixel_grid_size, pixel_block_size>>>(
      current_time, d_lights_, width_, height_, d_threshold_pos_map_,
      d_threshold_neg_map_, physic_config_.dark_current,
      physic_config_.refractory_time, physic_config_.shot_intensity,
      d_depth_map_, d_refractory_map_, d_last_log_intensity_, d_events_,
      d_event_count_, d_rand_states_);
#endif
#else
  // 理想事件生成器，不考虑噪声与高通滤波
  batch_generate_events_kernel<<<pixel_grid_size, pixel_block_size>>>(
      current_time, d_lights_, width_, height_, physic_config_.threshold_pos,
      physic_config_.threshold_neg, d_depth_map_, d_last_log_intensity_,
      d_events_, d_event_count_);
#endif

#ifdef DEBUG
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Error in [Event Generation]: " << cudaGetErrorString(err)
            << "\n";
#endif
  int event_count = 0;
  CUDA_CHECK(cudaMemcpy(&event_count, d_event_count_, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "Failed to copy event count from device to host");
  events.resize(static_cast<std::size_t>(event_count));

  CUDA_CHECK(cudaMemcpy(events.data(), d_events_, sizeof(Event) * event_count,
                        cudaMemcpyDeviceToHost),
             "Failed to copy events from device to host");
}
} // namespace dvs