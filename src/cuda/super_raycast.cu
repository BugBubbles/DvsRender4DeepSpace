#include "dvs/math_types.hpp"
#include "dvs/raycast.hpp"
#include "dvs/scene_types.hpp"
#include "helper_math.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vector_functions.h>

namespace dvs {
__global__ void
view_direction_batch_kernel(const int width, const int height, const float fx,
                            const float fy, const float cx, const float cy,
                            const float *camera_world, const float3 origin,
                            const int super_sample_count,
                            curandState *rand_states, float3 *out_rays) {
  // xy 是像素坐标索引，z 是超采样索引
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= width || y >= height ||
      z >= super_sample_count * super_sample_count) {
    return;
  }
  int idx = (y * width + x) * super_sample_count * super_sample_count + z;
  // 计算超采样偏移
  int ss_x = z % super_sample_count;
  int ss_y = z / super_sample_count;
  float offset_x =
      (ss_x + curand_uniform(rand_states + idx)) / super_sample_count; // -0.5f
  float offset_y =
      (ss_y + curand_uniform(rand_states + idx)) / super_sample_count; // -0.5f
  // 计算像素中心的视线方向
  float px = (x + offset_x - cx) / fx;            // -0.5+0.5=0
  float py = (y + offset_y - cy) / fy;            // -0.5+0.5=0
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

__global__ void batch_supersample_fuse_kernel(const int width, const int height,
                                              const int super_sample_count,
                                              const float *supersample_lights,
                                              const float *supersample_depths,
                                              float *lights, float *depths) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  int pixel_idx = y * width + x;
  float fused_light = 0.0f;
  float fused_depth = 0.0f;
  int valid_samples = 0;
  for (int ss_y = 0; ss_y < super_sample_count; ++ss_y) {
    for (int ss_x = 0; ss_x < super_sample_count; ++ss_x) {
      int ss_idx = pixel_idx * super_sample_count * super_sample_count +
                   ss_y * super_sample_count + ss_x;
      if (supersample_depths[ss_idx] <= 0.0f) {
        continue; // 跳过无效的深度样本
      }
      fused_light += supersample_lights[ss_idx];
      fused_depth += supersample_depths[ss_idx];
      valid_samples++;
    }
  }
  lights[pixel_idx] = fused_light / valid_samples;
  depths[pixel_idx] = fused_depth / valid_samples;
}
} // namespace dvs