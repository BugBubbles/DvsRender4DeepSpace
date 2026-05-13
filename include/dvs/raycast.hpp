#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "dvs/scene_types.hpp"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    const cudaError_t err__ = (expr);                                          \
    if (err__ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(err__));                     \
    }                                                                          \
  } while (0)

namespace dvs {

__global__ void batch_intersect_kernel(const BVHNode<float3> *nodes,
                                       int root_index,
                                       const Triangle<float3> *triangles,
                                       const Ray<float3> *rays, int ray_count,
                                       Intersection<float3> *out_hits);

__global__ void view_direction_batch_kernel(const int width, const int height,
                                            const float fx, const float fy,
                                            const float cx, const float cy,
                                            const float *camera_world,
                                            const float3 origin,
                                            Ray<float3> *out_rays);

class GpuRaycaster {
public:
  GpuRaycaster(const std::vector<BVHNode<Vec3>> &nodes,
               const std::vector<Triangle<Vec3>> &triangles,
               const int root_index);
  ~GpuRaycaster() {
    if (d_nodes_ != nullptr) {
      cudaFree(d_nodes_);
      d_nodes_ = nullptr;
    }
    if (d_tris_ != nullptr) {
      cudaFree(d_tris_);
      d_tris_ = nullptr;
    }
    if (d_rays_ != nullptr) {
      cudaFree(d_rays_);
      d_rays_ = nullptr;
    }
    if (d_hits_ != nullptr) {
      cudaFree(d_hits_);
      d_hits_ = nullptr;
    }
    node_count_ = 0;
    tri_count_ = 0;
    root_index_ = 0;
    ray_capacity_ = 0;
  };
  void ensure_ray_capacity(std::size_t ray_count) const {
    if (ray_capacity_ >= ray_count) {
      return;
    }
    if (d_rays_ != nullptr) {
      CUDA_CHECK(cudaFree(d_rays_));
      d_rays_ = nullptr;
    }
    if (d_hits_ != nullptr) {
      CUDA_CHECK(cudaFree(d_hits_));
      d_hits_ = nullptr;
    }
    CUDA_CHECK(cudaMalloc(&d_rays_, sizeof(Ray<float3>) * ray_count));
    CUDA_CHECK(cudaMalloc(&d_hits_, sizeof(Intersection<float3>) * ray_count));
    ray_capacity_ = ray_count;
  }
  void intersect(const SceneBundle<Vec3, Uint3, Mat3> &scene, int frame_idx,
                 std::vector<Intersection<Vec3>> &hits);
  void validate_bvh(const std::vector<BVHNode<Vec3>> &nodes, int root) const;

private:
  int node_count_ = 0;
  int tri_count_ = 0;
  int root_index_ = 0;
  BVHNode<float3> *d_nodes_ = nullptr;
  Triangle<float3> *d_tris_ = nullptr;
  mutable Ray<float3> *d_rays_ = nullptr;
  mutable Intersection<float3> *d_hits_ = nullptr;
  mutable std::size_t ray_capacity_ = 0;
};

} // namespace dvs
