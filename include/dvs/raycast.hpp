#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "dvs/hapke.hpp"
#include "dvs/math_types.hpp"
#include "dvs/scene_types.hpp"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    const cudaError_t err__ = (expr);                                          \
    if (err__ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("[CUDA] error: ") +                 \
                               cudaGetErrorString(err__));                     \
    }                                                                          \
  } while (0)

namespace dvs {
/**
 * @brief 批量光线求交
 * @param nodes BVH节点数组
 * @param root_index BVH根节点索引
 * @param triangles 三角形数组
 * @param rays 输入光线数组
 * @param ray_count 光线数量
 * @param out_hits 输出交点信息数组
 */
__global__ void batch_intersect_kernel(const BVHNode<float3> *nodes,
                                       int root_index,
                                       const Triangle<float3> *triangles,
                                       const float3 *rays,
                                       const float3 ray_origin, int ray_count,
                                       Intersection<float3> *out_hits);
/**
 * @brief 批量生成视线方向
 * @param width 图像宽度
 * @param height 图像高度
 * @param fx 相机焦距x
 * @param fy 相机焦距y
 * @param cx 相机主点x
 * @param cy 相机主点y
 * @param camera_world 相机世界变换矩阵，按行拉直的9个float元素
 * @param origin 光线起点
 * @param out_rays 输出光线数组
 */
__global__ void view_direction_batch_kernel(const int width, const int height,
                                            const float fx, const float fy,
                                            const float cx, const float cy,
                                            const float *camera_world,
                                            const float3 origin,
                                            float3 *out_rays);

/**
 * @brief CPU版本的批量光线求交，用于验证GPU结果
 * @param nodes BVH节点数组
 * @param root_index BVH根节点索引
 * @param triangles 三角形数组
 * @param rays 输入光线数组
 * @param ray_count 光线数量
 * @param out_hits 输出交点信息数组
 */
void batch_intersect_kernel_cpu(const BVHNode<float3> *nodes, int root_index,
                                const Triangle<float3> *triangles,
                                const float3 *rays, const float3 ray_origin,
                                const int ray_count,
                                Intersection<float3> *out_hits);
/**
 * @brief 批量渲染内核，根据交点信息计算光照强度
 * @param out_hits 输入交点信息数组
 * @param rays 输入光线数组
 * @param light_direction 光源方向
 * @param scene_light_intensity 场景光照强度
 * @param hapke_params Hapke模型参数
 * @param ray_count 光线数量
 * @param out_light_intensity 输出光照强度数组
 */
__global__ void batch_render_kernel(const Intersection<float3> *out_hits,
                                    const float3 *rays, const float3 ray_origin,
                                    const float3 light_direction,
                                    const float scene_light_intensity,
                                    const HapkeParams hapke_params,
                                    const int ray_count,
                                    float *out_light_intensity);
class GpuRaycaster {
public:
  GpuRaycaster(const SceneBundle<Vec3, Uint3, Mat3> &scene,
               const int root_index,
               const HapkeParams &hapke_params = HapkeParams());
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
    if (d_lights_ != nullptr) {
      cudaFree(d_lights_);
      d_lights_ = nullptr;
    }
    node_count_ = 0;
    tri_count_ = 0;
    root_index_ = 0;
    ray_capacity_ = 0;
    ray_count_ = 0;
  };

  void render_main(const FrameData<Vec3, Mat3> &frame,
                   std::vector<float> &light_frame);

  void validate_bvh(const std::vector<BVHNode<Vec3>> &nodes, int root) const;

private:
  int node_count_ = 0;
  int tri_count_ = 0;
  int root_index_ = 0;
  int ray_count_ = 0;
  // 相机参数
  float fx_, fy_, cx_, cy_;
  int width_, height_;
  // Hapke模型参数
  HapkeParams hapke_params_;

  // GPU数据
  BVHNode<float3> *d_nodes_ = nullptr;
  Triangle<float3> *d_tris_ = nullptr;
  mutable float3 *d_rays_ = nullptr;
  mutable float3 ray_origin_;
  mutable Intersection<float3> *d_hits_ = nullptr;
  mutable float *d_lights_ = nullptr;
  mutable std::size_t ray_capacity_ = 0;
};

} // namespace dvs
