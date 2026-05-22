#pragma once

#include <vector>

#include "dvs/dvs_base.hpp"
#include "dvs/scene_types.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <vector_types.h>

#define RAND_SEED 114514

namespace dvs {
/**
 * @brief 初始化每条光线的随机状态，用于超采样时生成随机偏移
 * @param states 每条光线对应的随机状态数组
 * @param count 光线的总数量（即像素数量）
 * @param seed 随机数生成的种子
 */
__global__ void init_rand_states(curandState *states, int count, unsigned seed);
/**
 * @brief 批量光线求交
 * @param nodes BVH节点数组
 * @param root_index BVH根节点索引
 * @param triangles 三角形数组
 * @param rays 输入光线数组
 * @param ray_count 光线数量
 * @param out_hits 输出交点信息数组
 * @param depth_map 输出深度图数组
 */
__global__ void
batch_intersect_kernel(const BVHNode<float3> *nodes, int root_index,
                       const Triangle<float3> *triangles, const float3 *rays,
                       const float3 ray_origin, int ray_count,
                       Intersection<float3> *out_hits, float *depth_map);
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

__global__ void view_direction_batch_kernel(
    const int width, const int height, const int delta_x, const int delta_y,
    const int supersample_factor, const float fx, const float fy,
    const float cx, const float cy, const float *camera_world,
    const float3 origin, float3 *out_rays, curandState *rand_states);
/**
 * @brief CPU版本的批量光线求交，用于验证GPU结果
 * @param nodes BVH节点数组
 * @param root_index BVH根节点索引
 * @param triangles 三角形数组
 * @param rays 输入光线数组
 * @param ray_count 光线数量
 * @param out_hits 输出交点信息数组
 * @param depth_map 输出深度图数组
 */
void batch_intersect_kernel_cpu(const BVHNode<float3> *nodes, int root_index,
                                const Triangle<float3> *triangles,
                                const float3 *rays, const float3 ray_origin,
                                const int ray_count,
                                Intersection<float3> *out_hits,
                                float *depth_map);
/**
 * @brief 批量渲染内核，根据交点信息计算光照强度
 * @param out_hits 输入交点信息数组
 * @param rays 输入光线数组
 * @param light_direction 光源方向
 * @param scene_light_intensity 场景光照强度
 * @param hapke_config Hapke模型参数
 * @param ray_count 光线数量
 * @param out_light_intensity 输出光照强度数组
 */
__global__ void batch_render_kernel(
    const Intersection<float3> *out_hits, const Triangle<float3> *triangles,
    const float3 *rays, const float3 ray_origin, const float3 light_direction,
    const float scene_light_intensity, const HapkeConfig hapke_config,
    const int ray_count, float *out_light_intensity);

class DvsRayCaster {
public:
  DvsRayCaster(const SceneBundle<float3> &scene, const Config &config,
               const int root_index);
  ~DvsRayCaster() {
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
    if (d_last_log_intensity_ != nullptr) {
      cudaFree(d_last_log_intensity_);
      d_last_log_intensity_ = nullptr;
    }
    if (d_depth_map_ != nullptr) {
      cudaFree(d_depth_map_);
      d_depth_map_ = nullptr;
    }
    if (d_events_ != nullptr) {
      cudaFree(d_events_);
      d_events_ = nullptr;
    }
    if (d_event_count_ != nullptr) {
      cudaFree(d_event_count_);
      d_event_count_ = nullptr;
    }
#ifdef ENABLE_PHYSIC_EFFECTS
    if (d_refractory_map_ != nullptr) {
      cudaFree(d_refractory_map_);
      d_refractory_map_ = nullptr;
    }
    if (d_threshold_pos_map_ != nullptr) {
      cudaFree(d_threshold_pos_map_);
      d_threshold_pos_map_ = nullptr;
    }
    if (d_threshold_neg_map_ != nullptr) {
      cudaFree(d_threshold_neg_map_);
      d_threshold_neg_map_ = nullptr;
    }
#endif
#ifdef ENABLE_PHYSIC_EFFECTS
#ifdef ENABLE_HIGH_PASS_FILTER
    if (d_last_voltage_ != nullptr) {
      cudaFree(d_last_voltage_);
      d_last_voltage_ = nullptr;
    }
    if (d_rand_states_ != nullptr) {
      cudaFree(d_rand_states_);
      d_rand_states_ = nullptr;
    }
#ifdef ENABLE_AER_DELAY
    if (d_event_count_row_ != nullptr) {
      cudaFree(d_event_count_row_);
      d_event_count_row_ = nullptr;
    }
    if (d_delay_row_ != nullptr) {
      cudaFree(d_delay_row_);
      d_delay_row_ = nullptr;
    }
#endif
#endif
#endif
    node_count_ = 0;
    tri_count_ = 0;
    root_index_ = 0;
    ray_capacity_ = 0;
    ray_count_ = 0;
  };

  void render_main(const float3 &camera_origin, const float *camera_world,
                   const float3 &light_dir, const float light_intensity,
                   std::vector<float> &light_frame);

  void render_main(const float3 &camera_origin, const float *camera_world,
                   const float3 &light_dir, const float light_intensity,
                   const float current_time, std::vector<Event> &events,
                   float *time_records);

  void initialize(const float frame_intensity);

private:
  int node_count_ = 0;
  int tri_count_ = 0;
  int root_index_ = 0;
  int ray_count_ = 0;
  // 相机参数
  float fx_, fy_, cx_, cy_;
  int width_, height_;
  // 事件相关参数
  PhysicConfig physic_config_;
  SceneConfig scene_config_;
  // 上一帧的log强度，用于事件生成
  float *d_last_log_intensity_ = nullptr;
  float *d_depth_map_ = nullptr;
#ifdef ENABLE_PHYSIC_EFFECTS
  // 不应期图，避免过快生成事件
  float *d_refractory_map_ = nullptr;
  float *d_threshold_pos_map_ = nullptr;
  float *d_threshold_neg_map_ = nullptr;
#ifdef ENABLE_HIGH_PASS_FILTER
  mutable float *d_last_voltage_ = nullptr;
  float alpha_;
  mutable curandState *d_rand_states_ = nullptr;
#ifdef ENABLE_AER_DELAY
  int *d_event_count_row_ = nullptr;
  float *d_delay_row_ = nullptr;
  std::vector<int> h_event_count_row_;
  std::vector<float> h_delay_row_;
  int last_row_index_;
  float unit_row_delay_;
#endif
#endif
#endif
  // 渲染相关参数
  HapkeConfig hapke_config_;

  // GPU数据
  BVHNode<float3> *d_nodes_ = nullptr;
  Triangle<float3> *d_tris_ = nullptr;
  mutable float3 *d_rays_ = nullptr;
  mutable float3 ray_origin_;
  mutable Intersection<float3> *d_hits_ = nullptr;
  mutable float *d_lights_ = nullptr;
  mutable Event *d_events_ = nullptr;
  mutable int *d_event_count_ = nullptr;
  mutable std::size_t ray_capacity_ = 0;
};

class CamRayCaster {
public:
  CamRayCaster(const SceneBundle<float3> &scene, const Config &config,
               const int root_index);
  ~CamRayCaster() {
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
    if (d_lights_step_ != nullptr) {
      cudaFree(d_lights_step_);
      d_lights_step_ = nullptr;
    }
    if (d_depth_map_ != nullptr) {
      cudaFree(d_depth_map_);
      d_depth_map_ = nullptr;
    }
    if (d_depth_map_step_ != nullptr) {
      cudaFree(d_depth_map_step_);
      d_depth_map_step_ = nullptr;
    }
    if (d_rand_states_ != nullptr) {
      cudaFree(d_rand_states_);
      d_rand_states_ = nullptr;
    }
    node_count_ = 0;
    tri_count_ = 0;
    root_index_ = 0;
    ray_capacity_ = 0;
    ray_count_ = 0;
  };

  void render_main(const float3 &camera_origin, const float *camera_world,
                   const float3 &light_dir, const float light_intensity,
                   std::vector<float> &light_frame,
                   std::vector<float> &depth_map);

protected:
  int node_count_ = 0;
  int tri_count_ = 0;
  int root_index_ = 0;
  int ray_count_ = 0;
  // 相机参数
  float fx_, fy_, cx_, cy_;
  int width_, height_;
  int supersample_factor_;
  // 渲染相关参数
  HapkeConfig hapke_config_;
  SceneConfig scene_config_;
  // GPU数据
  BVHNode<float3> *d_nodes_ = nullptr;
  Triangle<float3> *d_tris_ = nullptr;
  mutable float3 *d_rays_ = nullptr;
  mutable float3 ray_origin_;
  mutable Intersection<float3> *d_hits_ = nullptr;
  mutable float *d_lights_ = nullptr;
  mutable float *d_lights_step_ = nullptr;
  mutable curandState *d_rand_states_ = nullptr;
  mutable float *d_depth_map_ = nullptr;
  mutable float *d_depth_map_step_ = nullptr;
  mutable std::size_t ray_capacity_ = 0;
};

class CamBlurRayCaster : public CamRayCaster {
public:
  using CamRayCaster::CamRayCaster;
  void render_main(const float3 &camera_origin, const float *camera_world,
                   const float3 &light_dir, const float light_intensity,
                   std::vector<float> &light_frame);
};
} // namespace dvs
