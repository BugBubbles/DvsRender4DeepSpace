#pragma once

#include "dvs/math_types.hpp"
#include <vector>
#ifdef DEBUG
#define DEBUG_LOG(msg) std::cout << msg << std::endl;
#else
#define DEBUG_LOG(msg)
#endif
namespace dvs {

template <typename float3, typename uint3> struct MeshData {
  std::vector<float3> vertices, normals;
  std::vector<uint3> indices;
  std::vector<float> albedos;
};

template <typename float3> struct Triangle {
  float3 v0, v1, v2;
  float3 n0, n1, n2;
};

template <typename float3> struct AABB {
  float3 min, max;
};

template <typename float3> struct BVHNode {
  AABB<float3> box;
  int left = -1;
  int right = -1;
  int start = 0;
  int count = 0;
};

template <typename float3> struct Intersection {
  int hit = 0;
  int triangle_index = -1;
  float t, u, v;
};

template <typename float3, typename mat3> struct FrameData {
  int frame_index = 0;
  float3 camera_origin;
  mat3 camera_world;
  float3 light_dir;
  float light_intensity = 0.0f;
};

// 相机参数系列
struct CameraConfig {
  int width, height;
  float fx, fy, cx, cy;
};
// 事件相机物理噪声参数
struct PhysicConfig {
  float threshold_pos, threshold_neg, threshold_noise;
  float dark_current, refractory_time;
  float hpf_time_constant;
  // 散料噪声参数
  float shot_intensity;
  // AER读出延迟参数
  float readout_bandwidth, readout_latency;
};
// 场景配置参数
struct SceneConfig {
  int frame_count;
  float fps, imu_fps, cam_fps;
  float time_resolution, blur_resolution;
  int supersample_factor;
  std::vector<float> translation, rotation;
};
// Hapke模型参数
struct HapkeConfig {
  float b = 0.24f;
  float c = 0.305f;
  float hS = 0.08f;
  float hC = 1.0f;
  float thetap = 26.0f * kPi / 180.0f;
  float w = 0.324f;
  float B_S0 = 2.42f;
  float B_C0 = 0.0f;
  float phi = 0.3f;
};
// 参数集合
class Config {
public:
  CameraConfig camera;
  PhysicConfig physic;
  SceneConfig scene;
  HapkeConfig hapke;

  Config(const std::string &config_file, std::string &err);
};

template <typename float3, typename uint3> struct SceneBundle {
  std::vector<BVHNode<float3>> bvh_nodes;
  // std::vector<Triangle<float3>> bvh_triangles;
  std::vector<float3> vertices, normals;
  std::vector<uint3> bvh_triangles;
  std::vector<float> albedos;
  bool use_albedo = false;
};

} // namespace dvs
