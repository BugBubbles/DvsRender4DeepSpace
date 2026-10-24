#pragma once

#include "dvs/math_types.hpp"
#include <vector>
#include <cmath>
#include <limits>
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
  double timestamp_seconds = std::numeric_limits<double>::quiet_NaN();
  float3 camera_origin;
  mat3 camera_world;
  float3 light_dir;
  float light_intensity = 0.0f;
};

// Binary Blender frames store seconds; text trajectories store frame indices.
template <typename V, typename M>
inline double frame_time(const FrameData<V,M> &frame, double fps) {
  return std::isnan(frame.timestamp_seconds)?frame.frame_index/fps:frame.timestamp_seconds;
}

// 相机参数系列
struct CameraConfig {
  int width, height;
  float fx, fy, cx, cy;
};
// 事件相机物理噪声参数
struct PhysicConfig {
  float threshold_pos=.3f, threshold_neg=.3f, refractory_time=.0002f;
  float readout_bandwidth=5e6f, readout_latency=0;
  // Dimensioned optical and reduced circuit model (generic, uncalibrated).
  float threshold_std = 0.03f, threshold_min = 0.01f;
  float wavelength_nm=550, quantum_efficiency=.3f, pixel_pitch_um=15;
  float fill_factor=.2f, f_number=2, lens_transmission=.8f;
  float radiance_scale=.001f; // W m^-2 sr^-1 per renderer unit
  float temperature_k=300, subthreshold_kappa=.8f;
  float receptor_capacitance_f=1e-15f; // effective feedback capacitance
  float reset_capacitance_f=20e-15f, dark_current_a=1e-18f;
  float reset_leakage_a=0, circuit_step=1e-6f;
  int photon_shot_noise=1, reset_thermal_noise=0;
  int random_seed = 114514, max_events_per_step = 16;
  int readout_queue_capacity = 1000000;
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
