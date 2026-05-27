#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector_types.h>

#include "dvs/math_types.hpp"
#include "dvs/scene_types.hpp"

namespace dvs {

inline Vec3 float3_to_vec3(const float3 &f) { return Vec3(f.x, f.y, f.z); }
inline float3 vec3_to_float3(const Vec3 &v) {
  return float3{v.x(), v.y(), v.z()};
}
/**
 * @brief 任意时间处的相机与光照状态。
 *
 * position:
 *   世界坐标系下的相机位置。
 *
 * orientation:
 *   camera-to-world 旋转矩阵，语义与 FrameData::camera_world 保持一致。
 *
 * quaternion:
 *   camera-to-world 四元数。
 *
 * linear_velocity:
 *   世界坐标系下的相机线速度 dp/dt。
 *
 * angular_velocity:
 *   世界坐标系下的相机角速度 omega，满足 Rdot = [omega]_x R。
 *
 * light_direction:
 *   插值后的世界坐标系光照方向，已归一化。
 *
 * light_direction_rate:
 *   light_direction 的时间导数，解析求解，不使用差分。
 *
 * light_intensity:
 *   插值后的光照强度。
 *
 * light_intensity_rate:
 *   光照强度的一阶时间导数。
 */
struct CameraTrajectoryState {
  float timestamp = 0.0f;

  Vec3 position = Vec3::Zero();
  Mat3 orientation = Mat3::Identity();
  Eigen::Quaternionf quaternion = Eigen::Quaternionf::Identity();

  Vec3 linear_velocity = Vec3::Zero();
  Vec3 angular_velocity = Vec3::Zero();

  Vec3 light_direction = Vec3(0.0f, 0.0f, 1.0f);
  Vec3 light_direction_rate = Vec3::Zero();

  float light_intensity = 0.0f;
  float light_intensity_rate = 0.0f;
};

/**
 * @brief Blender 风格 cubic Bezier 相机/光照轨迹插值器。
 *
 * 支持解析查询：
 *   - 相机位置
 *   - 相机姿态
 *   - 相机线速度
 *   - 相机角速度
 *   - 光照方向
 *   - 光照方向变化率
 *   - 光照强度
 *   - 光照强度变化率
 *
 * 输入数据来自 SceneBundle::frames。
 */
class CameraTrajectory {
public:
  CameraTrajectory() = default;

  CameraTrajectory(const std::vector<FrameData<Vec3, Mat3>> &frames,
                   SceneConfig &scene_config) {
    build(frames, scene_config.fps, scene_config.translation,
          scene_config.rotation);
  }

  void build(const std::vector<FrameData<Vec3, Mat3>> &frames, float fps,
             std::vector<float> &translation, std::vector<float> &rotation);

  bool valid() const;
  std::size_t size() const;

  float start_time() const;
  float end_time() const;

  CameraTrajectoryState evaluate(float t) const;

  Vec3 position(float t) const;
  Mat3 orientation(float t) const;
  Eigen::Quaternionf quaternion(float t) const;
  Vec3 linear_velocity(float t) const;
  Vec3 angular_velocity(float t) const;

  Vec3 light_direction(float t) const;
  Vec3 light_direction_rate(float t) const;
  float light_intensity(float t) const;
  float light_intensity_rate(float t) const;

private:
  struct ScalarKey {
    float time = 0.0f;
    float value = 0.0f;

    float left_time = 0.0f;
    float left_value = 0.0f;

    float right_time = 0.0f;
    float right_value = 0.0f;
  };

  class ScalarBezierCurve {
  public:
    void clear();

    void build_auto_clamped(const std::vector<float> &times,
                            const std::vector<float> &values);

    bool valid() const;
    std::size_t size() const;

    float start_time() const;
    float end_time() const;

    float evaluate(float t) const;
    float derivative(float t) const;

  private:
    std::vector<ScalarKey> keys_;

    int find_segment(float t) const;

    static float bezier(float p0, float p1, float p2, float p3, float u);

    static float bezier_derivative(float p0, float p1, float p2, float p3,
                                   float u);

    static float solve_bezier_time(const ScalarKey &k0, const ScalarKey &k1,
                                   float t);
  };
  class RotationBezierCurve {
  public:
    void clear();

    void build(const std::vector<float> &times,
               const std::vector<Eigen::Quaternionf> &rotations);

    bool valid() const;
    std::size_t size() const;

    float start_time() const;
    float end_time() const;

    void evaluate(float t, Eigen::Quaternionf *out_q, Mat3 *out_R,
                  Vec3 *out_angular_velocity_world) const;

  private:
    std::vector<float> times_;
    std::vector<Eigen::Quaternionf> rotations_;

    // phase 曲线的 key 值为 0, 1, 2, ..., N-1。
    // 对任意时间 t，phase(t) = i + alpha。
    // i 是当前段索引，alpha 是当前段内的归一化插值参数。
    ScalarBezierCurve phase_;

  private:
    int find_segment_from_phase(float phase) const;

    static Eigen::Quaternionf quat_exp(const Vec3 &rotvec);
    static Vec3 quat_log(const Eigen::Quaternionf &q);
  };

private:
  static Eigen::Quaternionf
  matrix_to_quaternion_continuous(const Mat3 &R,
                                  const Eigen::Quaternionf *previous);

  static Vec3 normalized_vector_derivative(const Vec3 &v_raw,
                                           const Vec3 &v_raw_dot);

private:
  float fps_ = 0.0f;

  // Camera position.
  ScalarBezierCurve px_;
  ScalarBezierCurve py_;
  ScalarBezierCurve pz_;

  // Camera orientation quaternion.
  RotationBezierCurve rotation_;

  // Light direction.
  ScalarBezierCurve lx_;
  ScalarBezierCurve ly_;
  ScalarBezierCurve lz_;

  // Light intensity.
  ScalarBezierCurve light_intensity_;
};

} // namespace dvs