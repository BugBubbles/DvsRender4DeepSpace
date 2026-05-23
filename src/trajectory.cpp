#include "dvs/trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dvs {
namespace {

float clamp_float(float x, float lo, float hi) {
  return std::max(lo, std::min(x, hi));
}

bool nearly_equal(float a, float b, float eps = 1e-7f) {
  return std::fabs(a - b) <= eps;
}

/**
 * @brief Fritsch-Carlson / monotone cubic 风格的斜率构造。
 *
 * 这个函数用于在只有离散 samples、没有 Blender 原始 handle 的情况下，
 * 构造稳定的 Auto-Clamped 风格切线。
 *
 * 如果相邻两段斜率异号，则当前 key 处切线设为 0，避免过冲。
 */
std::vector<float> compute_auto_clamped_tangents(const std::vector<float> &t,
                                                 const std::vector<float> &y) {
  const std::size_t n = t.size();
  std::vector<float> m(n, 0.0f);

  if (n <= 1) {
    return m;
  }

  if (n == 2) {
    const float h = t[1] - t[0];
    const float d = (std::fabs(h) > 1e-12f) ? (y[1] - y[0]) / h : 0.0f;
    m[0] = d;
    m[1] = d;
    return m;
  }

  std::vector<float> h(n - 1, 0.0f);
  std::vector<float> d(n - 1, 0.0f);

  for (std::size_t i = 0; i + 1 < n; ++i) {
    h[i] = t[i + 1] - t[i];
    if (std::fabs(h[i]) > 1e-12f) {
      d[i] = (y[i + 1] - y[i]) / h[i];
    } else {
      d[i] = 0.0f;
    }
  }

  m[0] = d[0];
  m[n - 1] = d[n - 2];

  for (std::size_t i = 1; i + 1 < n; ++i) {
    const float d0 = d[i - 1];
    const float d1 = d[i];

    if (d0 == 0.0f || d1 == 0.0f || d0 * d1 <= 0.0f) {
      m[i] = 0.0f;
      continue;
    }

    const float h0 = h[i - 1];
    const float h1 = h[i];

    const float w1 = 2.0f * h1 + h0;
    const float w2 = h1 + 2.0f * h0;

    const float denom = w1 / d0 + w2 / d1;
    if (std::fabs(denom) > 1e-12f) {
      m[i] = (w1 + w2) / denom;
    } else {
      m[i] = 0.0f;
    }
  }

  return m;
}

} // namespace

// ============================================================================
// CameraTrajectory::ScalarBezierCurve
// ============================================================================

void CameraTrajectory::ScalarBezierCurve::clear() { keys_.clear(); }

bool CameraTrajectory::ScalarBezierCurve::valid() const {
  return !keys_.empty();
}

std::size_t CameraTrajectory::ScalarBezierCurve::size() const {
  return keys_.size();
}

float CameraTrajectory::ScalarBezierCurve::start_time() const {
  if (keys_.empty()) {
    return 0.0f;
  }
  return keys_.front().time;
}

float CameraTrajectory::ScalarBezierCurve::end_time() const {
  if (keys_.empty()) {
    return 0.0f;
  }
  return keys_.back().time;
}

float CameraTrajectory::ScalarBezierCurve::bezier(float p0, float p1, float p2,
                                                  float p3, float u) {
  const float v = 1.0f - u;
  return v * v * v * p0 + 3.0f * v * v * u * p1 + 3.0f * v * u * u * p2 +
         u * u * u * p3;
}

float CameraTrajectory::ScalarBezierCurve::bezier_derivative(float p0, float p1,
                                                             float p2, float p3,
                                                             float u) {
  const float v = 1.0f - u;
  return 3.0f * v * v * (p1 - p0) + 6.0f * v * u * (p2 - p1) +
         3.0f * u * u * (p3 - p2);
}

void CameraTrajectory::ScalarBezierCurve::build_auto_clamped(
    const std::vector<float> &times, const std::vector<float> &values) {
  clear();

  if (times.size() != values.size()) {
    throw std::invalid_argument(
        "ScalarBezierCurve::build_auto_clamped: size mismatch");
  }

  const std::size_t n = times.size();
  if (n == 0) {
    return;
  }

  keys_.resize(n);

  if (n == 1) {
    keys_[0].time = times[0];
    keys_[0].value = values[0];
    keys_[0].left_time = times[0];
    keys_[0].right_time = times[0];
    keys_[0].left_value = values[0];
    keys_[0].right_value = values[0];
    return;
  }

  for (std::size_t i = 1; i < n; ++i) {
    if (!(times[i] > times[i - 1])) {
      throw std::invalid_argument("ScalarBezierCurve::build_auto_clamped: "
                                  "timestamps must be strictly increasing");
    }
  }

  const std::vector<float> tangent =
      compute_auto_clamped_tangents(times, values);

  for (std::size_t i = 0; i < n; ++i) {
    keys_[i].time = times[i];
    keys_[i].value = values[i];

    keys_[i].left_time = times[i];
    keys_[i].right_time = times[i];

    keys_[i].left_value = values[i];
    keys_[i].right_value = values[i];
  }

  for (std::size_t i = 0; i + 1 < n; ++i) {
    const float t0 = times[i];
    const float t1 = times[i + 1];
    const float h = t1 - t0;

    keys_[i].right_time = t0 + h / 3.0f;
    keys_[i].right_value = values[i] + tangent[i] * h / 3.0f;

    keys_[i + 1].left_time = t1 - h / 3.0f;
    keys_[i + 1].left_value = values[i + 1] - tangent[i + 1] * h / 3.0f;
  }
}

int CameraTrajectory::ScalarBezierCurve::find_segment(float t) const {
  const int n = static_cast<int>(keys_.size());

  if (n < 2) {
    return -1;
  }

  if (t <= keys_.front().time) {
    return 0;
  }

  if (t >= keys_.back().time) {
    return n - 2;
  }

  auto it =
      std::upper_bound(keys_.begin(), keys_.end(), t,
                       [](float x, const ScalarKey &k) { return x < k.time; });

  int idx1 = static_cast<int>(it - keys_.begin());
  int idx0 = idx1 - 1;

  idx0 = std::max(0, std::min(idx0, n - 2));
  return idx0;
}

float CameraTrajectory::ScalarBezierCurve::solve_bezier_time(
    const ScalarKey &k0, const ScalarKey &k1, float t) {
  const float x0 = k0.time;
  const float x1 = k0.right_time;
  const float x2 = k1.left_time;
  const float x3 = k1.time;

  const float span = x3 - x0;
  if (span <= 1e-12f) {
    return 0.0f;
  }

  float u = clamp_float((t - x0) / span, 0.0f, 1.0f);

  // Newton iteration, same idea as Blender F-Curve Bezier evaluation:
  // solve x(u) = t.
  for (int i = 0; i < 8; ++i) {
    const float x = bezier(x0, x1, x2, x3, u);
    const float dx = bezier_derivative(x0, x1, x2, x3, u);

    if (std::fabs(dx) < 1e-10f) {
      break;
    }

    const float next = clamp_float(u - (x - t) / dx, 0.0f, 1.0f);

    if (std::fabs(next - u) < 1e-7f) {
      u = next;
      return u;
    }

    u = next;
  }

  // Robust fallback: bisection.
  float lo = 0.0f;
  float hi = 1.0f;

  for (int i = 0; i < 24; ++i) {
    const float mid = 0.5f * (lo + hi);
    const float x = bezier(x0, x1, x2, x3, mid);

    if (x < t) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return 0.5f * (lo + hi);
}

float CameraTrajectory::ScalarBezierCurve::evaluate(float t) const {
  if (keys_.empty()) {
    return 0.0f;
  }

  if (keys_.size() == 1) {
    return keys_.front().value;
  }

  if (t <= keys_.front().time) {
    return keys_.front().value;
  }

  if (t >= keys_.back().time) {
    return keys_.back().value;
  }

  const int seg = find_segment(t);
  const ScalarKey &k0 = keys_[static_cast<std::size_t>(seg)];
  const ScalarKey &k1 = keys_[static_cast<std::size_t>(seg + 1)];

  const float u = solve_bezier_time(k0, k1, t);

  return bezier(k0.value, k0.right_value, k1.left_value, k1.value, u);
}

float CameraTrajectory::ScalarBezierCurve::derivative(float t) const {
  if (keys_.size() < 2) {
    return 0.0f;
  }

  // 将查询时间钳位到曲线定义域内，并避免严格等于端点
  const float lo = keys_.front().time;
  const float hi = keys_.back().time;
  const float eps = 1e-7f;
  float tc = t;
  if (tc < lo + eps)
    tc = lo + eps;
  if (tc > hi - eps)
    tc = hi - eps;

  const int seg = find_segment(tc); // 使用钳位后的时间
  const ScalarKey &k0 = keys_[static_cast<std::size_t>(seg)];
  const ScalarKey &k1 = keys_[static_cast<std::size_t>(seg + 1)];

  const float u = solve_bezier_time(k0, k1, t);

  const float dx_du =
      bezier_derivative(k0.time, k0.right_time, k1.left_time, k1.time, u);

  const float dy_du =
      bezier_derivative(k0.value, k0.right_value, k1.left_value, k1.value, u);

  if (std::fabs(dx_du) < 1e-12f) {
    return 0.0f;
  }

  return dy_du / dx_du;
}
// ============================================================================
// CameraTrajectory::RotationBezierCurve
// ============================================================================

void CameraTrajectory::RotationBezierCurve::clear() {
  times_.clear();
  rotations_.clear();
  phase_.clear();
}

bool CameraTrajectory::RotationBezierCurve::valid() const {
  return !times_.empty() && times_.size() == rotations_.size();
}

std::size_t CameraTrajectory::RotationBezierCurve::size() const {
  return rotations_.size();
}

float CameraTrajectory::RotationBezierCurve::start_time() const {
  if (times_.empty()) {
    return 0.0f;
  }
  return times_.front();
}

float CameraTrajectory::RotationBezierCurve::end_time() const {
  if (times_.empty()) {
    return 0.0f;
  }
  return times_.back();
}

Eigen::Quaternionf
CameraTrajectory::RotationBezierCurve::quat_exp(const Vec3 &rotvec) {
  const float theta = rotvec.norm();

  if (theta < 1e-8f) {
    // Exp(phi) ≈ [1, 0.5 * phi]
    Eigen::Quaternionf q(1.0f, 0.5f * rotvec.x(), 0.5f * rotvec.y(),
                         0.5f * rotvec.z());
    q.normalize();
    return q;
  }

  const Vec3 axis = rotvec / theta;
  const float half_theta = 0.5f * theta;
  const float s = std::sin(half_theta);

  Eigen::Quaternionf q(std::cos(half_theta), s * axis.x(), s * axis.y(),
                       s * axis.z());
  q.normalize();
  return q;
}

Vec3 CameraTrajectory::RotationBezierCurve::quat_log(
    const Eigen::Quaternionf &q_in) {
  Eigen::Quaternionf q = q_in;
  q.normalize();

  // 使用短弧，避免绕远路。
  if (q.w() < 0.0f) {
    q.coeffs() *= -1.0f;
  }

  const Vec3 v(q.x(), q.y(), q.z());
  const float sin_half_theta = v.norm();

  if (sin_half_theta < 1e-8f) {
    // Log(q) ≈ 2 * vector part
    return 2.0f * v;
  }

  const float half_theta = std::atan2(sin_half_theta, q.w());
  const float theta = 2.0f * half_theta;

  return v * (theta / sin_half_theta);
}

void CameraTrajectory::RotationBezierCurve::build(
    const std::vector<float> &times,
    const std::vector<Eigen::Quaternionf> &rotations) {
  clear();

  if (times.size() != rotations.size()) {
    throw std::invalid_argument("RotationBezierCurve::build: size mismatch");
  }

  const std::size_t n = times.size();
  if (n == 0) {
    return;
  }

  for (std::size_t i = 1; i < n; ++i) {
    if (!(times[i] > times[i - 1])) {
      throw std::invalid_argument(
          "RotationBezierCurve::build: timestamps must be strictly increasing");
    }
  }

  times_ = times;
  rotations_.resize(n);

  for (std::size_t i = 0; i < n; ++i) {
    rotations_[i] = rotations[i];
    rotations_[i].normalize();

    // 再次保证四元数符号连续。
    if (i > 0 && rotations_[i].dot(rotations_[i - 1]) < 0.0f) {
      rotations_[i].coeffs() *= -1.0f;
    }
  }

  // 构建 phase 曲线：
  //   key 0 -> 0
  //   key 1 -> 1
  //   key 2 -> 2
  //   ...
  //
  // phase(t) = i + alpha
  // alpha_dot = d phase / dt
  std::vector<float> phase_values(n, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    phase_values[i] = static_cast<float>(i);
  }

  phase_.build_auto_clamped(times_, phase_values);
}

int CameraTrajectory::RotationBezierCurve::find_segment_from_phase(
    float phase) const {
  const int n = static_cast<int>(rotations_.size());

  if (n < 2) {
    return -1;
  }

  int seg = static_cast<int>(std::floor(phase));

  if (seg < 0) {
    seg = 0;
  }

  if (seg > n - 2) {
    seg = n - 2;
  }

  return seg;
}

void CameraTrajectory::RotationBezierCurve::evaluate(
    float t, Eigen::Quaternionf *out_q, Mat3 *out_R,
    Vec3 *out_angular_velocity_world) const {
  if (out_q == nullptr || out_R == nullptr ||
      out_angular_velocity_world == nullptr) {
    return;
  }

  *out_q = Eigen::Quaternionf::Identity();
  *out_R = Mat3::Identity();
  *out_angular_velocity_world = Vec3::Zero();

  if (!valid()) {
    return;
  }

  const float lo = times_.front();
  const float hi = times_.back();
  const float eps = 1e-7f;
  float tc = t;
  if (tc < lo + eps)
    tc = lo + eps;
  if (tc > hi - eps)
    tc = hi - eps;

  const float phase = phase_.evaluate(t);
  const float phase_dot = phase_.derivative(t);

  const int seg = find_segment_from_phase(phase);

  if (seg < 0) {
    return;
  }

  float alpha = phase - static_cast<float>(seg);
  alpha = clamp_float(alpha, 0.0f, 1.0f);

  const Eigen::Quaternionf &q0 = rotations_[static_cast<std::size_t>(seg)];
  const Eigen::Quaternionf &q1 = rotations_[static_cast<std::size_t>(seg + 1)];

  // 相对旋转：
  //
  //   q_rel = q0^{-1} * q1
  //
  // 对应：
  //
  //   R_rel = R0^T * R1
  //
  Eigen::Quaternionf q_rel = q0.conjugate() * q1;
  q_rel.normalize();

  // 短弧处理。
  if (q_rel.w() < 0.0f) {
    q_rel.coeffs() *= -1.0f;
  }

  // rho = Log(q0^{-1} q1)
  const Vec3 rho = quat_log(q_rel);

  // q(t) = q0 * Exp(alpha * rho)
  const Eigen::Quaternionf dq = quat_exp(alpha * rho);

  Eigen::Quaternionf q = q0 * dq;
  q.normalize();

  const Mat3 R = q.toRotationMatrix();

  // 李代数曲线：
  //
  //   phi(t) = alpha(t) * rho
  //
  // 因为 phi_dot 与 phi 共线，所以 SO(3) 右雅可比对 rho 的作用可以简化。
  //
  // body angular velocity:
  //
  //   omega_body = alpha_dot * rho
  //
  // world angular velocity:
  //
  //   omega_world = R * omega_body
  //
  const Vec3 omega_body = phase_dot * rho;
  const Vec3 omega_world = R * omega_body;

  *out_q = q;
  *out_R = R;
  *out_angular_velocity_world = omega_world;
}

// ============================================================================
// CameraTrajectory
// ============================================================================

CameraTrajectory::CameraTrajectory(
    const std::vector<FrameData<Vec3, Mat3>> &frames, float fps) {
  build(frames, fps);
}

bool CameraTrajectory::valid() const {
  return px_.valid() && py_.valid() && pz_.valid() && rotation_.valid();
}

std::size_t CameraTrajectory::size() const { return px_.size(); }

float CameraTrajectory::start_time() const { return px_.start_time(); }

float CameraTrajectory::end_time() const { return px_.end_time(); }

Eigen::Quaternionf CameraTrajectory::matrix_to_quaternion_continuous(
    const Mat3 &R, const Eigen::Quaternionf *previous) {
  Eigen::Quaternionf q(R);
  q.normalize();

  if (previous != nullptr) {
    if (q.dot(*previous) < 0.0f) {
      q.coeffs() *= -1.0f;
    }
  }

  return q;
}

Vec3 CameraTrajectory::normalized_vector_derivative(const Vec3 &v_raw,
                                                    const Vec3 &v_raw_dot) {
  const float n = v_raw.norm();

  if (n < 1e-12f) {
    return Vec3::Zero();
  }

  // u = v / |v|
  // udot = vdot / |v| - v * (v dot vdot) / |v|^3
  const float dot_v_vdot = v_raw.dot(v_raw_dot);

  return v_raw_dot / n - v_raw * (dot_v_vdot / (n * n * n));
}

void CameraTrajectory::build(const std::vector<FrameData<Vec3, Mat3>> &frames,
                             float fps) {
  fps_ = fps;

  px_.clear();
  py_.clear();
  pz_.clear();

  rotation_.clear();

  lx_.clear();
  ly_.clear();
  lz_.clear();

  light_intensity_.clear();

  if (frames.empty()) {
    return;
  }

  const std::size_t n = frames.size();

  std::vector<float> times(n, 0.0f);

  std::vector<float> xs(n, 0.0f);
  std::vector<float> ys(n, 0.0f);
  std::vector<float> zs(n, 0.0f);

  std::vector<Eigen::Quaternionf> rotations(n);

  std::vector<float> lxs(n, 0.0f);
  std::vector<float> lys(n, 0.0f);
  std::vector<float> lzs(n, 0.0f);

  std::vector<float> intensities(n, 0.0f);

  bool timestamps_strict = true;
  for (std::size_t i = 0; i < n; ++i) {
    times[i] = static_cast<float>(frames[i].frame_index) / fps_;

    if (i > 0 && !(times[i] > times[i - 1])) {
      timestamps_strict = false;
    }
  }

  // 如果输入 timestamp 不严格递增，则根据 fps 重建时间轴。
  if (!timestamps_strict) {
    if (fps_ <= 0.0f) {
      throw std::invalid_argument("CameraTrajectory::build: frame timestamps "
                                  "are not strictly increasing and fps <= 0");
    }

    for (std::size_t i = 0; i < n; ++i) {
      times[i] = static_cast<float>(i) / fps_;
    }
  }

  Eigen::Quaternionf prev_q = Eigen::Quaternionf::Identity();
  bool has_prev_q = false;

  for (std::size_t i = 0; i < n; ++i) {
    const auto &f = frames[i];

    // Camera position.
    xs[i] = f.camera_origin.x();
    ys[i] = f.camera_origin.y();
    zs[i] = f.camera_origin.z();

    // Camera orientation.
    Eigen::Quaternionf q = matrix_to_quaternion_continuous(
        f.camera_world, has_prev_q ? &prev_q : nullptr);

    q.normalize();

    prev_q = q;
    has_prev_q = true;

    rotations[i] = q;

    // Light direction.
    Vec3 ld = f.light_dir;
    const float ld_norm = ld.norm();

    if (ld_norm > 1e-12f) {
      ld /= ld_norm;
    } else {
      ld = Vec3(0.0f, 0.0f, 1.0f);
    }

    lxs[i] = ld.x();
    lys[i] = ld.y();
    lzs[i] = ld.z();

    // Light intensity.
    intensities[i] = f.light_intensity;
  }

  px_.build_auto_clamped(times, xs);
  py_.build_auto_clamped(times, ys);
  pz_.build_auto_clamped(times, zs);

  rotation_.build(times, rotations);

  lx_.build_auto_clamped(times, lxs);
  ly_.build_auto_clamped(times, lys);
  lz_.build_auto_clamped(times, lzs);

  light_intensity_.build_auto_clamped(times, intensities);
}

CameraTrajectoryState CameraTrajectory::evaluate(float t) const {
  CameraTrajectoryState out{};
  out.timestamp = t;

  if (!valid()) {
    return out;
  }

  const float tc = clamp_float(t, start_time(), end_time());
  out.timestamp = tc;

  // --------------------------------------------------------------------------
  // Camera position and linear velocity.
  // --------------------------------------------------------------------------
  const float px = px_.evaluate(tc);
  const float py = py_.evaluate(tc);
  const float pz = pz_.evaluate(tc);

  const float vx = px_.derivative(tc);
  const float vy = py_.derivative(tc);
  const float vz = pz_.derivative(tc);

  out.position = Vec3(px, py, pz);
  out.linear_velocity = Vec3(vx, vy, vz);

  // --------------------------------------------------------------------------
  // Camera orientation and angular velocity on SO(3).
  // --------------------------------------------------------------------------
  rotation_.evaluate(tc, &out.quaternion, &out.orientation,
                     &out.angular_velocity);

  // --------------------------------------------------------------------------
  // Light direction and analytic direction rate.
  // --------------------------------------------------------------------------
  Vec3 light_raw(lx_.evaluate(tc), ly_.evaluate(tc), lz_.evaluate(tc));

  Vec3 light_raw_dot(lx_.derivative(tc), ly_.derivative(tc),
                     lz_.derivative(tc));

  const float light_norm = light_raw.norm();

  if (light_norm > 1e-12f) {
    out.light_direction = light_raw / light_norm;
    out.light_direction_rate =
        normalized_vector_derivative(light_raw, light_raw_dot);
  } else {
    out.light_direction = Vec3(0.0f, 0.0f, 1.0f);
    out.light_direction_rate = Vec3::Zero();
  }

  // --------------------------------------------------------------------------
  // Light intensity and analytic intensity rate.
  // --------------------------------------------------------------------------
  out.light_intensity = light_intensity_.evaluate(tc);
  out.light_intensity_rate = light_intensity_.derivative(tc);

  return out;
}

Vec3 CameraTrajectory::position(float t) const { return evaluate(t).position; }

Mat3 CameraTrajectory::orientation(float t) const {
  return evaluate(t).orientation;
}

Eigen::Quaternionf CameraTrajectory::quaternion(float t) const {
  return evaluate(t).quaternion;
}

Vec3 CameraTrajectory::linear_velocity(float t) const {
  return evaluate(t).linear_velocity;
}

Vec3 CameraTrajectory::angular_velocity(float t) const {
  return evaluate(t).angular_velocity;
}
Vec3 CameraTrajectory::light_direction(float t) const {
  return evaluate(t).light_direction;
}

Vec3 CameraTrajectory::light_direction_rate(float t) const {
  return evaluate(t).light_direction_rate;
}

float CameraTrajectory::light_intensity(float t) const {
  return evaluate(t).light_intensity;
}

float CameraTrajectory::light_intensity_rate(float t) const {
  return evaluate(t).light_intensity_rate;
}

} // namespace dvs