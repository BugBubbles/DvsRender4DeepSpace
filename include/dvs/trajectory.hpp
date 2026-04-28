#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include "dvs/math_types.hpp"

namespace dvs {

struct BezierKeyframe {
    float frame = 0.0f;
    float value = 0.0f;
    float handle_left_frame = 0.0f;
    float handle_left_value = 0.0f;
    float handle_right_frame = 0.0f;
    float handle_right_value = 0.0f;
};

struct FCurve {
    std::vector<BezierKeyframe> keys;
};

struct CameraFCurves {
    std::array<FCurve, 3> location_xyz;
    std::array<FCurve, 4> rotation_quat_wxyz;
};

struct PoseSample {
    float frame = 0.0f;
    Vec3 location;
    std::array<float, 4> rotation_quat_wxyz{1.0f, 0.0f, 0.0f, 0.0f};
};

float evaluate_fcurve_bezier(const FCurve& curve, float frame);

std::vector<PoseSample> sample_camera_pose_bezier(
    const CameraFCurves& curves,
    float start_frame,
    float end_frame,
    float step_frame);

bool load_camera_fcurves_csv(
    const std::filesystem::path& path,
    CameraFCurves* out_curves,
    std::string* error_message);

}  // namespace dvs
