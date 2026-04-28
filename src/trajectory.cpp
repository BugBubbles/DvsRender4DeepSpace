#include "dvs/trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace dvs {
namespace {

float bezier_point(float p0, float p1, float p2, float p3, float u) {
    const float one_minus_u = 1.0f - u;
    return one_minus_u * one_minus_u * one_minus_u * p0 +
           3.0f * one_minus_u * one_minus_u * u * p1 +
           3.0f * one_minus_u * u * u * p2 + u * u * u * p3;
}

float bezier_derivative(float p0, float p1, float p2, float p3, float u) {
    const float one_minus_u = 1.0f - u;
    return 3.0f * one_minus_u * one_minus_u * (p1 - p0) +
           6.0f * one_minus_u * u * (p2 - p1) + 3.0f * u * u * (p3 - p2);
}

float solve_bezier_param_for_x(float x_target, const BezierKeyframe& k0, const BezierKeyframe& k1) {
    const float x0 = k0.frame;
    const float x1 = k0.handle_right_frame;
    const float x2 = k1.handle_left_frame;
    const float x3 = k1.frame;
    const float span = x3 - x0;
    if (span <= 1e-6f) {
        return 0.0f;
    }

    float u = clampf((x_target - x0) / span, 0.0f, 1.0f);
    for (int i = 0; i < 8; ++i) {
        const float fx = bezier_point(x0, x1, x2, x3, u) - x_target;
        const float dfx = bezier_derivative(x0, x1, x2, x3, u);
        if (fabsf(dfx) < 1e-6f) {
            break;
        }
        const float next_u = clampf(u - fx / dfx, 0.0f, 1.0f);
        if (fabsf(next_u - u) < 1e-6f) {
            u = next_u;
            return u;
        }
        u = next_u;
    }

    float lo = 0.0f;
    float hi = 1.0f;
    for (int i = 0; i < 18; ++i) {
        const float mid = 0.5f * (lo + hi);
        const float x_mid = bezier_point(x0, x1, x2, x3, mid);
        if (x_mid < x_target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5f * (lo + hi);
}

void normalize_curve(FCurve* curve) {
    std::sort(curve->keys.begin(), curve->keys.end(), [](const BezierKeyframe& a, const BezierKeyframe& b) {
        return a.frame < b.frame;
    });
}

float evaluate_channel_with_default(const FCurve& curve, float frame, float fallback) {
    if (curve.keys.empty()) {
        return fallback;
    }
    return evaluate_fcurve_bezier(curve, frame);
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        out.push_back(field);
    }
    return out;
}

std::unordered_map<std::string, int> build_channel_map() {
    return {
        {"loc_x", 0},
        {"loc_y", 1},
        {"loc_z", 2},
        {"quat_w", 3},
        {"quat_x", 4},
        {"quat_y", 5},
        {"quat_z", 6},
    };
}

}  // namespace

float evaluate_fcurve_bezier(const FCurve& curve, float frame) {
    if (curve.keys.empty()) {
        throw std::invalid_argument("evaluate_fcurve_bezier: empty curve");
    }
    if (curve.keys.size() == 1 || frame <= curve.keys.front().frame) {
        return curve.keys.front().value;
    }
    if (frame >= curve.keys.back().frame) {
        return curve.keys.back().value;
    }

    auto it = std::upper_bound(
        curve.keys.begin(), curve.keys.end(), frame, [](float f, const BezierKeyframe& k) {
            return f < k.frame;
        });
    const std::size_t i1 = static_cast<std::size_t>(it - curve.keys.begin());
    const std::size_t i0 = i1 - 1;
    const BezierKeyframe& k0 = curve.keys[i0];
    const BezierKeyframe& k1 = curve.keys[i1];

    const float u = solve_bezier_param_for_x(frame, k0, k1);
    return bezier_point(
        k0.value, k0.handle_right_value, k1.handle_left_value, k1.value, u);
}

std::vector<PoseSample> sample_camera_pose_bezier(
    const CameraFCurves& curves,
    float start_frame,
    float end_frame,
    float step_frame) {
    if (step_frame <= 0.0f) {
        throw std::invalid_argument("step_frame must be positive");
    }
    if (end_frame < start_frame) {
        throw std::invalid_argument("end_frame must be >= start_frame");
    }

    std::vector<PoseSample> out;
    const int count = static_cast<int>(std::floor((end_frame - start_frame) / step_frame)) + 1;
    out.reserve(static_cast<std::size_t>(std::max(1, count)));

    for (float frame = start_frame; frame <= end_frame + 1e-4f; frame += step_frame) {
        PoseSample p;
        p.frame = frame;
        p.location = Vec3{
            evaluate_channel_with_default(curves.location_xyz[0], frame, 0.0f),
            evaluate_channel_with_default(curves.location_xyz[1], frame, 0.0f),
            evaluate_channel_with_default(curves.location_xyz[2], frame, 0.0f),
        };
        p.rotation_quat_wxyz = {
            evaluate_channel_with_default(curves.rotation_quat_wxyz[0], frame, 1.0f),
            evaluate_channel_with_default(curves.rotation_quat_wxyz[1], frame, 0.0f),
            evaluate_channel_with_default(curves.rotation_quat_wxyz[2], frame, 0.0f),
            evaluate_channel_with_default(curves.rotation_quat_wxyz[3], frame, 0.0f),
        };
        const float qn = std::sqrt(
            p.rotation_quat_wxyz[0] * p.rotation_quat_wxyz[0] +
            p.rotation_quat_wxyz[1] * p.rotation_quat_wxyz[1] +
            p.rotation_quat_wxyz[2] * p.rotation_quat_wxyz[2] +
            p.rotation_quat_wxyz[3] * p.rotation_quat_wxyz[3]);
        if (qn > 1e-8f) {
            for (float& q : p.rotation_quat_wxyz) {
                q /= qn;
            }
        }
        out.push_back(p);
    }

    return out;
}

bool load_camera_fcurves_csv(
    const std::filesystem::path& path,
    CameraFCurves* out_curves,
    std::string* error_message) {
    std::ifstream fin(path);
    if (!fin) {
        if (error_message != nullptr) {
            *error_message = "failed to open " + path.string();
        }
        return false;
    }

    CameraFCurves curves;
    const auto channel_map = build_channel_map();

    std::string line;
    int line_no = 0;
    while (std::getline(fin, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::vector<std::string> f = split_csv(line);
        if (f.size() != 7) {
            if (error_message != nullptr) {
                *error_message = "invalid column count in line " + std::to_string(line_no);
            }
            return false;
        }
        auto it = channel_map.find(f[0]);
        if (it == channel_map.end()) {
            if (error_message != nullptr) {
                *error_message = "unknown channel in line " + std::to_string(line_no) + ": " + f[0];
            }
            return false;
        }

        BezierKeyframe kf{};
        try {
            kf.frame = std::stof(f[1]);
            kf.value = std::stof(f[2]);
            kf.handle_left_frame = std::stof(f[3]);
            kf.handle_left_value = std::stof(f[4]);
            kf.handle_right_frame = std::stof(f[5]);
            kf.handle_right_value = std::stof(f[6]);
        } catch (const std::exception&) {
            if (error_message != nullptr) {
                *error_message = "invalid float value in line " + std::to_string(line_no);
            }
            return false;
        }

        const int channel = it->second;
        if (channel <= 2) {
            curves.location_xyz[static_cast<std::size_t>(channel)].keys.push_back(kf);
        } else {
            curves.rotation_quat_wxyz[static_cast<std::size_t>(channel - 3)].keys.push_back(kf);
        }
    }

    for (FCurve& c : curves.location_xyz) {
        normalize_curve(&c);
    }
    for (FCurve& c : curves.rotation_quat_wxyz) {
        normalize_curve(&c);
    }

    *out_curves = std::move(curves);
    return true;
}

}  // namespace dvs
