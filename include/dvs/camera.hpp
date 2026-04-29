#pragma once

#include <vector>

#include "dvs/scene_types.hpp"

namespace dvs {

std::vector<Ray<Vec3>> generate_camera_rays(const FrameData<Vec3, Mat3>& frame, const SceneBundle<Vec3, Uint3, Mat3>& scene);

}  // namespace dvs
