#pragma once

#include <vector>

#include "dvs/scene_types.hpp"

namespace dvs {

std::vector<Ray> generate_camera_rays(const FrameData& frame, const SceneBundle& scene);

}  // namespace dvs
