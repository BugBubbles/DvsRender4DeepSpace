#pragma once

#include <memory>
#include <vector>

#include "dvs/scene_types.hpp"

namespace dvs {

class GpuRaycaster {
public:
    GpuRaycaster(const std::vector<BVHNode>& nodes, const std::vector<Triangle>& triangles);
    ~GpuRaycaster();

    GpuRaycaster(GpuRaycaster&&) noexcept;
    GpuRaycaster& operator=(GpuRaycaster&&) noexcept;

    GpuRaycaster(const GpuRaycaster&) = delete;
    GpuRaycaster& operator=(const GpuRaycaster&) = delete;

    std::vector<Intersection> intersect(const std::vector<Ray>& rays) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dvs
