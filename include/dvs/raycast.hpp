#pragma once

#include <memory>
#include <vector>

#include "dvs/scene_types.hpp"

#define CUDA_CHECK(expr)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (expr);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
        }                                                                                      \
    } while (0)
    
namespace dvs {

class GpuRaycaster {
public:
    GpuRaycaster(const std::vector<BVHNode<Vec3>>& nodes, const std::vector<Triangle<Vec3>>& triangles);
    ~GpuRaycaster();

    GpuRaycaster(GpuRaycaster&&) noexcept;
    GpuRaycaster& operator=(GpuRaycaster&&) noexcept;

    GpuRaycaster(const GpuRaycaster&) = delete;
    GpuRaycaster& operator=(const GpuRaycaster&) = delete;

    void intersect(const std::vector<Ray<Vec3>>& rays, std::vector<Intersection<Vec3>>& hits) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dvs
