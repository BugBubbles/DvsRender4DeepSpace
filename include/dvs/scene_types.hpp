#pragma once

#include <cstdint>
#include <vector>
#include "dvs/math_types.hpp"
#ifdef __CUDACC__
#define DVS_ALIGN(N) __align__(N)
#else
#define DVS_ALIGN(N) alignas(N)
#endif
#define DEBUG_INDEX 131328
namespace dvs {

template<typename float3, typename uint3>
struct MeshData {
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<uint3> indices;
};

template<typename float3>
struct DVS_ALIGN(4) Ray {
    float3 origin;
    float3 direction;
};

template <typename float3>
struct DVS_ALIGN(4) Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
    float3 n0;
    float3 n1;
    float3 n2;
};

template <typename float3>
struct AABB {
    float3 min;
    float3 max;
};

template <typename float3>
struct BVHNode {
    AABB<float3> box;
    int left = -1;
    int right = -1;
    int start = 0;
    int count = 0;
};

template <typename float3>
struct Intersection {
    int hit = 0;
    float t = -1.0f;
    int triangle_index = -1;
    float3 point;
    float3 normal;
};

template <typename float3, typename mat3>
struct FrameData {
    float timestamp = 0.0f;
    float3 camera_origin;
    mat3 camera_world;
    float3 light_dir;
    float light_intensity = 0.0f;
};

template <typename float3, typename uint3, typename mat3>
struct SceneBundle {
    int width = 0;
    int height = 0;
    float fx, fy, cx, cy;
    float fps = 0.0f;
    std::vector<FrameData<float3, mat3>> frames;
    std::vector<BVHNode<float3>> bvh_nodes;
    std::vector<Triangle<float3>> bvh_triangles;
};

}  // namespace dvs
