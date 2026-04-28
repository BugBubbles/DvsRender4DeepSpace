#pragma once

#include <cstdint>
#include <vector>

#include "dvs/math_types.hpp"

namespace dvs {

struct UInt3 {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;
};

struct MeshData {
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;
    std::vector<UInt3> indices;
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 n0;
    Vec3 n1;
    Vec3 n2;
};

struct AABB {
    Vec3 min;
    Vec3 max;
};

struct BVHNode {
    AABB box;
    int left = -1;
    int right = -1;
    int start = 0;
    int count = 0;
};

struct Intersection {
    int hit = 0;
    float t = -1.0f;
    int triangle_index = -1;
    Vec3 point;
    Vec3 normal;
};

struct FrameData {
    float timestamp = 0.0f;
    Vec3 camera_origin;
    Mat4 camera_world;
    Vec3 light_dir;
    float light_intensity = 0.0f;
};

struct SceneBundle {
    int width = 0;
    int height = 0;
    float fx, fy, cx, cy;
    float fps = 0.0f;
    std::vector<FrameData> frames;
    std::vector<BVHNode> bvh_nodes;
    std::vector<Triangle> bvh_triangles;
};

}  // namespace dvs
