#pragma once

#include "dvs/scene_types.hpp"
#include <algorithm>
#include <cfloat>

namespace dvs {


// 扩展 AABB 的辅助函数
inline AABB<Vec3> aabb_empty() {
    AABB<Vec3> box;
    box.min = Vec3( FLT_MAX,  FLT_MAX,  FLT_MAX );
    box.max = Vec3( -FLT_MAX, -FLT_MAX, -FLT_MAX );
    return box;
}

inline bool aabb_is_empty(const AABB<Vec3>& box) {
    return box.min.x() > box.max.x() ||
           box.min.y() > box.max.y() ||
           box.min.z() > box.max.z();
}

inline void aabb_grow(AABB<Eigen::Vector3f>& box, const Eigen::Vector3f& p) {
    box.min = box.min.cwiseMin(p);
    box.max = box.max.cwiseMax(p);
}

inline void aabb_grow(AABB<Vec3>& box, const AABB<Vec3>& other) {
    if (aabb_is_empty(other)) {
        return;
    }

    aabb_grow(box, other.min);
    aabb_grow(box, other.max);
}

inline Vec3 aabb_center(const AABB<Vec3>& box) {
    return (box.min + box.max) * 0.5f;
}

inline float aabb_half_area(const AABB<Vec3>& box) {
    Vec3 d = box.max - box.min;
    return (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
}

inline float aabb_safe_area(const AABB<Vec3>& box) {
    return std::max(aabb_half_area(box), 1e-12f);
}

// 在构建过程中使用的三角形引用
struct BVHReference {
    AABB<Vec3> bounds;
    Vec3 center;
    int tri_index;   // 原始三角形数组的索引
};

// 构建范围：表示 references 数组中的一个区间
struct BVHRange {
    AABB<Vec3> bounds;
    int start;
    int size;

    BVHRange() : bounds(), start(0), size(0) {}
    BVHRange(const AABB<Vec3>& b, int s, int sz) : bounds(b), start(s), size(sz) {}
    int end() const { return start + size; }
};

} // namespace dvs