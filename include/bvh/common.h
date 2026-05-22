#pragma once

#include "dvs/scene_types.hpp"
#include "helper_math.hpp"
#include <algorithm>
#include <cfloat>

namespace dvs {

// 扩展 AABB 的辅助函数
inline AABB<float3> aabb_empty() {
  AABB<float3> box;
  box.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  box.max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  return box;
}

inline bool aabb_is_empty(const AABB<float3> &box) {
  return (box.min.x > box.max.x) | (box.min.y > box.max.y) |
         (box.min.z > box.max.z);
}

inline void aabb_grow(AABB<float3> &box, const float3 &p) {
  box.min.x = (p.x < box.min.x ? p.x : box.min.x);
  box.min.y = (p.y < box.min.y ? p.y : box.min.y);
  box.min.z = (p.z < box.min.z ? p.z : box.min.z);

  box.max.x = (p.x > box.max.x ? p.x : box.max.x);
  box.max.y = (p.y > box.max.y ? p.y : box.max.y);
  box.max.z = (p.z > box.max.z ? p.z : box.max.z);
}

inline void aabb_grow(AABB<float3> &box, const AABB<float3> &other) {
  if (aabb_is_empty(other)) {
    return;
  }

  auto xmin = std::min(other.min.x, other.max.x);
  auto ymin = std::min(other.min.y, other.max.y);
  auto zmin = std::min(other.min.z, other.max.z);

  auto xmax = std::max(other.min.x, other.max.x);
  auto ymax = std::max(other.min.y, other.max.y);
  auto zmax = std::max(other.min.z, other.max.z);

  box.min.x = std::min(box.min.x, xmin);
  box.min.y = std::min(box.min.y, ymin);
  box.min.z = std::min(box.min.z, zmin);

  box.max.x = std::max(box.max.x, xmax);
  box.max.y = std::max(box.max.y, ymax);
  box.max.z = std::max(box.max.z, zmax);
}

inline float3 aabb_center(const AABB<float3> &box) {
  return (box.min + box.max) * 0.5f;
}

inline float aabb_safe_area(const AABB<float3> &box) {
  float3 d = box.max - box.min;
  return std::max(d.x * d.y + d.y * d.z + d.z * d.x, 1e-12f);
}

// 在构建过程中使用的三角形引用
struct BVHReference {
  AABB<float3> bounds;
  float3 center;
  int tri_index; // 原始三角形数组的索引
};

// 构建范围：表示 references 数组中的一个区间
struct BVHRange {
  AABB<float3> bounds;
  int start;
  int size;

  BVHRange() : bounds(), start(0), size(0) {}
  BVHRange(const AABB<float3> &b, int s, int sz)
      : bounds(b), start(s), size(sz) {}
  int end() const { return start + size; }
};

} // namespace dvs