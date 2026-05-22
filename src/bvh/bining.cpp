#include "bvh/binning.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace dvs {

BVHObjectBinning::BVHObjectBinning(const BVHRange &range, BVHReference *refs)
    : leafSAH(0.0f), splitSAH(0.0f), dim(0), pos(0) {

  // 1. 计算所有引用的中心点包围盒
  AABB<float3> cent_bounds = aabb_empty();
  for (int i = 0; i < range.size; ++i) {
    const BVHReference &ref = refs[range.start + i];
    float3 center = aabb_center(ref.bounds);
    aabb_grow(cent_bounds, center);
  }
  cent_bounds_ = cent_bounds;

  // 2. 确定桶数量和缩放因子
  num_bins_ =
      (int)std::min((size_t)NUM_BINS, (size_t)(4.0f + 0.05f * range.size));
  float3 extent = cent_bounds.max - cent_bounds.min;
  // 防止除零
  extent = fmaxf(extent, 1e-12f);
  scale_ = (float)num_bins_ / extent;

  // 3. 将重心分配到桶中，统计每个桶的包围盒和进入/离开计数（此处只用包围盒）
  // 简化：不计算进入/离开，只累积每个桶内图元的总包围盒，用于后续 SAH 扫描
  struct BinInfo {
    AABB<float3> bounds;
    int count;
  };
  BinInfo bins[3][NUM_BINS]; // 三个维度分别存储
  for (int d = 0; d < 3; ++d) {
    for (int b = 0; b < NUM_BINS; ++b) {
      bins[d][b].bounds = aabb_empty();
      bins[d][b].count = 0;
    }
  }

  for (int i = 0; i < range.size; ++i) {
    const BVHReference &ref = refs[range.start + i];
    const float3 &center = ref.center;
    // 计算桶索引
    int bin[3];
    for (int d = 0; d < 3; ++d) {
      float coord = (d == 0) ? center.x : (d == 1) ? center.y : center.z;
      float minCoord = (d == 0)   ? cent_bounds.min.x
                       : (d == 1) ? cent_bounds.min.y
                                  : cent_bounds.min.z;
      int idx = (int)((coord - minCoord) *
                      (d == 0 ? scale_.x : (d == 1 ? scale_.y : scale_.z)));
      bin[d] = std::clamp(idx, 0, num_bins_ - 1);
    }
    for (int d = 0; d < 3; ++d) {
      int b = bin[d];
      aabb_grow(bins[d][b].bounds, ref.bounds);
      bins[d][b].count++;
    }
  }

  // 4. 在每个维度上进行 SAH 扫描
  leafSAH = aabb_safe_area(range.bounds) * range.size; // 近似叶子代价

  float best_cost = FLT_MAX;
  int best_dim = 0;
  int best_pos = 0;

  for (int d = 0; d < 3; ++d) {
    // 从右向左扫描，累积右侧的包围盒和计数
    AABB<float3> right_bounds[NUM_BINS];
    int right_count[NUM_BINS];
    AABB<float3> cur_right = aabb_empty();
    int cur_cnt = 0;
    for (int b = num_bins_ - 1; b >= 0; --b) {
      aabb_grow(cur_right, bins[d][b].bounds);
      cur_cnt += bins[d][b].count;
      right_bounds[b] = cur_right;
      right_count[b] = cur_cnt;
    }

    // 从左向右扫描
    AABB<float3> left_bounds = aabb_empty();
    int left_cnt = 0;
    for (int b = 1; b < num_bins_; ++b) {
      aabb_grow(left_bounds, bins[d][b - 1].bounds);
      left_cnt += bins[d][b - 1].count;
      int right_cnt = right_count[b];

      float cost = aabb_safe_area(left_bounds) * left_cnt +
                   aabb_safe_area(right_bounds[b]) * right_cnt;
      if (cost < best_cost) {
        best_cost = cost;
        best_dim = d;
        best_pos = b;
      }
    }
  }

  dim = best_dim;
  pos = best_pos;
  splitSAH = best_cost;
  // printf("[SAH] range=%d leaf=%f split=%f dim=%d pos=%d\n", range.size,
  // leafSAH,
  //        splitSAH, dim, pos);
}

void BVHObjectBinning::split(const BVHRange &range, BVHReference *refs,
                             BVHRange &left, BVHRange &right) const {
  int start = range.start;
  int end = range.start + range.size;

  int l = start;
  int r = end - 1;

  float split_val;
  if (dim == 0) {
    split_val =
        cent_bounds_.min.x +
        (float(pos) / num_bins_) * (cent_bounds_.max.x - cent_bounds_.min.x);
    while (l <= r) {
      const float3 &c = refs[l].center;
      if (c.x < split_val) {
        l++;
      } else {
        std::swap(refs[l], refs[r]);
        r--;
      }
    }
  } else if (dim == 1) {
    split_val =
        cent_bounds_.min.y +
        (float(pos) / num_bins_) * (cent_bounds_.max.y - cent_bounds_.min.y);
    while (l <= r) {
      const float3 &c = refs[l].center;
      if (c.y < split_val) {
        l++;
      } else {
        std::swap(refs[l], refs[r]);
        r--;
      }
    }
  } else if (dim == 2) {
    split_val =
        cent_bounds_.min.z +
        (float(pos) / num_bins_) * (cent_bounds_.max.z - cent_bounds_.min.z);
    while (l <= r) {
      const float3 &c = refs[l].center;
      if (c.z < split_val) {
        l++;
      } else {
        std::swap(refs[l], refs[r]);
        r--;
      }
    }
  } else {
    std::runtime_error("Invalid split dimension");
  }

  int mid = l;

  left.start = start;
  left.size = mid - start;
  left.bounds = aabb_empty();

  right.start = mid;
  right.size = end - mid;
  right.bounds = aabb_empty();

  if (left.size == 0 || right.size == 0) {
    int total = end - start;
    int mid2 = start + total / 2;

    left.start = start;
    left.size = mid2 - start;

    right.start = mid2;
    right.size = end - mid2;
  }
}

} // namespace dvs