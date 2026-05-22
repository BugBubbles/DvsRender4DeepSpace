#pragma once

#include "bvh/common.h"
#include <cstring>

namespace dvs {

class BVHObjectBinning {
public:
  float leafSAH;
  float splitSAH;
  int dim; // 最佳分割维度（0/1/2）
  int pos; // 最佳分割位置（桶编号）

  // 从给定范围构造，计算最佳分割
  BVHObjectBinning(const BVHRange &range, BVHReference *refs);

  // 根据计算出的最佳分割，重新排列 refs 数组，并输出左右子范围
  void split(const BVHRange &range, BVHReference *refs, BVHRange &left,
             BVHRange &right) const;

private:
  enum { NUM_BINS = 32 };

  int num_bins_;             // 实际使用的桶数量
  float3 scale_;             // 将中心坐标映射到桶索引的缩放因子
  AABB<float3> cent_bounds_; // 所有三角形中心点的包围盒
};

} // namespace dvs