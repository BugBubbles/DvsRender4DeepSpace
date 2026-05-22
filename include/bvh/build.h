#pragma once

#include "bvh/common.h"
#include "dvs/scene_types.hpp"
#include <vector>

namespace dvs {

struct BVHBuildParams {
  int max_leaf_size = 8;
  float sah_node_cost = 1.0f;
  float sah_primitive_cost = 1.0f;
};

class BVHBuilder {
public:
  // 输入：三角形数组（原始顺序）
  // 输出：构建后的 BVH 节点数组（线性二叉树）和按 BVH 顺序重排的三角形数组
  static int build(const BVHBuildParams &params,
                   std::vector<BVHNode<Vec3>> &out_nodes,
                   std::vector<Triangle<Vec3>> &triangles);
  static void validate(const std::vector<BVHNode<Vec3>> &nodes, int root);

private:
  struct BuildContext {
    const BVHBuildParams &params;
    std::vector<BVHReference> refs;
    std::vector<BVHNode<Vec3>> nodes;
    int node_count = 0;
  };

  // 递归构建，返回新创建节点的索引
  static int build_node(BuildContext &ctx, int start, int end, int level);

  static int create_leaf_node(BuildContext &ctx, int start, int end,
                              const AABB<Vec3> &box);
};

} // namespace dvs