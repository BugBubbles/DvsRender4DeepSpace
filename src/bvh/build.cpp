#include "bvh/build.h"
#include "bvh/binning.h"
#include <climits>
#include <cmath>
#include <iostream>
#include <time.h>
#include <vector>
#include <vector_types.h>

namespace dvs {

// 计算三角形的包围盒
static AABB<float3> compute_tri_aabb(const Triangle<float3> &tri) {
  AABB<float3> box;
  box.min = fminf(fminf(tri.v0, tri.v1), tri.v2);
  box.max = fmaxf(fmaxf(tri.v0, tri.v1), tri.v2);
  return box;
}

int BVHBuilder::build(const BVHBuildParams &params,
                      std::vector<BVHNode<float3>> &out_nodes,
                      std::vector<Triangle<float3>> &triangles) {
  if (triangles.empty()) {
    out_nodes.clear();
    return -1;
  }

  float start_time = (float)clock() / CLOCKS_PER_SEC;
  float end_time;

  BuildContext ctx{params, {}, {}, 0};

  ctx.refs.reserve(triangles.size());
  for (int i = 0; i < static_cast<int>(triangles.size()); ++i) {
    AABB<float3> box = compute_tri_aabb(triangles[i]);
    float3 center = aabb_center(box);
    ctx.refs.push_back({box, center, i});
  }
  end_time = (float)clock() / CLOCKS_PER_SEC;
  printf("BVH build: computed triangle bounds in %.2f seconds\n",
         end_time - start_time);
  start_time = end_time;

  ctx.nodes.reserve(2 * triangles.size());

  int root_index = build_node(ctx, 0, static_cast<int>(ctx.refs.size()), 0);
  end_time = (float)clock() / CLOCKS_PER_SEC;
  printf("BVH build: constructed BVH nodes in %.2f seconds\n",
         end_time - start_time);
  start_time = end_time;

  std::vector<Triangle<float3>> out_triangles;
  out_triangles.resize(triangles.size());
  for (int i = 0; i < static_cast<int>(ctx.refs.size()); ++i) {
    out_triangles[i] = triangles[ctx.refs[i].tri_index];
  }

  triangles = std::move(out_triangles);
  out_nodes = std::move(ctx.nodes);

  end_time = (float)clock() / CLOCKS_PER_SEC;
  printf("BVH build: reordered triangles in %.2f seconds\n",
         end_time - start_time);

  return root_index;
}

int BVHBuilder::build_node(BuildContext &ctx, int start, int end, int level) {

  int range_size = end - start;
  if (range_size <= 0) {
    throw std::runtime_error("BVH build_node empty range");
  }
  AABB<float3> range_box = aabb_empty();
  for (int i = start; i < end; ++i) {
    aabb_grow(range_box, ctx.refs[i].bounds);
  }
  // 先判断叶子大小，不要对叶子做 binning
  if (range_size <= ctx.params.max_leaf_size) {
    return create_leaf_node(ctx, start, end, range_box);
  }
  BVHRange range(range_box, start, range_size);
  BVHObjectBinning binning(range, ctx.refs.data());
  float leaf_cost = ctx.params.sah_primitive_cost * binning.leafSAH;
  float split_cost = ctx.params.sah_node_cost * aabb_safe_area(range_box) +
                     ctx.params.sah_primitive_cost * binning.splitSAH;
  if (leaf_cost <= split_cost) {
    return create_leaf_node(ctx, start, end, range_box);
  }
  // 执行分割
  BVHRange left_range, right_range;
  binning.split(range, ctx.refs.data(), left_range, right_range);
  // printf("split L=%d  R=%d\n", left_range.size, right_range.size);
  if (left_range.size <= 0 || right_range.size <= 0) {
    printf("ERROR: invalid split: range=%d left=%d right=%d start=%d end=%d\n",
           range_size, left_range.size, right_range.size, start, end);
    return create_leaf_node(ctx, start, end, range_box);
  }

  // 递归构建左右子树
  int left_idx = build_node(ctx, left_range.start,
                            left_range.start + left_range.size, level + 1);
  int right_idx = build_node(ctx, right_range.start,
                             right_range.start + right_range.size, level + 1);

  // 创建内部节点
  BVHNode<float3> node;
  node.box = range_box;
  node.left = left_idx;
  node.right = right_idx;
  node.start = 0; // 内部节点不使用
  node.count = 0;
  int cur_idx = (int)ctx.nodes.size();
  ctx.nodes.push_back(node);
  return cur_idx;
}

int BVHBuilder::create_leaf_node(BuildContext &ctx, int start, int end,
                                 const AABB<float3> &box) {
  BVHNode<float3> node;
  node.box = box;
  node.left = -1;
  node.right = -1;
  node.start = start;
  node.count = end - start;
  int idx = static_cast<int>(ctx.nodes.size());
  ctx.nodes.push_back(node);
  return idx;
}
void BVHBuilder::validate(const std::vector<BVHNode<float3>> &nodes, int root) {
  if (root < 0 || root >= static_cast<int>(nodes.size())) {
    throw std::runtime_error("BVH root index out of range");
  }

  std::vector<bool> visited(nodes.size(), false);
  std::vector<int> stack;
  stack.push_back(root);

  int visited_count = 0;
  int leaf_count = 0;
  int internal_count = 0;

  while (!stack.empty()) {
    int idx = stack.back();
    stack.pop_back();

    if (idx < 0 || idx >= static_cast<int>(nodes.size())) {
      throw std::runtime_error("BVH invalid node index");
    }

    if (visited[idx]) {
      throw std::runtime_error("BVH cycle detected");
    }

    visited[idx] = true;
    visited_count++;

    const auto &node = nodes[idx];

    if (node.count > 0) {
      leaf_count++;

      if (node.left != -1 || node.right != -1) {
        throw std::runtime_error("BVH leaf has non-negative children");
      }

      continue;
    }

    internal_count++;

    if (node.left < 0 || node.left >= static_cast<int>(nodes.size()) ||
        node.right < 0 || node.right >= static_cast<int>(nodes.size())) {
      throw std::runtime_error("BVH child index out of range");
    }

    stack.push_back(node.right);
    stack.push_back(node.left);
  }

  std::cout << "BVH validation visited nodes: " << visited_count
            << ", leaves: " << leaf_count << ", internals: " << internal_count
            << std::endl;
}
} // namespace dvs