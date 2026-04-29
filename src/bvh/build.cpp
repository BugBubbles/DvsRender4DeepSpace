#include "bvh/build.h"
#include "bvh/binning.h"
#include <climits>
#include <cmath>
#include <time.h>

namespace dvs {

// 计算三角形的包围盒
static AABB<Vec3> compute_tri_aabb(const Triangle<Vec3>& tri) {
    AABB<Vec3> box;
    box.min = tri.v0.cwiseMin(tri.v1.cwiseMin(tri.v2));
    box.max = tri.v0.cwiseMax(tri.v1.cwiseMax(tri.v2));
    return box;
}

void BVHBuilder::build(const std::vector<Triangle<Vec3>>& triangles,
                       const BVHBuildParams& params,
                       std::vector<BVHNode<Vec3>>& out_nodes,
                       std::vector<Triangle<Vec3>>& out_triangles) {
    if (triangles.empty()) {
        out_nodes.clear();
        out_triangles.clear();
        return;
    }
    float start_time = (float)clock() / CLOCKS_PER_SEC;

    // 使用聚合初始化按声明顺序显式填充所有成员，避免 missing-field-initializers 警告
    BuildContext ctx{params,     // BVHBuildParams params
                     {},         // vector<BVHReference> refs
                     {},         // vector<BVHNode> nodes
                     0};         // int node_count

    ctx.refs.reserve(triangles.size());
    for (int i = 0; i < (int)triangles.size(); ++i) {
        ctx.refs.push_back({ compute_tri_aabb(triangles[i]), i });
    }

    ctx.nodes.reserve(2 * triangles.size());

    // 根节点索引在内部已存入 ctx.nodes，无需接收返回值，消除未使用变量警告
    build_node(ctx, 0, (int)ctx.refs.size(), 0);

    out_triangles.resize(triangles.size());
    for (int i = 0; i < (int)ctx.refs.size(); ++i) {
        out_triangles[i] = triangles[ctx.refs[i].tri_index];
    }

    out_nodes = std::move(ctx.nodes);
    float end_time = (float)clock() / CLOCKS_PER_SEC;
    printf("BVH build completed in %.2f seconds\n", end_time - start_time);
}

int BVHBuilder::build_node(BuildContext& ctx, int start, int end, int level) {
    int range_size = end - start;

    // 计算当前范围的包围盒
    AABB range_box = aabb_empty();
    for (int i = start; i < end; ++i) {
        aabb_grow(range_box, ctx.refs[i].bounds);
    }

    // 判断是否可以创建叶子节点
    BVHObjectBinning binning(BVHRange(range_box, start, range_size), ctx.refs.data());
    float leaf_cost = ctx.params.sah_primitive_cost * binning.leafSAH;
    float split_cost = ctx.params.sah_node_cost * aabb_safe_area(range_box) +
                       ctx.params.sah_primitive_cost * binning.splitSAH;

    if (range_size <= ctx.params.max_leaf_size || leaf_cost <= split_cost) {
        return create_leaf_node(ctx, start, end);
    }

    // 执行分割
    BVHRange left_range, right_range;
    binning.split(ctx.refs.data(), left_range, right_range);

    // 递归构建左右子树
    int left_idx = build_node(ctx, left_range.start, left_range.start + left_range.size, level + 1);
    int right_idx = build_node(ctx, right_range.start, right_range.start + right_range.size, level + 1);

    // 创建内部节点
    BVHNode<Vec3> node;
    node.box = range_box;
    node.left = left_idx;
    node.right = right_idx;
    node.start = 0;   // 内部节点不使用
    node.count = 0;
    int cur_idx = (int)ctx.nodes.size();
    ctx.nodes.push_back(node);
    return cur_idx;
}

int BVHBuilder::create_leaf_node(BuildContext& ctx, int start, int end) {
    BVHNode<Vec3> node;
    // 叶子包围盒
    node.box = aabb_empty();
    for (int i = start; i < end; ++i) {
        aabb_grow(node.box, ctx.refs[i].bounds);
    }
    node.left = -1;
    node.right = -1;
    node.start = start;   // 注意：此时 refs 数组仍在构建过程中，最终 start 会指向有序三角形数组的起始位置
    node.count = end - start;
    int idx = (int)ctx.nodes.size();
    ctx.nodes.push_back(node);
    return idx;
}

} // namespace dvs