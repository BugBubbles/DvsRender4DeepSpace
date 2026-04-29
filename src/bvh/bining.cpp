#include "bvh/binning.h"
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace dvs {

BVHObjectBinning::BVHObjectBinning(const BVHRange& range, BVHReference* refs)
    : leafSAH(0.0f), splitSAH(0.0f), dim(0), pos(0) {

    // 1. 计算所有引用的中心点包围盒
    AABB<Vec3> cent_bounds = aabb_empty();
    for (int i = 0; i < range.size; ++i) {
        const BVHReference& ref = refs[range.start + i];
        Vec3 center = aabb_center(ref.bounds);
        aabb_grow(cent_bounds, center);
    }
    this->cent_bounds = cent_bounds;

    // 2. 确定桶数量和缩放因子
    num_bins = (int)std::min((size_t)NUM_BINS, (size_t)(4.0f + 0.05f * range.size));
    Vec3 extent = cent_bounds.max - cent_bounds.min;
    // 防止除零
    extent = extent.cwiseMax(1e-12f);
    scale = ((float)num_bins / extent.array()).matrix();

    // 3. 将重心分配到桶中，统计每个桶的包围盒和进入/离开计数（此处只用包围盒）
    // 简化：不计算进入/离开，只累积每个桶内图元的总包围盒，用于后续 SAH 扫描
    struct BinInfo {
        AABB<Vec3> bounds;
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
        const BVHReference& ref = refs[range.start + i];
        Vec3 center = aabb_center(ref.bounds);
        // 计算桶索引
        int bin[3];
        for (int d = 0; d < 3; ++d) {
            float coord = (d == 0) ? center.x() : (d == 1) ? center.y() : center.z();
            float minCoord = (d == 0) ? cent_bounds.min.x() : (d == 1) ? cent_bounds.min.y() : cent_bounds.min.z();
            int idx = (int)((coord - minCoord) * (d == 0 ? scale.x() : (d == 1 ? scale.y() : scale.z())));
            bin[d] = std::clamp(idx, 0, num_bins - 1);
        }
        for (int d = 0; d < 3; ++d) {
            int b = bin[d];
            aabb_grow(bins[d][b].bounds, ref.bounds);
            bins[d][b].count++;
        }
    }

    // 4. 在每个维度上进行 SAH 扫描
    leafSAH = aabb_safe_area(range.bounds) * range.size;  // 近似叶子代价

    float best_cost = FLT_MAX;
    int best_dim = 0;
    int best_pos = 0;

    for (int d = 0; d < 3; ++d) {
        // 从右向左扫描，累积右侧的包围盒和计数
        AABB<Vec3> right_bounds[NUM_BINS];
        int right_count[NUM_BINS];
        AABB<Vec3> cur_right = aabb_empty();
        int cur_cnt = 0;
        for (int b = num_bins - 1; b >= 0; --b) {
            aabb_grow(cur_right, bins[d][b].bounds);
            cur_cnt += bins[d][b].count;
            right_bounds[b] = cur_right;
            right_count[b] = cur_cnt;
        }

        // 从左向右扫描
        AABB<Vec3> left_bounds = aabb_empty();
        int left_cnt = 0;
        for (int b = 1; b < num_bins; ++b) {
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
}

void BVHObjectBinning::split(BVHReference* refs, BVHRange& left, BVHRange& right) const {
    // 根据 dim 和 pos 重新排列引用，左部分中心坐标 < 分割位置，右部分 >=
    int start = left.start; // 调用方应传入原范围
    int end = start + left.size; // 实际上我们拿到的是同一个范围，需要重新计算 left/right
    // 这里我们假设调用方传入的 left/right 只是占位，我们内部执行分区
    // 简化：采用快速分割，所有中心小于分割点的放左边
    int l = start, r = end - 1;
    // 计算沿 dim 维度的分割平面坐标
    float split_pos = cent_bounds.min[dim] + (pos / (float)num_bins) * (cent_bounds.max[dim] - cent_bounds.min[dim]);

    while (l <= r) {
        while (l <= r && aabb_center(refs[l].bounds)[dim] < split_pos) ++l;
        while (l <= r && aabb_center(refs[r].bounds)[dim] >= split_pos) --r;
        if (l < r) {
            std::swap(refs[l], refs[r]);
        }
    }
    int left_end = l; // 左部分为 [start, left_end)
    int right_start = l;

    // 计算左右包围盒
    AABB<Vec3> lb = aabb_empty(), rb = aabb_empty();
    for (int i = start; i < left_end; ++i) aabb_grow(lb, refs[i].bounds);
    for (int i = left_end; i < end; ++i) aabb_grow(rb, refs[i].bounds);

    left = BVHRange(lb, start, left_end - start);
    right = BVHRange(rb, left_end, end - left_end);
}

} // namespace dvs