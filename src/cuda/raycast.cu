#include "dvs/raycast.hpp"
#include "dvs/scene_types.hpp"
#include "dvs/helper_math.h"
#include "bvh/build.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace dvs {
namespace {

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
__constant__ float kRayEps = 1e-6f;
static constexpr int kMaxStackSize = 128;          // 足够支持 6900 万三角形的深度
static constexpr int kEntryPointSentinel = 0x76543210; // 与 Blender 相同的栈底哨兵

__device__ __forceinline__ float3 bvh_inverse_direction(const float3 dir) {
    float3 inv;
    inv.x = 1.0f / (fabsf(dir.x) > 1e-20f ? dir.x : copysignf(1e-20f, dir.x));
    inv.y = 1.0f / (fabsf(dir.y) > 1e-20f ? dir.y : copysignf(1e-20f, dir.y));
    inv.z = 1.0f / (fabsf(dir.z) > 1e-20f ? dir.z : copysignf(1e-20f, dir.z));
    return inv;
}

// ---------------------------------------------------------------------------
// 核心一：同时测试内部节点两个子节点的 AABB（Blender 方式）
// 输入：射线原点 P，逆方向 idir，当前最近交点 t_max，
//       左子节点包围盒 box_left，右子节点包围盒 box_right
// 输出：dist[0] = 左子交点进入距离，dist[1] = 右子交点进入距离
// 返回：掩码 bit0=左子命中，bit1=右子命中（同 Blender 的 traverse_mask）
// ---------------------------------------------------------------------------
__device__ __forceinline__ int intersect_two_children(
    const float3 P,
    const float3 idir,
    const float t_min,
    const float t_max,
    const AABB<float3> &box_left,
    const AABB<float3> &box_right,
    float dist[2])
{
    // 左子
    float3 l_min = make_float3(box_left.min.x, box_left.min.y, box_left.min.z);
    float3 l_max = make_float3(box_left.max.x, box_left.max.y, box_left.max.z);
    float t1x0 = (l_min.x - P.x) * idir.x, t1x1 = (l_max.x - P.x) * idir.x;
    float t1y0 = (l_min.y - P.y) * idir.y, t1y1 = (l_max.y - P.y) * idir.y;
    float t1z0 = (l_min.z - P.z) * idir.z, t1z1 = (l_max.z - P.z) * idir.z;
    float t1min = fmaxf(fmaxf(fminf(t1x0, t1x1), fminf(t1y0, t1y1)), fminf(t1z0, t1z1));
    float t1max = fminf(fminf(fmaxf(t1x0, t1x1), fmaxf(t1y0, t1y1)), fmaxf(t1z0, t1z1));
    t1min = fmaxf(t_min, t1min);
    t1max = fminf(t_max, t1max);

    // 右子
    float3 r_min = make_float3(box_right.min.x, box_right.min.y, box_right.min.z);
    float3 r_max = make_float3(box_right.max.x, box_right.max.y, box_right.max.z);
    float t2x0 = (r_min.x - P.x) * idir.x, t2x1 = (r_max.x - P.x) * idir.x;
    float t2y0 = (r_min.y - P.y) * idir.y, t2y1 = (r_max.y - P.y) * idir.y;
    float t2z0 = (r_min.z - P.z) * idir.z, t2z1 = (r_max.z - P.z) * idir.z;
    float t2min = fmaxf(fmaxf(fminf(t2x0, t2x1), fminf(t2y0, t2y1)), fminf(t2z0, t2z1));
    float t2max = fminf(fminf(fmaxf(t2x0, t2x1), fmaxf(t2y0, t2y1)), fmaxf(t2z0, t2z1));
    t2min = fmaxf(t_min, t2min);
    t2max = fminf(t_max, t2max);

    dist[0] = t1min;
    dist[1] = t2min;

    int mask = 0;
    if (t1max >= t1min) mask |= 1;
    if (t2max >= t2min) mask |= 2;
    return mask;
}

// ---------------------------------------------------------------------------
// 核心二：三角形求交（沿用你原本的实现）
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool intersect_triangle(
    const Ray<float3> &ray, const Triangle<float3> &tri, float *out_t, float *out_u, float *out_v)
{
    const float3 edge1 = tri.v1 - tri.v0;
    const float3 edge2 = tri.v2 - tri.v0;
    const float3 pvec = cross(ray.direction, edge2);
    const float det = dot(edge1, pvec);
    if (fabsf(det) < kRayEps) return false;
    const float inv_det = 1.0f / det;
    const float3 tvec = ray.origin - tri.v0;
    const float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    const float3 qvec = cross(tvec, edge1);
    const float v = dot(ray.direction, qvec) * inv_det;
    if (v < 0.0f || (u + v) > 1.0f) return false;
    const float t = dot(edge2, qvec) * inv_det;
    if (t <= kRayEps) return false;
    *out_t = t;
    *out_u = u;
    *out_v = v;
    return true;
}

// ---------------------------------------------------------------------------
// 主遍历 kernel：严格按照 Blender 逻辑，适配你的 BVHNode / Triangle
// ---------------------------------------------------------------------------
__global__ void batch_intersect_kernel(
    const BVHNode<float3> *nodes,
    int node_count,
    int root_index,
    const Triangle<float3> *triangles,
    const Ray<float3> *rays,
    int ray_count,
    Intersection<float3> *out_hits)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ray_count) return;
    if (node_count <= 0) {
        out_hits[tid] = Intersection<float3>{};
        return;
    }

    // 射线准备
    Ray ray = rays[tid];
    float3 P = ray.origin;
    float3 dir = ray.direction;
    float3 idir = bvh_inverse_direction(dir);        // 预计算逆方向
    float t_min = 0.0f;
    float t_max = 1e30f;                            // 最近命中距离

    // 栈（存储候选节点索引）
    int stack[kMaxStackSize];
    stack[0] = kEntryPointSentinel;                  // 栈底哨兵
    int stack_ptr = 0;

    // 当前节点索引
    int node_addr = root_index;

    // 结果变量
    int hit_prim = -1;
    float hit_u = 0.0f, hit_v = 0.0f;

    // ---- 遍历循环 ----
    while (node_addr != kEntryPointSentinel) {

        // 遍历内部节点，直到遇到叶子
        while (node_addr >= 0 && node_addr != kEntryPointSentinel) {
            const BVHNode<float3> *node = &nodes[node_addr];
            if (node->count > 0) break;   // 叶子节点，跳出内部循环

            // 读取左、右子节点
            const BVHNode<float3> *left_child  = &nodes[node->left];
            const BVHNode<float3> *right_child = &nodes[node->right];

            float dist[2];
            int mask = intersect_two_children(P, idir, t_min, t_max,
                                              left_child->box,
                                              right_child->box,
                                              dist);

            if (mask == 3) {                       // 两个子节点都相交
                // 确保先访问较近的子节点
                int closer_idx = (dist[1] < dist[0]) ? node->right : node->left;
                int farther_idx = (dist[1] < dist[0]) ? node->left : node->right;

                // 近的子节点成为当前节点，远的压栈
                stack[++stack_ptr] = farther_idx;
                node_addr = closer_idx;
            }
            else if (mask == 1) {                  // 只有左子相交
                node_addr = node->left;
            }
            else if (mask == 2) {                  // 只有右子相交
                node_addr = node->right;
            }
            else {                                 // mask == 0，都不相交
                // 从栈中弹出下一个节点
                node_addr = stack[stack_ptr--];
            }
        }

        // ---- 处理叶子节点 ----
        if (node_addr < 0) {    // 我们的叶子节点索引不会为负，这里仅作兼容，实际用 count > 0 判断
            // Blender 中负索引表示叶子，但你的数据结构是统一的 BVHNode，叶子就在正常索引中。
            // 因此这个分支实际上不会进入，你的遍历在内部循环中直接跳到叶子索引。
        }
        else if (node_addr != kEntryPointSentinel && node_addr >= 0) {
            const BVHNode<float3> *leaf = &nodes[node_addr];
            // 叶子节点，遍历其包含的三角形
            for (int i = 0; i < leaf->count; ++i) {
                int tri_idx = leaf->start + i;
                const Triangle<float3> &tri = triangles[tri_idx];
                float t, u, v;
                if (intersect_triangle(ray, tri, &t, &u, &v)) {
                    if (t < t_max) {
                        t_max = t;
                        hit_prim = tri_idx;
                        hit_u = u;
                        hit_v = v;
                    }
                }
            }
            // 处理完叶子后，从栈中取下一个节点
            node_addr = stack[stack_ptr--];
        }
    }

    // ---- 写入结果 ----
    if (hit_prim < 0) {
        out_hits[tid] = Intersection<float3>{};
        return;
    }

    const Triangle<float3> &tri = triangles[hit_prim];
    float w = 1.0f - hit_u - hit_v;
    float3 interp_n = normalize(tri.n0 * w + tri.n1 * hit_u + tri.n2 * hit_v);
    float3 hit_point = ray.origin + ray.direction * t_max;

    Intersection<float3> out;
    out.hit = 1;
    out.t = t_max;
    out.triangle_index = hit_prim;
    out.point = hit_point;
    out.normal = interp_n;
    out_hits[tid] = out;
}


__global__ void batch_intersect_brute_force_kernel(
    const Triangle<float3>* triangles,
    int tri_count,
    const Ray<float3>* rays,
    int ray_count,
    Intersection<float3>* out_hits) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ray_count) {
        return;
    }
    if (tri_count <= 0) {
        out_hits[tid] = Intersection<float3>{};
        return;
    }
    
    const Ray ray = rays[tid];
    float closest_t = 1e30f;
    int closest_tri_idx = -1;
    float best_u = 0.0f;
    float best_v = 0.0f;

    for (int i = 0; i < tri_count; ++i) {
        float t = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (!intersect_triangle(ray, triangles[i], &t, &u, &v)) {
            continue;
        }
        if (t < closest_t) {
            closest_t = t;
            closest_tri_idx = i;
            best_u = u;
            best_v = v;
        }
    }

    if (closest_tri_idx < 0) {
        out_hits[tid] = Intersection<float3>{};
        return;
    }

    const Triangle tri = triangles[closest_tri_idx];
    const float w = 1.0f - best_u - best_v;
    const float3 interp_n = normalize(tri.n0 * w + tri.n1 * best_u + tri.n2 * best_v);
    const float3 hit_point = ray.origin + ray.direction * closest_t;

    Intersection<float3> out;
    out.hit = 1;
    out.t = closest_t;
    out.triangle_index = closest_tri_idx;
    out.point = hit_point;
    out.normal = interp_n;
    out_hits[tid] = out;
}

}  // namespace

struct GpuRaycaster::Impl {
    int node_count = 0;
    int tri_count = 0;
    int root_index = 0;
    BVHNode<float3>* d_nodes = nullptr;
    Triangle<float3>* d_tris = nullptr;
    mutable Ray<float3>* d_rays = nullptr;
    mutable Intersection<float3>* d_hits = nullptr;
    mutable std::size_t ray_capacity = 0;

    void release() {
        if (d_nodes != nullptr) {
            cudaFree(d_nodes);
            d_nodes = nullptr;
        }
        if (d_tris != nullptr) {
            cudaFree(d_tris);
            d_tris = nullptr;
        }
        if (d_rays != nullptr) {
            cudaFree(d_rays);
            d_rays = nullptr;
        }
        if (d_hits != nullptr) {
            cudaFree(d_hits);
            d_hits = nullptr;
        }
        node_count = 0;
        tri_count = 0;
        root_index = 0;
        ray_capacity = 0;
    }

    void init_with_prebuilt_bvh(
        const std::vector<BVHNode<Vec3>>& nodes, const std::vector<Triangle<Vec3>>& triangles) {
        node_count = static_cast<int>(nodes.size());
        tri_count = static_cast<int>(triangles.size());
        if (node_count <= 0 || tri_count <= 0) {
            throw std::invalid_argument("prebuilt BVH requires non-empty nodes and triangles");
        }

        root_index = 0;
        // 将Vec3格式的nodes转为float3格式的nodes
        std::vector<BVHNode<float3>> float_nodes(node_count);
        std::vector<Triangle<float3>> float_tris(tri_count);
        for (int i = 0; i < node_count; ++i) {
          const BVHNode<Vec3> &src_node = nodes[static_cast<std::size_t>(i)];
          const Triangle<Vec3> &src_tri = triangles[static_cast<std::size_t>(i)];
            BVHNode<float3> &dst_node = float_nodes[static_cast<std::size_t>(i)];
            Triangle<float3> &dst_tri = float_tris[static_cast<std::size_t>(i)];
            dst_tri.v0 = make_float3(src_tri.v0.x(), src_tri.v0.y(), src_tri.v0.z());
            dst_tri.v1 = make_float3(src_tri.v1.x(), src_tri.v1.y(), src_tri.v1.z());
            dst_tri.v2 = make_float3(src_tri.v2.x(), src_tri.v2.y(), src_tri.v2.z());
            dst_tri.n0 = make_float3(src_tri.n0.x(), src_tri.n0.y(), src_tri.n0.z());
            dst_tri.n1 = make_float3(src_tri.n1.x(), src_tri.n1.y(), src_tri.n1.z());
            dst_tri.n2 = make_float3(src_tri.n2.x(), src_tri.n2.y(), src_tri.n2.z());
            
            dst_node.box.min = make_float3(src_node.box.min.x(), src_node.box.min.y(), src_node.box.min.z());
            dst_node.box.max = make_float3(src_node.box.max.x(), src_node.box.max.y(), src_node.box.max.z());
        }
        CUDA_CHECK(cudaMalloc(&d_nodes, sizeof(BVHNode<float3>) * static_cast<std::size_t>(node_count)));
        CUDA_CHECK(cudaMalloc(&d_tris, sizeof(Triangle<float3>) * static_cast<std::size_t>(tri_count)));
        CUDA_CHECK(cudaMemcpy(
            d_nodes,
            float_nodes.data(),
            sizeof(BVHNode<float3>) * static_cast<std::size_t>(node_count),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_tris,
            float_tris.data(),
            sizeof(Triangle<float3>) * static_cast<std::size_t>(tri_count),
            cudaMemcpyHostToDevice));
    }

    // void init_with_gpu_lbvh(const std::vector<Triangle>& triangles) {
    //     tri_count = static_cast<int>(triangles.size());
    //     if (tri_count <= 0) {
    //         throw std::invalid_argument("triangles must be non-empty");
    //     }

    //     Triangle* d_input_tris = nullptr;
    //     std::uint32_t* d_codes = nullptr;
    //     int* d_sorted_indices = nullptr;
    //     int* d_parent = nullptr;
    //     int* d_ready = nullptr;

    //     CUDA_CHECK(cudaMalloc(&d_tris, sizeof(Triangle) * static_cast<std::size_t>(tri_count)));
    //     CUDA_CHECK(cudaMalloc(&d_input_tris, sizeof(Triangle) * static_cast<std::size_t>(tri_count)));
    //     CUDA_CHECK(cudaMemcpy(
    //         d_input_tris,
    //         triangles.data(),
    //         sizeof(Triangle) * static_cast<std::size_t>(tri_count),
    //         cudaMemcpyHostToDevice));

    //     if (tri_count == 1) {
    //         node_count = 1;
    //         root_index = 0;
    //         CUDA_CHECK(cudaMalloc(&d_nodes, sizeof(BVHNode)));
    //         const int threads = 256;
    //         const int blocks = 1;
    //         init_leaf_nodes_kernel<<<blocks, threads>>>(d_input_tris, tri_count, d_nodes);
    //         CUDA_CHECK(cudaGetLastError());
    //         CUDA_CHECK(cudaFree(d_input_tris));
    //         return;
    //     }

    //     node_count = tri_count * 2 - 1;
    //     CUDA_CHECK(cudaMalloc(&d_nodes, sizeof(BVHNode) * static_cast<std::size_t>(node_count)));
    //     CUDA_CHECK(cudaMalloc(&d_codes, sizeof(std::uint32_t) * static_cast<std::size_t>(tri_count)));
    //     CUDA_CHECK(cudaMalloc(&d_sorted_indices, sizeof(int) * static_cast<std::size_t>(tri_count)));
    //     CUDA_CHECK(cudaMalloc(&d_parent, sizeof(int) * static_cast<std::size_t>(node_count)));
    //     CUDA_CHECK(cudaMalloc(&d_ready, sizeof(int) * static_cast<std::size_t>(tri_count - 1)));

    //     float3 centroid_min{
    //         std::numeric_limits<float>::infinity(),
    //         std::numeric_limits<float>::infinity(),
    //         std::numeric_limits<float>::infinity()};
    //     float3 centroid_max{
    //         -std::numeric_limits<float>::infinity(),
    //         -std::numeric_limits<float>::infinity(),
    //         -std::numeric_limits<float>::infinity()};
    //     for (const Triangle& tri : triangles) {
    //         const float3 centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
    //         centroid_min = min_vec3(centroid_min, centroid);
    //         centroid_max = max_vec3(centroid_max, centroid);
    //     }

    //     const int threads = 256;
    //     const int tri_blocks = (tri_count + threads - 1) / threads;
    //     const int internal_blocks = ((tri_count - 1) + threads - 1) / threads;

    //     compute_morton_codes_kernel<<<tri_blocks, threads>>>(
    //         d_input_tris, tri_count, centroid_min, centroid_max, d_codes, d_sorted_indices);
    //     CUDA_CHECK(cudaGetLastError());

    //     thrust::sort_by_key(
    //         thrust::device,
    //         thrust::device_pointer_cast(d_codes),
    //         thrust::device_pointer_cast(d_codes + tri_count),
    //         thrust::device_pointer_cast(d_sorted_indices));

    //     reorder_triangles_kernel<<<tri_blocks, threads>>>(
    //         d_input_tris, d_sorted_indices, tri_count, d_tris);
    //     CUDA_CHECK(cudaGetLastError());

    //     init_leaf_nodes_kernel<<<tri_blocks, threads>>>(d_tris, tri_count, d_nodes);
    //     CUDA_CHECK(cudaGetLastError());

    //     CUDA_CHECK(cudaMemset(d_parent, 0xFF, sizeof(int) * static_cast<std::size_t>(node_count)));
    //     build_internal_nodes_kernel<<<internal_blocks, threads>>>(d_codes, tri_count, d_nodes, d_parent);
    //     CUDA_CHECK(cudaGetLastError());

    //     CUDA_CHECK(cudaMemset(d_ready, 0, sizeof(int) * static_cast<std::size_t>(tri_count - 1)));
    //     propagate_bounds_from_leaves_kernel<<<tri_blocks, threads>>>(
    //         tri_count, d_nodes, d_parent, d_ready);
    //     CUDA_CHECK(cudaGetLastError());

    //     std::vector<int> h_parent(static_cast<std::size_t>(node_count), -1);
    //     CUDA_CHECK(cudaMemcpy(
    //         h_parent.data(),
    //         d_parent,
    //         sizeof(int) * static_cast<std::size_t>(node_count),
    //         cudaMemcpyDeviceToHost));

    //     int detected_root = -1;
    //     for (int i = 0; i < node_count; ++i) {
    //         if (h_parent[static_cast<std::size_t>(i)] == -1) {
    //             detected_root = i;
    //             break;
    //         }
    //     }
    //     if (detected_root < 0) {
    //         throw std::runtime_error("failed to determine LBVH root");
    //     }
    //     root_index = detected_root;

    //     CUDA_CHECK(cudaFree(d_input_tris));
    //     CUDA_CHECK(cudaFree(d_codes));
    //     CUDA_CHECK(cudaFree(d_sorted_indices));
    //     CUDA_CHECK(cudaFree(d_parent));
    //     CUDA_CHECK(cudaFree(d_ready));
    // }

    Impl(const std::vector<BVHNode<Vec3>>& nodes, const std::vector<Triangle<Vec3>>& triangles) {
          init_with_prebuilt_bvh(nodes, triangles);
    }

    ~Impl() {
        release();
    }

    void ensure_ray_capacity(std::size_t ray_count) const {
        if (ray_capacity >= ray_count) {
            return;
        }
        if (d_rays != nullptr) {
            CUDA_CHECK(cudaFree(d_rays));
            d_rays = nullptr;
        }
        if (d_hits != nullptr) {
            CUDA_CHECK(cudaFree(d_hits));
            d_hits = nullptr;
        }
        CUDA_CHECK(cudaMalloc(&d_rays, sizeof(Ray<float>) * ray_count));
        CUDA_CHECK(cudaMalloc(&d_hits, sizeof(Intersection<float3>) * ray_count));
        ray_capacity = ray_count;
    }

    void intersect(const std::vector<Ray<Vec3>>& rays, std::vector<Intersection<Vec3>>& hits) const {
      if (rays.empty()) {
            hits.clear();
            return ;
        }
        ensure_ray_capacity(rays.size());
        CUDA_CHECK(cudaMemcpy(
            d_rays, rays.data(), sizeof(Ray<float3>) * rays.size(), cudaMemcpyHostToDevice));

        const int ray_count = static_cast<int>(rays.size());
        const int threads = 256;
        const int blocks = (ray_count + threads - 1) / threads;
        batch_intersect_kernel<<<blocks, threads>>>(
            d_nodes, node_count, root_index, d_tris, d_rays, ray_count,
            d_hits);
        // 采用暴力方式进行遍历，主要是为了测试 LBVH 构建的正确性
        // batch_intersect_brute_force_kernel<<<blocks, threads>>>(
        //     d_tris, tri_count, d_rays, ray_count, d_hits);
        CUDA_CHECK(cudaGetLastError());

        std::vector<Intersection<float3>> float_hits(rays.size());
        CUDA_CHECK(cudaMemcpy(float_hits.data(), d_hits,
                              sizeof(Intersection<float3>) * hits.size(),
                              cudaMemcpyDeviceToHost));
        //将float3格式的hits转为Vec3格式的hits
        hits.reserve(float_hits.size());
        for (int i = 0; i < hits.size(); ++i) {
            const Intersection<float3> &src_hit = float_hits[static_cast<std::size_t>(i)];
            Intersection<Vec3> &dst_hit = hits[static_cast<std::size_t>(i)];
            dst_hit.hit = src_hit.hit;
            dst_hit.t = src_hit.t;
            dst_hit.triangle_index = src_hit.triangle_index;
            dst_hit.point = {src_hit.point.x, src_hit.point.y, src_hit.point.z};
            dst_hit.normal = {src_hit.normal.x, src_hit.normal.y, src_hit.normal.z};
        }
    }
};

GpuRaycaster::GpuRaycaster(
    const std::vector<BVHNode<Vec3>>& nodes, const std::vector<Triangle<Vec3>>& triangles)
    : impl_(std::make_unique<Impl>(nodes, triangles)) {}

GpuRaycaster::~GpuRaycaster() = default;

GpuRaycaster::GpuRaycaster(GpuRaycaster&& other) noexcept = default;

GpuRaycaster& GpuRaycaster::operator=(GpuRaycaster&& other) noexcept = default;

void GpuRaycaster::intersect(const std::vector<Ray<Vec3>>& rays, std::vector<Intersection<Vec3>>& hits) const {
    impl_->intersect(rays, hits);
}

}  // namespace dvs
