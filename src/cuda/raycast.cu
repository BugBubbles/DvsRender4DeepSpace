#include "bvh/build.h"
#include "dvs/math_types.hpp"
#include "dvs/raycast.hpp"
#include "dvs/scene_types.hpp"

#include <fstream>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <vector_functions.h>

namespace dvs {
namespace {

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
constexpr float kRayEps = 1e-6f;
static constexpr int kMaxStackSize = 128; // 足够支持 6900 万三角形的深度
static constexpr int kEntryPointSentinel =
    0x76543210; // 与 Blender 相同的栈底哨兵

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}
inline __host__ __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
// inline __host__ __device__ float3 operator*(float b, float3 a) {
//   return make_float3(a.x * b, a.y * b, a.z * b);
// }
inline __host__ __device__ float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator/(float3 a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ float3 normalize(float3 v) {
  float len = sqrtf(dot(v, v));
  if (len > 0.0f) {
    return v / len;
  }
  return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float3 bvh_inverse_direction(const float3 dir) {
  float3 inv;
  inv.x = 1.0f / (fabsf(dir.x) > 1e-20f ? dir.x : copysignf(1e-20f, dir.x));
  inv.y = 1.0f / (fabsf(dir.y) > 1e-20f ? dir.y : copysignf(1e-20f, dir.y));
  inv.z = 1.0f / (fabsf(dir.z) > 1e-20f ? dir.z : copysignf(1e-20f, dir.z));
  return inv;
}

float3 bvh_inverse_direction_cpu(const float3 dir) {
  float3 inv;
  inv.x = 1.0f /
          (std::fabs(dir.x) > 1e-20f ? dir.x : std::copysignf(1e-20f, dir.x));
  inv.y = 1.0f /
          (std::fabs(dir.y) > 1e-20f ? dir.y : std::copysignf(1e-20f, dir.y));
  inv.z = 1.0f /
          (std::fabs(dir.z) > 1e-20f ? dir.z : std::copysignf(1e-20f, dir.z));
  return inv;
}

// ---------------------------------------------------------------------------
// 核心一：同时测试内部节点两个子节点的 AABB（Blender 方式）
// 输入：射线原点 P，逆方向 idir，当前最近交点 t_max，
//       左子节点包围盒 box_left，右子节点包围盒 box_right
// 输出：dist[0] = 左子交点进入距离，dist[1] = 右子交点进入距离
// 返回：掩码 bit0=左子命中，bit1=右子命中（同 Blender 的 traverse_mask）
// ---------------------------------------------------------------------------
__device__ __forceinline__ int
intersect_two_children(const float3 P, const float3 idir, const float t_min,
                       const float t_max, const AABB<float3> &box_left,
                       const AABB<float3> &box_right, float *dist) {
  // 左子
  float3 l_min = make_float3(box_left.min.x, box_left.min.y, box_left.min.z);
  float3 l_max = make_float3(box_left.max.x, box_left.max.y, box_left.max.z);
  float t1x0 = (l_min.x - P.x) * idir.x, t1x1 = (l_max.x - P.x) * idir.x;
  float t1y0 = (l_min.y - P.y) * idir.y, t1y1 = (l_max.y - P.y) * idir.y;
  float t1z0 = (l_min.z - P.z) * idir.z, t1z1 = (l_max.z - P.z) * idir.z;
  float t1min =
      fmaxf(fmaxf(fminf(t1x0, t1x1), fminf(t1y0, t1y1)), fminf(t1z0, t1z1));
  float t1max =
      fminf(fminf(fmaxf(t1x0, t1x1), fmaxf(t1y0, t1y1)), fmaxf(t1z0, t1z1));
  t1min = fmaxf(t_min, t1min);
  t1max = fminf(t_max, t1max);

  // 右子
  float3 r_min = make_float3(box_right.min.x, box_right.min.y, box_right.min.z);
  float3 r_max = make_float3(box_right.max.x, box_right.max.y, box_right.max.z);
  float t2x0 = (r_min.x - P.x) * idir.x, t2x1 = (r_max.x - P.x) * idir.x;
  float t2y0 = (r_min.y - P.y) * idir.y, t2y1 = (r_max.y - P.y) * idir.y;
  float t2z0 = (r_min.z - P.z) * idir.z, t2z1 = (r_max.z - P.z) * idir.z;
  float t2min =
      fmaxf(fmaxf(fminf(t2x0, t2x1), fminf(t2y0, t2y1)), fminf(t2z0, t2z1));
  float t2max =
      fminf(fminf(fmaxf(t2x0, t2x1), fmaxf(t2y0, t2y1)), fmaxf(t2z0, t2z1));
  t2min = fmaxf(t_min, t2min);
  t2max = fminf(t_max, t2max);

  dist[0] = t1min;
  dist[1] = t2min;

  int mask = 0;
  if (t1max >= t1min)
    mask |= 1;
  if (t2max >= t2min)
    mask |= 2;
  return mask;
}

int intersect_two_children_cpu(const float3 P, const float3 idir,
                               const float t_min, const float t_max,
                               const AABB<float3> &box_left,
                               const AABB<float3> &box_right, float *dist) {
  // 左子
  float3 l_min = make_float3(box_left.min.x, box_left.min.y, box_left.min.z);
  float3 l_max = make_float3(box_left.max.x, box_left.max.y, box_left.max.z);
  float t1x0 = (l_min.x - P.x) * idir.x, t1x1 = (l_max.x - P.x) * idir.x;
  float t1y0 = (l_min.y - P.y) * idir.y, t1y1 = (l_max.y - P.y) * idir.y;
  float t1z0 = (l_min.z - P.z) * idir.z, t1z1 = (l_max.z - P.z) * idir.z;
  float t1min =
      std::fmax(std::fmax(std::fmin(t1x0, t1x1), std::fmin(t1y0, t1y1)),
                std::fmin(t1z0, t1z1));
  float t1max =
      std::fmin(std::fmin(std::fmax(t1x0, t1x1), std::fmax(t1y0, t1y1)),
                std ::fmax(t1z0, t1z1));
  t1min = std::fmax(t_min, t1min);
  t1max = std::fmin(t_max, t1max);

  // 右子
  float3 r_min = make_float3(box_right.min.x, box_right.min.y, box_right.min.z);
  float3 r_max = make_float3(box_right.max.x, box_right.max.y, box_right.max.z);
  float t2x0 = (r_min.x - P.x) * idir.x, t2x1 = (r_max.x - P.x) * idir.x;
  float t2y0 = (r_min.y - P.y) * idir.y, t2y1 = (r_max.y - P.y) * idir.y;
  float t2z0 = (r_min.z - P.z) * idir.z, t2z1 = (r_max.z - P.z) * idir.z;
  float t2min =
      std::fmax(std::fmax(std::fmin(t2x0, t2x1), std::fmin(t2y0, t2y1)),
                std::fmin(t2z0, t2z1));
  float t2max =
      std::fmin(std::fmin(std::fmax(t2x0, t2x1), std::fmax(t2y0, t2y1)),
                std::fmax(t2z0, t2z1));
  t2min = std::fmax(t_min, t2min);
  t2max = std::fmin(t_max, t2max);

  dist[0] = t1min;
  dist[1] = t2min;

  int mask = 0;
  if (t1max >= t1min)
    mask |= 1;
  if (t2max >= t2min)
    mask |= 2;
  return mask;
}

// ---------------------------------------------------------------------------
// 核心二：三角形求交（沿用你原本的实现）
// ---------------------------------------------------------------------------
/**
 * @brief 三角形求交，Blender 方式（Möller–Trumbore 算法）
 * @param origin 射线原点，都是 float3 类型
 * @param direction 射线方向（不必归一化）
 * @param v0,v1,v2 三角形顶点
 * @param n0,n1,n2 三角形顶点法线（用于插值计算交点法线）
 */
__device__ bool intersect_triangle(const Ray<float3> &ray,
                                   const Triangle<float3> &tri, float *out_t,
                                   float *out_u, float *out_v) {
  const float3 edge1 = tri.v1 - tri.v0;
  const float3 edge2 = tri.v2 - tri.v0;
  const float3 pvec = cross(ray.direction, edge2);
  const float det = dot(edge1, pvec);
  if (fabsf(det) < kRayEps)
    return false;
  const float inv_det = 1.0f / det;
  const float3 tvec = ray.origin - tri.v0;
  const float u = dot(tvec, pvec) * inv_det;
  if (u < 0.0f || u > 1.0f)
    return false;
  const float3 qvec = cross(tvec, edge1);
  const float v = dot(ray.direction, qvec) * inv_det;
  if (v < 0.0f || (u + v) > 1.0f)
    return false;
  const float t = dot(edge2, qvec) * inv_det;
  if (t <= kRayEps)
    return false;
  *out_t = t;
  *out_u = u;
  *out_v = v;
  return true;
}
bool intersect_triangle_cpu(const Ray<float3> &ray, const Triangle<float3> &tri,
                            float *out_t, float *out_u, float *out_v) {

  const float3 edge1 = tri.v1 - tri.v0;
  const float3 edge2 = tri.v2 - tri.v0;
  const float3 pvec = cross(ray.direction, edge2);
  const float det = dot(edge1, pvec);
  if (std::fabs(det) < kRayEps)
    return false;
  const float inv_det = 1.0f / det;
  const float3 tvec = ray.origin - tri.v0;
  const float u = dot(tvec, pvec) * inv_det;
  if (u < 0.0f || u > 1.0f)
    return false;
  const float3 qvec = cross(tvec, edge1);
  const float v = dot(ray.direction, qvec) * inv_det;
  if (v < 0.0f || (u + v) > 1.0f)
    return false;
  const float t = dot(edge2, qvec) * inv_det;
  if (t <= kRayEps)
    return false;
  *out_t = t;
  *out_u = u;
  *out_v = v;
  return true;
}

// ---------------------------------------------------------------------------
// 主遍历 kernel：严格按照 Blender 逻辑，适配你的 BVHNode / Triangle
// ---------------------------------------------------------------------------
__global__ void batch_intersect_kernel(const BVHNode<float3> *nodes,
                                       int root_index,
                                       const Triangle<float3> *triangles,
                                       const Ray<float3> *rays, int ray_count,
                                       Intersection<float3> *out_hits) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ray_count)
    return;

  Ray<float3> ray = rays[idx];
  float3 idir = bvh_inverse_direction(ray.direction);

  Intersection<float3> hit;
  hit.hit = 0;
  hit.t = 1e30f;
  hit.triangle_index = -1;

  float t_max = hit.t;

  int stack[kMaxStackSize];
  int sp = 0;
  stack[sp++] = root_index;

  while (sp > 0) {
    int ni = stack[--sp];
    const BVHNode<float3> &node = nodes[ni];

    if (node.count > 0) {

      // Leaf
      for (int i = 0; i < node.count; i++) {
        int tri_id = node.start + i;
        float t, u, v;

        if (intersect_triangle(ray, triangles[tri_id], &t, &u, &v)) {
          if (t < hit.t) {
            hit.hit = 1;
            hit.t = t;
            t_max = t;
            hit.triangle_index = tri_id;
            const float w = 1 - u - v;
            hit.point = ray.origin + ray.direction * t;
            hit.normal =
                normalize(triangles[tri_id].n0 * w + triangles[tri_id].n1 * u +
                          triangles[tri_id].n2 * v);
          }
        }
      }
      continue;
    }

    // Internal node
    float dist[2];
    int mask = intersect_two_children(ray.origin, idir, kRayEps, t_max,
                                      nodes[node.left].box,
                                      nodes[node.right].box, dist);

    if (mask == 0)
      continue;

    if (mask == 3) {
      if (dist[0] < dist[1]) {
        stack[sp++] = node.right;
        stack[sp++] = node.left;
      } else {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    } else if (mask == 1) {
      stack[sp++] = node.left;
    } else {
      stack[sp++] = node.right;
    }

    if (sp >= kMaxStackSize) {
      printf("Warning: stack overflow, aborting ray.\n");
      break;
    }
  }

  out_hits[idx] = hit;
}

Intersection<float3> intersect_single_ray_cpu(const BVHNode<float3> *nodes,
                                              int root_index,
                                              const Triangle<float3> *triangles,
                                              const Ray<float3> &ray,
                                              const int node_count) {
  // ---- 初始化 ----
  float3 idir = bvh_inverse_direction_cpu(ray.direction);

  Intersection<float3> hit;
  hit.hit = 0;
  hit.t = 1e30f;
  hit.triangle_index = -1;

  float t_max = hit.t;

  int stack[kMaxStackSize];
  int sp = 0;

  stack[sp++] = root_index;

  printf("CPU Test: root node has %d triangles\n", nodes[root_index].count);

  // ---- BVH 遍历 ----
  while (sp > 0) {
    int ni = stack[--sp];
    const BVHNode<float3> &node = nodes[ni];

    if (node.count > 0) {
      // ========= 叶子节点 — 三角面求交 =========
      for (int i = 0; i < node.count; i++) {
        int tri_id = node.start + i;
        float t, u, v;

        if (intersect_triangle_cpu(ray, triangles[tri_id], &t, &u, &v)) {
          if (t < hit.t) {
            hit.hit = 1;
            hit.t = t;
            t_max = t; // 重要：更新裁剪距离
            hit.triangle_index = tri_id;

            // 如果需要也可计算交点与法线：
            const float w = 1 - u - v;
            hit.point = ray.origin + ray.direction * t;
            hit.normal =
                normalize(triangles[tri_id].n0 * w + triangles[tri_id].n1 * u +
                          triangles[tri_id].n2 * v);
          }
        }
      }
      continue; // 继续处理栈
    }
    std::cout << "CPU Test: internal node with left=" << node.left
              << " right=" << node.right << std::endl;

    // ========= 内部节点 — AABB 检测 =========
    float dist[2];
    int mask = intersect_two_children_cpu(ray.origin, idir, kRayEps, t_max,
                                          nodes[node.left].box,
                                          nodes[node.right].box, dist);

    if (mask == 0)
      continue;

    // Blender-style order: near first
    if (mask == 3) {
      if (dist[0] < dist[1]) {
        stack[sp++] = node.right;
        stack[sp++] = node.left;
      } else {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    } else if (mask == 1) {
      stack[sp++] = node.left;
    } else {
      stack[sp++] = node.right;
    }

    if (sp >= kMaxStackSize) {
      printf("Warning: stack overflow, aborting ray.\n");
      break;
    }
  }
  return hit;
}
__global__ void view_direction_batch_kernel(const int width, const int height,
                                            const float fx, const float fy,
                                            const float cx, const float cy,
                                            const float *camera_world,
                                            const float3 origin,
                                            Ray<float3> *out_rays) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  int idx = y * width + x;
  // 计算像素中心的视线方向
  float px = (x + 0.5f - cx) / fx;
  float py = (y + 0.5f - cy) / fy;
  float3 dir_camera = make_float3(px, py, -1.0f); // 摄像机空间的视线方向
  // 转换到世界空间
  float3 dir_world = make_float3(
      camera_world[0] * dir_camera.x + camera_world[3] * dir_camera.y +
          camera_world[6] * dir_camera.z,
      camera_world[1] * dir_camera.x + camera_world[4] * dir_camera.y +
          camera_world[7] * dir_camera.z,
      camera_world[2] * dir_camera.x + camera_world[5] * dir_camera.y +
          camera_world[8] * dir_camera.z);

  dir_world = normalize(dir_world);
  out_rays[idx].origin = origin;
  out_rays[idx].direction = dir_world;
}
__global__ void intersect_brute_batch_kernel(const int width, const int height,
                                             int tri_count, int node_count,
                                             const BVHNode<float3> *d_nodes,
                                             const Triangle<float3> *d_tris,
                                             const Ray<float3> *d_rays,
                                             Intersection<float3> *out_hits) {
  // 这里是你之前实现的核心遍历逻辑，直接复制过来即可
  // 每个视线都执行一次暴力求交，验证intersect_triangle的正确性
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = y * width + x;
  if (x >= width || y >= height) {
    return;
  }
  if (tri_count <= 0) {
    out_hits[tid] = Intersection<float3>{};
    return;
  }
  if (node_count <= 0) {
    out_hits[tid] = Intersection<float3>{};
    return;
  }

  const Ray<float3> ray = d_rays[tid];
  float closest_t = 1e30f;
  int closest_tri_idx = -1;
  float best_u = 0.0f;
  float best_v = 0.0f;
  for (int i = 0; i < tri_count; ++i) {
    float t = 0.0f;
    float u = 0.0f;
    float v = 0.0f;
    if (!intersect_triangle(ray, d_tris[i], &t, &u, &v)) {
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

  const auto tri = d_tris[closest_tri_idx];
  const float w = 1.0f - best_u - best_v;
  const float3 interp_n =
      normalize(tri.n0 * w + tri.n1 * best_u + tri.n2 * best_v);
  const float3 hit_point = ray.origin + ray.direction * closest_t;
  Intersection<float3> out;
  out.hit = 1;
  out.t = closest_t;
  out.triangle_index = closest_tri_idx;
  out.point = hit_point;
  out.normal = interp_n;
  out_hits[tid] = out;
}

} // namespace

GpuRaycaster::GpuRaycaster(const std::vector<BVHNode<Vec3>> &nodes,
                           const std::vector<Triangle<Vec3>> &triangles,
                           const int root_index)
    : root_index_(root_index) {
  node_count_ = static_cast<int>(nodes.size());
  tri_count_ = static_cast<int>(triangles.size());
  if (node_count_ <= 0 || tri_count_ <= 0) {
    throw std::invalid_argument(
        "prebuilt BVH requires non-empty nodes and triangles");
  }

  std::cout << "BVH root_index_ = " << root_index_
            << ", root count = " << nodes[root_index_].count
            << ", left = " << nodes[root_index_].left
            << ", right = " << nodes[root_index_].right << std::endl;
  validate_bvh(nodes, root_index_);
  std::cout << "BVH validated: " << node_count_ << " nodes, " << tri_count_
            << " triangles." << std::endl;
  // 将Vec3格式的nodes转为float3格式的nodes
  std::vector<BVHNode<float3>> float_nodes(node_count_);
  std::vector<Triangle<float3>> float_tris(tri_count_);
  for (int i = 0; i < tri_count_; ++i) {
    const auto &src_tri = triangles[static_cast<size_t>(i)];
    auto &dst_tri = float_tris[static_cast<size_t>(i)];
    dst_tri.v0 = make_float3(src_tri.v0.x(), src_tri.v0.y(), src_tri.v0.z());
    dst_tri.v1 = make_float3(src_tri.v1.x(), src_tri.v1.y(), src_tri.v1.z());
    dst_tri.v2 = make_float3(src_tri.v2.x(), src_tri.v2.y(), src_tri.v2.z());
    dst_tri.n0 = make_float3(src_tri.n0.x(), src_tri.n0.y(), src_tri.n0.z());
    dst_tri.n1 = make_float3(src_tri.n1.x(), src_tri.n1.y(), src_tri.n1.z());
    dst_tri.n2 = make_float3(src_tri.n2.x(), src_tri.n2.y(), src_tri.n2.z());
  }
  for (int i = 0; i < node_count_; ++i) {
    const auto &src_node = nodes[static_cast<size_t>(i)];
    auto &dst_node = float_nodes[static_cast<size_t>(i)];
    dst_node.box.min = make_float3(src_node.box.min.x(), src_node.box.min.y(),
                                   src_node.box.min.z());
    dst_node.box.max = make_float3(src_node.box.max.x(), src_node.box.max.y(),
                                   src_node.box.max.z());
    dst_node.left = src_node.left;
    dst_node.right = src_node.right;
    dst_node.start = src_node.start;
    dst_node.count = src_node.count;
  }
  CUDA_CHECK(cudaMalloc(&d_nodes_, sizeof(BVHNode<float3>) *
                                       static_cast<size_t>(node_count_)));
  CUDA_CHECK(cudaMalloc(&d_tris_, sizeof(Triangle<float3>) *
                                      static_cast<size_t>(tri_count_)));
  CUDA_CHECK(
      cudaMemcpy(d_nodes_, float_nodes.data(),
                 sizeof(BVHNode<float3>) * static_cast<size_t>(node_count_),
                 cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_tris_, float_tris.data(),
                 sizeof(Triangle<float3>) * static_cast<size_t>(tri_count_),
                 cudaMemcpyHostToDevice));
}
void GpuRaycaster::validate_bvh(const std::vector<BVHNode<Vec3>> &nodes,
                                int root) const {
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
void GpuRaycaster::intersect(const SceneBundle<Vec3, Uint3, Mat3> &scene,
                             int frame_idx,
                             std::vector<Intersection<Vec3>> &hits) {
  int width = scene.width;
  int height = scene.height;
  float fx = scene.fx;
  float fy = scene.fy;
  float cx = scene.cx;
  float cy = scene.cy;
  auto err = cudaGetLastError();

  auto &frame = scene.frames[frame_idx];
  float3 origin = make_float3(frame.camera_origin.x(), frame.camera_origin.y(),
                              frame.camera_origin.z());
  float *d_camera_world;
  CUDA_CHECK(cudaMalloc(&d_camera_world, sizeof(float) * 9));
  CUDA_CHECK(cudaMemcpy(d_camera_world, frame.camera_world.data(),
                        sizeof(float) * 9, cudaMemcpyHostToDevice));

  ensure_ray_capacity(static_cast<std::size_t>(width * height));
  // 生成视线
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  view_direction_batch_kernel<<<grid_size, block_size>>>(
      width, height, fx, fy, cx, cy, d_camera_world, origin, d_rays_);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Kernel error = " << cudaGetErrorString(err) << "\n";
  // 进行批量求交
  intersect_brute_batch_kernel<<<grid_size, block_size>>>(
      width, height, tri_count_, node_count_, d_nodes_, d_tris_, d_rays_,
      d_hits_);
  // batch_intersect_kernel<<<grid_size, block_size>>>(
  //     d_nodes_, root_index_, d_tris_, d_rays_, width * height, d_hits_);
  // 测试单测，只计算256，256这个像素的交点，验证intersect_triangle的正确性
  // Ray<float3> test_ray;
  // test_ray.origin = make_float3(21.7763, -32185.2109, 7938.9810);
  // test_ray.direction = make_float3(-0.0104, 0.1778, -0.9840);

  // BVHNode<float3> *test_nodes = new BVHNode<float3>[node_count_];
  // Triangle<float3> *test_tris = new Triangle<float3>[tri_count_];
  // CUDA_CHECK(
  //     cudaMemcpy(test_nodes, d_nodes_,
  //                sizeof(BVHNode<float3>) * static_cast<size_t>(node_count_),
  //                cudaMemcpyDeviceToHost));
  // CUDA_CHECK(
  //     cudaMemcpy(test_tris, d_tris_,
  //                sizeof(Triangle<float3>) * static_cast<size_t>(tri_count_),
  //                cudaMemcpyDeviceToHost));

  // Intersection<float3> hit = intersect_single_ray_cpu(
  //     test_nodes, root_index_, test_tris, test_ray, node_count_);
  // std::cout << "CPU single ray test: hit=" << hit.hit << " t=" << hit.t
  //           << " tri_idx=" << hit.triangle_index << " point=(" << hit.point.x
  //           << "," << hit.point.y << "," << hit.point.z << ") normal=("
  //           << hit.normal.x << "," << hit.normal.y << "," << hit.normal.z
  //           << ")\n";
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  std::cout << "Kernel error = " << cudaGetErrorString(err) << "\n";
  // 检查求交结果
  hits.resize(static_cast<std::size_t>(width * height));
  CUDA_CHECK(cudaMemcpy(hits.data(), d_hits_,
                        sizeof(Intersection<float3>) * width * height,
                        cudaMemcpyDeviceToHost));
  cudaFree(d_camera_world);
  // 将所有的光线、交点等信息打印出来，验证是否正确
  std::ofstream ofs("debug_output.txt");
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x !=256 || y != 256) {
        continue; // 只打印256,256这个像素的结果
      }
      int idx = y * width + x;
      ofs << "Pixel (" << x << "," << y << "): "
          << "hit=" << hits[idx].hit << " t=" << hits[idx].t
          << " tri_idx=" << hits[idx].triangle_index << " point=("
          << hits[idx].point.x() << "," << hits[idx].point.y() << ","
          << hits[idx].point.z() << ")"
          << " normal=(" << hits[idx].normal.x() << "," << hits[idx].normal.y()
          << "," << hits[idx].normal.z() << ")" << std::endl;

      ofs.flush();
    }
  }
}

} // namespace dvs
