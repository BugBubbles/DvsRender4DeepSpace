#include "bvh/build.h"
#include "dvs/raycast.hpp"
#include "dvs/scene_types.hpp"

#include <cstdio>
#include <fstream>
#include <limits>
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
inline __host__ __device__ float3 operator*(float b, float3 a) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
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
                       const AABB<float3> &box_right, float dist[2]) {
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
// __device__ bool intersect_triangle(const float3 origin,
//                                    const float3 direction,
//                                    const float3 v0, const float3 v1, const
//                                    float3 v2, const float3 n0, const float3
//                                    n1, const float3 n2,
//                                     float *out_t, float *out_u, float *out_v)
// {
//     const float3 edge1 = v1 - v0;
//     const float3 edge2 = v2 - v0;
//     const float3 pvec = cross(direction, edge2);
//     const float det = dot(edge1, pvec);
//     if (fabsf(det) < kRayEps) return false;
//     const float inv_det = 1.0f / det;
//     const float3 tvec = origin - v0;
//     const float u = dot(tvec, pvec) * inv_det;
//     if (u < 0.0f || u > 1.0f) return false;
//     const float3 qvec = cross(tvec, edge1);
//     const float v = dot(direction, qvec) * inv_det;
//     if (v < 0.0f || (u + v) > 1.0f) return false;
//     const float t = dot(edge2, qvec) * inv_det;
//     if (t <= kRayEps) return false;
//     *out_t = t;
//     *out_u = u;
//     *out_v = v;
//     return true;
// }

// ---------------------------------------------------------------------------
// 主遍历 kernel：严格按照 Blender 逻辑，适配你的 BVHNode / Triangle
// ---------------------------------------------------------------------------
// __global__ void batch_intersect_kernel(
//     const BVHNode<float3> *nodes,
//     int node_count,
//     int root_index,
//     const Triangle<float3> *triangles,
//     const Ray<float3> *rays,
//     int ray_count,
//     Intersection<float3> *out_hits)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= ray_count) return;
//     if (node_count <= 0) {
//         out_hits[tid] = Intersection<float3>{};
//         return;
//     }

//     // 防御：最大迭代次数（防止死循环）
//     const int max_iterations = node_count * 2 + 1000;
//     int iter_count = 0;
//     // 射线准备
//     Ray ray = rays[tid];
//     float3 P = ray.origin;
//     float3 dir = ray.direction;
//     float3 idir = bvh_inverse_direction(dir);        // 预计算逆方向
//     float t_min = 0.0f;
//     float t_max = 1e30f;                            // 最近命中距离

//     // 栈（存储候选节点索引）
//     int stack[kMaxStackSize];
//     stack[0] = kEntryPointSentinel;                  // 栈底哨兵
//     int stack_ptr = 0;

//     // 当前节点索引
//     int node_addr = root_index;

//     // 结果变量
//     int hit_prim = -1;
//     float hit_u = 0.0f, hit_v = 0.0f;

//     // ---- 遍历循环 ----
//     while (node_addr != kEntryPointSentinel) {
//         if (++iter_count > max_iterations) {
//             // 直接退出并标记错误（可通过输出全局内存标记，或printf）
//             out_hits[tid] = Intersection<float3>{}; // 或返回空
//             return;
//         }

//         // 内部节点遍历
//         while (node_addr >= 0 && node_addr != kEntryPointSentinel) {
//             const BVHNode<float3> *node = &nodes[node_addr];
//             if (node->count > 0) break;   // 叶子

//             // 检查子索引合法性
//             if (node->left < 0 || node->left >= node_count ||
//                 node->right < 0 || node->right >= node_count) {
//                 out_hits[tid] = Intersection<float3>{};
//                 return; // 索引无效，退出
//             }

//             const BVHNode<float3> *left_child  = &nodes[node->left];
//             const BVHNode<float3> *right_child = &nodes[node->right];
//             float dist[2];
//             int mask = intersect_two_children(P, idir, t_min, t_max,
//                                               left_child->box,
//                                               right_child->box,
//                                               dist);

//             if (mask == 3) {
//                 int closer_idx = (dist[1] < dist[0]) ? node->right :
//                 node->left; int farther_idx = (dist[1] < dist[0]) ?
//                 node->left : node->right;
//                 // 防止栈溢出
//                 if (stack_ptr + 1 >= kMaxStackSize) {
//                     out_hits[tid] = Intersection<float3>{};
//                     return;
//                 }
//                 stack[++stack_ptr] = farther_idx;
//                 node_addr = closer_idx;
//             }
//             else if (mask == 1) node_addr = node->left;
//             else if (mask == 2) node_addr = node->right;
//             else {
//                 if (stack_ptr < 0) { out_hits[tid] = Intersection<float3>{};
//                 return; } node_addr = stack[stack_ptr--];
//             }
//         }

//         // 叶子处理
//         if (node_addr >= 0 && node_addr != kEntryPointSentinel) {
//             const BVHNode<float3> *leaf = &nodes[node_addr];
//             for (int i = 0; i < leaf->count; ++i) {
//                 int tri_idx = leaf->start + i;
//                 const Triangle<float3> &tri = triangles[tri_idx];
//                 float t, u, v;
//                 if (intersect_triangle(ray, tri, &t, &u, &v)) {
//                     if (t < t_max) {
//                         t_max = t;
//                         hit_prim = tri_idx;
//                         hit_u = u;
//                         hit_v = v;
//                     }
//                 }
//             }
//             if (stack_ptr < 0) { out_hits[tid] = Intersection<float3>{};
//             return; } node_addr = stack[stack_ptr--];
//         }
//     }

//     if (iter_count > max_iterations) {
//         printf("Thread %d: exceeded max iterations, node_addr=%d\n", tid,
//         node_addr); return;
//     }
//     // ---- 写入结果 ----
//     if (hit_prim < 0) {
//         out_hits[tid] = Intersection<float3>{};
//         return;
//     }

//     const Triangle<float3> &tri = triangles[hit_prim];
//     float w = 1.0f - hit_u - hit_v;
//     float3 interp_n = normalize(tri.n0 * w + tri.n1 * hit_u + tri.n2 *
//     hit_v); float3 hit_point = ray.origin + ray.direction * t_max;

//     Intersection<float3> out;
//     out.hit = 1;
//     out.t = t_max;
//     out.triangle_index = hit_prim;
//     out.point = hit_point;
//     out.normal = interp_n;
//     out_hits[tid] = out;
// }
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
__global__ void intersect_batch_kernel(const int width, const int height,
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
                           const std::vector<Triangle<Vec3>> &triangles) {
  node_count_ = static_cast<int>(nodes.size());
  tri_count_ = static_cast<int>(triangles.size());
  if (node_count_ <= 0 || tri_count_ <= 0) {
    throw std::invalid_argument(
        "prebuilt BVH requires non-empty nodes and triangles");
  }

  root_index_ = 0;
  validate_bvh(nodes, root_index_);
  std::cout << "BVH validated: " << node_count_ << " nodes, " << tri_count_
            << " triangles." << std::endl;
  // 将Vec3格式的nodes转为float3格式的nodes
  std::vector<BVHNode<float3>> float_nodes(node_count_);
  std::vector<Triangle<float3>> float_tris(tri_count_);
  for (int i = 0; i < tri_count_; ++i) {
    const auto &src_tri = triangles[static_cast<std::size_t>(i)];
    auto &dst_tri = float_tris[static_cast<std::size_t>(i)];
    dst_tri.v0 = make_float3(src_tri.v0.x(), src_tri.v0.y(), src_tri.v0.z());
    dst_tri.v1 = make_float3(src_tri.v1.x(), src_tri.v1.y(), src_tri.v1.z());
    dst_tri.v2 = make_float3(src_tri.v2.x(), src_tri.v2.y(), src_tri.v2.z());
    dst_tri.n0 = make_float3(src_tri.n0.x(), src_tri.n0.y(), src_tri.n0.z());
    dst_tri.n1 = make_float3(src_tri.n1.x(), src_tri.n1.y(), src_tri.n1.z());
    dst_tri.n2 = make_float3(src_tri.n2.x(), src_tri.n2.y(), src_tri.n2.z());
  }
  for (int i = 0; i < node_count_; ++i) {
    const auto &src_node = nodes[static_cast<std::size_t>(i)];
    auto &dst_node = float_nodes[static_cast<std::size_t>(i)];
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
                                       static_cast<std::size_t>(node_count_)));
  CUDA_CHECK(cudaMalloc(&d_tris_, sizeof(Triangle<float3>) *
                                      static_cast<std::size_t>(tri_count_)));
  CUDA_CHECK(cudaMemcpy(d_nodes_, float_nodes.data(),
                        sizeof(BVHNode<float3>) *
                            static_cast<std::size_t>(node_count_),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tris_, float_tris.data(),
                        sizeof(Triangle<float3>) *
                            static_cast<std::size_t>(tri_count_),
                        cudaMemcpyHostToDevice));
}
void GpuRaycaster::validate_bvh(const std::vector<BVHNode<Vec3>> &nodes,
                                int root) const {
  std::vector<bool> visited(nodes.size(), false);
  std::vector<int> stack;
  stack.push_back(root);
  while (!stack.empty()) {
    int idx = stack.back();
    stack.pop_back();
    if (idx < 0 || idx >= (int)nodes.size()) {
      std::cerr << "BVH invalid index: " << idx << std::endl;
      return;
    }
    if (visited[idx]) {
      std::cerr << "BVH cycle detected at " << idx << std::endl;
      return;
    }
    visited[idx] = true;
    const auto &node = nodes[idx];
    if (node.count > 0)
      continue; // leaf
    if (node.left < 0 || node.left >= (int)nodes.size() || node.right < 0 ||
        node.right >= (int)nodes.size()) {
      std::cerr << "BVH child index out of range at " << idx << std::endl;
      return;
    }
    stack.push_back(node.right);
    stack.push_back(node.left);
  }
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
  intersect_batch_kernel<<<grid_size, block_size>>>(width, height, tri_count_,
                                                    node_count_, d_nodes_,
                                                    d_tris_, d_rays_, d_hits_);
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
