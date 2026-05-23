#include "helper_transform.hpp"
#include "tools/build_from_raster.hpp"
#include "tools/loader.hpp"

#include <cuda_runtime.h>
#include <vector_functions.h>

namespace tools {

__device__ inline float clamp_acos_input(float v) {
  return fminf(1.0f, fmaxf(-1.0f, v));
}

__device__ inline bool is_valid_vertex_hash(int h) { return h >= 0; }

__device__ inline bool is_valid_vec3(const float3 &v) {
  return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

/**
 * @brief 计算三角形 tri_a, tri_b, tri_c 对 target_vid 顶点的角加权法线贡献。
 *
 * 三角形有向顺序为：
 *
 *   tri_a -> tri_b -> tri_c
 *
 * 法线为：
 *
 *   normal = normalize(cross(p_b - p_a, p_c - p_a))
 *
 * 对 target_vid 的贡献为：
 *
 *   normal * angle_at_target
 *
 * 如果 target_vid 不属于该三角形，则不贡献。
 */
__device__ inline void accumulate_triangle_normal_for_vertex(
    const int target_vid, const int tri_a, const int tri_b, const int tri_c,
    const float3 *vertices, float3 &normal_acc,
    NormalOrientationMode orient_mode) {
  if (target_vid != tri_a && target_vid != tri_b && target_vid != tri_c) {
    return;
  }

  const float3 pa = vertices[tri_a];
  const float3 pb = vertices[tri_b];
  const float3 pc = vertices[tri_c];

  if (!is_valid_vec3(pa) || !is_valid_vec3(pb) || !is_valid_vec3(pc)) {
    return;
  }

  float3 face_normal = cross(pb - pa, pc - pa);
  const float face_len = length(face_normal);

  if (!isfinite(face_len) || face_len < 1e-12f) {
    return;
  }

  face_normal = face_normal / face_len;

  float3 p_target;
  float3 p_other_1;
  float3 p_other_2;

  if (target_vid == tri_a) {
    p_target = pa;
    p_other_1 = pb;
    p_other_2 = pc;
  } else if (target_vid == tri_b) {
    p_target = pb;
    p_other_1 = pc;
    p_other_2 = pa;
  } else {
    p_target = pc;
    p_other_1 = pa;
    p_other_2 = pb;
  }

  float3 e1 = p_other_1 - p_target;
  float3 e2 = p_other_2 - p_target;

  const float len1 = length(e1);
  const float len2 = length(e2);

  if (!isfinite(len1) || !isfinite(len2) || len1 < 1e-12f || len2 < 1e-12f) {
    return;
  }

  e1 = e1 / len1;
  e2 = e2 / len2;

  const float cos_angle = clamp_acos_input(dot(e1, e2));
  const float angle_weight = acosf(cos_angle);

  if (!isfinite(angle_weight)) {
    return;
  }

  /**
   * 可选：保证法线大致朝向球体外侧。
   *
   * 如果你的三角形绕序已经稳定且确认朝外，可以删除这段。
   * 对球形 DEM 来说，用 dot(face_normal, p_target) 判断朝向通常比较稳。
   */
  if (orient_mode == NORMAL_ORIENT_SPHERE_OUTWARD) {
    if (dot(face_normal, p_target) < 0.0f) {
      face_normal = face_normal * -1.0f;
    }
  } else if (orient_mode == NORMAL_ORIENT_PLANAR_Z_UP) {
    if (face_normal.z < 0.0f) {
      face_normal = face_normal * -1.0f;
    }
  }
  normal_acc += face_normal * angle_weight;
}

/**
 * @brief 判断 cell 是否完整有效，并取出该 cell 四个角点对应的 compact vertex
 * id。
 *
 * cell 左上角 raster 坐标为 cx, cy。
 *
 * v1 ---- v2
 * |    / |
 * |  /   |
 * v3 ---- v4
 */
__device__ inline bool fetch_cell_vertices(const int cx, const int cy,
                                           const int width, const int height,
                                           const int *vertices_hash,
                                           int &tri_v1, int &tri_v2,
                                           int &tri_v3, int &tri_v4) {
  if (cx < 0 || cy < 0 || cx >= width - 1 || cy >= height - 1) {
    return false;
  }

  const int r_v1 = cy * width + cx;
  const int r_v2 = cy * width + (cx + 1);
  const int r_v3 = (cy + 1) * width + cx;
  const int r_v4 = (cy + 1) * width + (cx + 1);

  const int h1 = vertices_hash[r_v1];
  const int h2 = vertices_hash[r_v2];
  const int h3 = vertices_hash[r_v3];
  const int h4 = vertices_hash[r_v4];

  if (!is_valid_vertex_hash(h1) || !is_valid_vertex_hash(h2) ||
      !is_valid_vertex_hash(h3) || !is_valid_vertex_hash(h4)) {
    return false;
  }

  tri_v1 = static_cast<int>(h1);
  tri_v2 = static_cast<int>(h2);
  tri_v3 = static_cast<int>(h3);
  tri_v4 = static_cast<int>(h4);

  return true;
}

/**
 * @brief 根据 DEM、纬度、经度构建 compact vertices。
 *
 * 注意：
 * vertices_hash 的类型建议使用 int。
 * 无效位置为 -1。
 *
 * 调用该 kernel 前应执行：
 *
 * cudaMemset(d_vertices_hash, 0xff, width * height * sizeof(int));
 * cudaMemset(d_vertice_count, 0, sizeof(int));
 */
__global__ void
batch_build_vertices_kernel(const float *dem_data, const float *lats,
                            const float *lons, const float radii,
                            const int width, const int height, float3 *vertices,
                            int *vertice_count, int *vertices_hash) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= static_cast<int>(width) || y >= static_cast<int>(height)) {
    return;
  }

  const int raster_idx = y * width + x;
  const float z = dem_data[raster_idx];

  // 高程无效则跳过
  if (!isfinite(z) || fabsf(z) > 1e8f) {
    return;
  }

  const float lat = lats[y];
  const float lon = lons[x];

  // 经纬度无效则跳过
  if (!isfinite(lat) || !isfinite(lon) || fabsf(lat) > 90.0f ||
      fabsf(lon) > 360.0f) {
    return;
  }

  const int vertex_idx = atomicAdd(vertice_count, 1);

  if (vertex_idx >= static_cast<int>(width * height)) {
    return;
  }

  vertices[vertex_idx] = latlon_to_xyz(radiansf(lat), radiansf(lon), z, radii);

  vertices_hash[raster_idx] = static_cast<int>(vertex_idx);
}

__global__ void
batch_build_vertices_kernel(const float *dem_data, const float *y_coords,
                            const float *x_coords, const int width,
                            const int height, float3 *vertices,
                            int *vertice_count, int *vertices_hash) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= static_cast<int>(width) || y >= static_cast<int>(height)) {
    return;
  }

  const int raster_idx = y * width + x;
  const float z = dem_data[raster_idx];

  // 高程无效则跳过
  if (!isfinite(z) || fabsf(z) > 1e8f) {
    return;
  }

  const float y_coord = y_coords[y];
  const float x_coord = x_coords[x];

  const int vertex_idx = atomicAdd(vertice_count, 1);

  if (vertex_idx >= static_cast<int>(width * height)) {
    return;
  }

  vertices[vertex_idx] = make_float3(x_coord, y_coord, z);

  vertices_hash[raster_idx] = static_cast<int>(vertex_idx);
}

__global__ void batch_build_vertices_kernel(
    const float *dem_data, const float *albedo_data, const float *lats,
    const float *lons, const float radii, const int width, const int height,
    float3 *vertices, float *abledos, int *vertice_count, int *vertices_hash) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= static_cast<int>(width) || y >= static_cast<int>(height)) {
    return;
  }

  const int raster_idx = y * width + x;
  const float z = dem_data[raster_idx];
  const float albedo = albedo_data[raster_idx];

  // 高程无效则跳过
  if (!isfinite(z) || fabsf(z) > 1e8f || albedo < 0.0f || albedo > 1.0f) {
    return;
  }

  const float lat = lats[y];
  const float lon = lons[x];

  // 经纬度无效则跳过
  if (!isfinite(lat) || !isfinite(lon) || fabsf(lat) > 90.0f ||
      fabsf(lon) > 360.0f) {
    return;
  }

  const int vertex_idx = atomicAdd(vertice_count, 1);

  if (vertex_idx >= static_cast<int>(width * height)) {
    return;
  }

  vertices[vertex_idx] = latlon_to_xyz(radiansf(lat), radiansf(lon), z, radii);
  abledos[vertex_idx] = albedo;

  vertices_hash[raster_idx] = static_cast<int>(vertex_idx);
}

__global__ void batch_build_vertices_kernel(
    const float *dem_data, const float *albedo_data, const float *y_coords,
    const float *x_coords, const int width, const int height, float3 *vertices,
    float *abledos, int *vertice_count, int *vertices_hash) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= static_cast<int>(width) || y >= static_cast<int>(height)) {
    return;
  }

  const int raster_idx = y * width + x;
  const float z = dem_data[raster_idx];

  // 高程无效则跳过
  if (!isfinite(z) || fabsf(z) > 1e8f) {
    return;
  }

  const float y_coord = y_coords[y];
  const float x_coord = x_coords[x];

  const int vertex_idx = atomicAdd(vertice_count, 1);

  if (vertex_idx >= static_cast<int>(width * height)) {
    return;
  }

  vertices[vertex_idx] = make_float3(x_coord, y_coord, z);
  abledos[vertex_idx] = albedo_data[raster_idx];

  vertices_hash[raster_idx] = static_cast<int>(vertex_idx);
}
/**
 * @brief 只构建三角形，不计算 normals。
 *
 * 每个有效 cell 生成两个三角形：
 *
 *   T1 = v1, v3, v2
 *   T2 = v2, v3, v4
 *
 * 调用前应执行：
 *
 * cudaMemset(d_triangle_count, 0, sizeof(int));
 */
__global__ void batch_build_triangles_kernel(const int width, const int height,
                                             const int *vertices_hash,
                                             int3 *triangles,
                                             int *triangle_count) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= static_cast<int>(width - 1) || y >= static_cast<int>(height - 1)) {
    return;
  }

  int tri_v1;
  int tri_v2;
  int tri_v3;
  int tri_v4;

  const bool valid_cell = fetch_cell_vertices(
      static_cast<int>(x), static_cast<int>(y), width, height, vertices_hash,
      tri_v1, tri_v2, tri_v3, tri_v4);

  if (!valid_cell) {
    return;
  }

  const int tri_idx = atomicAdd(triangle_count, 2);

  triangles[tri_idx] =
      make_int3(static_cast<int>(tri_v1), static_cast<int>(tri_v3),
                static_cast<int>(tri_v2));

  triangles[tri_idx + 1] =
      make_int3(static_cast<int>(tri_v2), static_cast<int>(tri_v3),
                static_cast<int>(tri_v4));
}

/**
 * @brief 逐顶点 gather 邻域三角形来计算顶点法线。
 *
 * 该 kernel 的并行粒度是 raster pixel。
 *
 * 每个有效 raster pixel 对应一个 compact vertex。
 * 该线程只写 normals[my_vertex_id]，因此不需要 atomicAdd。
 *
 * 一个 raster 点最多属于周围 4 个 cell：
 *
 *   左上 cell:     x - 1, y - 1
 *   上方 cell:     x,     y - 1
 *   左侧 cell:     x - 1, y
 *   当前右下 cell: x,     y
 *
 * 每个 cell 包含两个三角形：
 *
 *   T1 = v1, v3, v2
 *   T2 = v2, v3, v4
 */
__global__ void batch_gather_vertex_normals_kernel(
    const int width, const int height, const float3 *vertices,
    const int *vertices_hash, NormalOrientationMode orient_mode,
    float3 *normals) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

  if (x >= width || y >= height) {
    return;
  }

  const int raster_idx = y * width + x;
  const int my_hash = vertices_hash[raster_idx];

  if (!is_valid_vertex_hash(my_hash)) {
    return;
  }

  const int my_vertex_id = static_cast<int>(my_hash);

  float3 normal_acc = make_float3(0.0f, 0.0f, 0.0f);

  /**
   * 遍历当前 raster 点周围 4 个 cell。
   *
   * offsets:
   *   (-1, -1)
   *   ( 0, -1)
   *   (-1,  0)
   *   ( 0,  0)
   */
#pragma unroll
  for (int oy = -1; oy <= 0; ++oy) {
#pragma unroll
    for (int ox = -1; ox <= 0; ++ox) {
      const int cell_x = x + ox;
      const int cell_y = y + oy;

      int tri_v1;
      int tri_v2;
      int tri_v3;
      int tri_v4;

      const bool valid_cell =
          fetch_cell_vertices(cell_x, cell_y, width, height, vertices_hash,
                              tri_v1, tri_v2, tri_v3, tri_v4);

      if (!valid_cell) {
        continue;
      }

      /**
       * cell 结构：
       *
       *   v1 ---- v2
       *    |    / |
       *    |  /   |
       *   v3 ---- v4
       *
       * 两个三角形：
       *
       *   T1 = v1, v3, v2
       *   T2 = v2, v3, v4
       */
      accumulate_triangle_normal_for_vertex(my_vertex_id, tri_v1, tri_v3,
                                            tri_v2, vertices, normal_acc,
                                            orient_mode);

      accumulate_triangle_normal_for_vertex(my_vertex_id, tri_v2, tri_v3,
                                            tri_v4, vertices, normal_acc,
                                            orient_mode);
    }
  }

  const float len = length(normal_acc);

  if (!isfinite(len) || len < 1e-12f) {
    normals[my_vertex_id] = make_float3(0.0f, 0.0f, 0.0f);
    return;
  }

  normals[my_vertex_id] = normal_acc / len;
}

/**
 * @brief 从 raster 数据构建三角形网格，加载经纬度坐标
 */
__host__ void build_from_raster(const float *d_dem_data, const float *d_lats,
                                const float *d_lons, const float radii,
                                const int width, const int height,
                                float3 *d_vertices, int *d_vertice_count,
                                int3 *d_triangles, int *d_triangle_count,
                                float3 *d_normals) {
  dim3 block_size(16, 16);
  dim3 grid_vertices((width + block_size.x - 1) / block_size.x,
                     (height + block_size.y - 1) / block_size.y);
  dim3 grid_cells((width - 1 + block_size.x - 1) / block_size.x,
                  (height - 1 + block_size.y - 1) / block_size.y);
  int total_pixels = width * height;
  int *d_vertices_hash;
  cudaMalloc(&d_vertices_hash, total_pixels * sizeof(int));
  cudaMemset(d_vertices_hash, 0xff,
             total_pixels * sizeof(int)); // -1 表示无效值

  // 构建 vertices
  batch_build_vertices_kernel<<<grid_vertices, block_size>>>(
      d_dem_data, d_lats, d_lons, radii, width, height, d_vertices,
      d_vertice_count, d_vertices_hash);
  // 构建 triangles
  batch_build_triangles_kernel<<<grid_cells, block_size>>>(
      width, height, d_vertices_hash, d_triangles, d_triangle_count);
  // 计算 normals，得到的法线已经是单位向量
  batch_gather_vertex_normals_kernel<<<grid_vertices, block_size>>>(
      width, height, d_vertices, d_vertices_hash, NORMAL_ORIENT_SPHERE_OUTWARD,
      d_normals);

  cudaFree(d_vertices_hash);
}
/**
 * @brief 从 raster 数据构建三角形网格，加载平面坐标
 */
__host__ void build_from_raster(const float *d_dem_data,
                                const float *d_y_coords,
                                const float *d_x_coords, const int width,
                                const int height, float3 *d_vertices,
                                int *d_vertice_count, int3 *d_triangles,
                                int *d_triangle_count, float3 *d_normals) {
  dim3 block_size(16, 16);
  dim3 grid_vertices((width + block_size.x - 1) / block_size.x,
                     (height + block_size.y - 1) / block_size.y);
  dim3 grid_cells((width - 1 + block_size.x - 1) / block_size.x,
                  (height - 1 + block_size.y - 1) / block_size.y);
  int total_pixels = width * height;
  int *d_vertices_hash;
  cudaMalloc(&d_vertices_hash, total_pixels * sizeof(int));
  cudaMemset(d_vertices_hash, 0xff,
             total_pixels * sizeof(int)); // -1 表示无效值

  // 构建 vertices
  batch_build_vertices_kernel<<<grid_vertices, block_size>>>(
      d_dem_data, d_y_coords, d_x_coords, width, height, d_vertices,
      d_vertice_count, d_vertices_hash);
  // 构建 triangles
  batch_build_triangles_kernel<<<grid_cells, block_size>>>(
      width, height, d_vertices_hash, d_triangles, d_triangle_count);
  // 计算 normals，得到的法线已经是单位向量
  batch_gather_vertex_normals_kernel<<<grid_vertices, block_size>>>(
      width, height, d_vertices, d_vertices_hash, NORMAL_ORIENT_PLANAR_Z_UP,
      d_normals);

  cudaFree(d_vertices_hash);
}
/**
 * @brief 从 raster 数据构建三角形网格和反照率，加载经纬度坐标
 */
__host__ void build_from_raster(const float *d_dem_data,
                                const float *d_albedo_data, const float *d_lats,
                                const float *d_lons, const float radii,
                                const int width, const int height,
                                float3 *d_vertices, int *d_vertice_count,
                                int3 *d_triangles, int *d_triangle_count,
                                float3 *d_normals, float *d_albedos) {
  dim3 block_size(16, 16);
  dim3 grid_vertices((width + block_size.x - 1) / block_size.x,
                     (height + block_size.y - 1) / block_size.y);
  dim3 grid_cells((width - 1 + block_size.x - 1) / block_size.x,
                  (height - 1 + block_size.y - 1) / block_size.y);
  int total_pixels = width * height;
  int *d_vertices_hash;
  cudaMalloc(&d_vertices_hash, total_pixels * sizeof(int));
  cudaMemset(d_vertices_hash, 0xff,
             total_pixels * sizeof(int)); // -1 表示无效值

  // 构建 vertices 和 albedos
  batch_build_vertices_kernel<<<grid_vertices, block_size>>>(
      d_dem_data, d_albedo_data, d_lats, d_lons, radii, width, height,
      d_vertices, d_albedos, d_vertice_count, d_vertices_hash);

  // 构建 triangles
  batch_build_triangles_kernel<<<grid_cells, block_size>>>(
      width, height, d_vertices_hash, d_triangles, d_triangle_count);
  // 计算 normals，得到的法线已经是单位向量
  batch_gather_vertex_normals_kernel<<<grid_vertices, block_size>>>(
      width, height, d_vertices, d_vertices_hash, NORMAL_ORIENT_SPHERE_OUTWARD,
      d_normals);

  cudaFree(d_vertices_hash);
}

/**
 * @brief 从 raster 数据构建三角形网格和反照率，加载平面坐标
 */
__host__ void
build_from_raster(const float *d_dem_data, const float *d_albedo_data,
                  const float *d_y_coords, const float *d_x_coords,
                  const int width, const int height, float3 *d_vertices,
                  int *d_vertice_count, int3 *d_triangles,
                  int *d_triangle_count, float3 *d_normals, float *d_albedos) {
  dim3 block_size(16, 16);
  dim3 grid_vertices((width + block_size.x - 1) / block_size.x,
                     (height + block_size.y - 1) / block_size.y);
  dim3 grid_cells((width - 1 + block_size.x - 1) / block_size.x,
                  (height - 1 + block_size.y - 1) / block_size.y);
  int total_pixels = width * height;
  int *d_vertices_hash;
  cudaMalloc(&d_vertices_hash, total_pixels * sizeof(int));
  cudaMemset(d_vertices_hash, 0xff,
             total_pixels * sizeof(int)); // -1 表示无效值

  // 构建 vertices 和 albedos
  batch_build_vertices_kernel<<<grid_vertices, block_size>>>(
      d_dem_data, d_albedo_data, d_y_coords, d_x_coords, width, height,
      d_vertices, d_albedos, d_vertice_count, d_vertices_hash);
  // 构建 triangles
  batch_build_triangles_kernel<<<grid_cells, block_size>>>(
      width, height, d_vertices_hash, d_triangles, d_triangle_count);
  // 计算 normals，得到的法线已经是单位向量
  batch_gather_vertex_normals_kernel<<<grid_vertices, block_size>>>(
      width, height, d_vertices, d_vertices_hash, NORMAL_ORIENT_PLANAR_Z_UP,
      d_normals);

  cudaFree(d_vertices_hash);
}

__global__ void apply_rigid_transform(float3 *vectors, const int vector_count,
                                      const float3 translation,
                                      const float4 quaternion) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= vector_count) {
    return;
  }

  float3 v = vectors[idx];
  // 旋转
  float3 rotated = rotation_vector(quaternion, v);
  // 平移
  vectors[idx] = rotated + translation;
}

void transform_normalization(const float lat, const float lon,
                             const float radii, float3 &translation,
                             float4 &quaternion) {
  // 将当前点转换至北极点
  float3 north_pole =
      latlon_to_xyz(radiansf(90.0f), radiansf(0.0f), 0.0f, radii);
  float3 current_point =
      latlon_to_xyz(radiansf(lat), radiansf(lon), 0.0f, radii);
  float3 axis = cross(current_point, north_pole);
  float axis_len = length(axis);
  if (axis_len < 1e-12f) {
    // 如果当前点已经在北极点，选择任意垂直轴
    axis = make_float3(1.0f, 0.0f, 0.0f);
    axis_len = 1.0f;
  }
  axis /= axis_len;
  float angle = acosf(dot(current_point, north_pole) /
                      (length(current_point) * length(north_pole)));
  float half_angle = angle / 2.0f;
  quaternion = make_float4(sinf(half_angle) * axis.x, sinf(half_angle) * axis.y,
                           sinf(half_angle) * axis.z, cosf(half_angle));

  translation = -rotation_vector(quaternion, current_point);
}

__global__ void trianglize(const float3 *vertices, const float3 *normals,
                           const int3 *indices, const int triangle_count,
                           dvs::Triangle<float3> *triangles) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= triangle_count)
    return;

  const int3 &tri_idx = indices[idx];
  triangles[idx].v0 = vertices[tri_idx.x];
  triangles[idx].v1 = vertices[tri_idx.y];
  triangles[idx].v2 = vertices[tri_idx.z];
  triangles[idx].n0 = normals[tri_idx.x];
  triangles[idx].n1 = normals[tri_idx.y];
  triangles[idx].n2 = normals[tri_idx.z];
}

__device__ void find_closest_pixel(const float *lats, const float height,
                                   const float *lons, const float width,
                                   const float target_lat,
                                   const float target_lon, int2 *out_lat_idx,
                                   int2 *out_lon_idx) {
  // 二分法查找最接近 target_lat 和 target_lon 的像素位置
  // 查找纬度，注意纬度坐标是倒序的，所以比较时需要反过来
  int left = 0, right = height - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (lats[mid] > target_lat) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  if (left >= height || right < 0) {
    *out_lat_idx = make_int2(-1, -1); // 无效索引
  } else {
    *out_lat_idx = make_int2(left, right);
  }
  // 查找经度
  left = 0;
  right = width - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (lons[mid] < target_lon) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  if (left >= width || right < 0) {
    *out_lon_idx = make_int2(-1, -1); // 无效索引
  } else {
    *out_lon_idx = make_int2(right, left);
  }
}

__global__ void batch_align_kernel(const float *ref_rst, const float *ref_lats,
                                   const float *ref_lons, const int ref_width,
                                   const int ref_height, const float *src_rst,
                                   const float *src_lats, const float *src_lons,
                                   const int src_width, const int src_height,
                                   float *aligned_src_rst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ref_width || y >= ref_height) {
    return;
  }
  int idx = y * ref_width + x;

  float ref_lat = ref_lats[y];
  float ref_lon = ref_lons[x];

  // 二分法查找 src_rst 中最接近 ref_lat 和 ref_lon 的像素位置
  int2 src_lat_idx, src_lon_idx;
  find_closest_pixel(src_lats, src_height, src_lons, src_width, ref_lat,
                     ref_lon, &src_lat_idx, &src_lon_idx);

  if (src_lat_idx.x == -1 || src_lon_idx.x == -1) {
    aligned_src_rst[idx] = 0.0f; // 无效位置，设置为0
    return;
  }

  // 双线性插值计算对齐后的像素值
  float lat1 = src_lats[src_lat_idx.y];
  float lat2 = src_lats[src_lat_idx.x];
  float lon1 = src_lons[src_lon_idx.y];
  float lon2 = src_lons[src_lon_idx.x];
  float q11 = src_rst[src_lat_idx.y * src_width + src_lon_idx.y];
  float q12 = src_rst[src_lat_idx.y * src_width + src_lon_idx.x];
  float q21 = src_rst[src_lat_idx.x * src_width + src_lon_idx.y];
  float q22 = src_rst[src_lat_idx.x * src_width + src_lon_idx.x];

  float aligned_value = (q11 * (lat2 - ref_lat) * (lon2 - ref_lon) +
                         q21 * (ref_lat - lat1) * (lon2 - ref_lon) +
                         q12 * (lat2 - ref_lat) * (ref_lon - lon1) +
                         q22 * (ref_lat - lat1) * (ref_lon - lon1)) /
                        ((lat2 - lat1) * (lon2 - lon1));
  aligned_src_rst[idx] = aligned_value;
}

__host__ bool load_from_raster(const std::string &dem_path,
                               const std::string &albedo_path,
                               std::vector<dvs::Triangle<float3>> &h_tris,
                               std::vector<float> &h_albs) {
  // 加载 DEM 和 albedo 数据
  tools::Imagery dem, albedo;
  if (!tools::loadImageryData(dem_path, dem)) {
    throw std::runtime_error("failed to load DEM data from: " + dem_path);
    return false;
  }

  std::cout << "DEM dimensions: " << dem.width << "x" << dem.height
            << std::endl;
  if (!tools::loadImageryData(albedo_path, albedo)) {
    throw std::runtime_error("failed to load albedo data from: " + albedo_path);
    return false;
  }

  std::cout << "Albedo dimensions: " << albedo.width << "x" << albedo.height
            << std::endl;

  // Albedo 对齐至 DEM，DEM各个顶点生成三角形和表面法线
  // 将 DEM 和 albedo 数据上传到 GPU，执行对齐和法线计算的 CUDA 内核
  float *d_dem, *d_albedo, *d_aligned_albedo, *d_dem_xs, *d_dem_ys,
      *d_albedo_xs, *d_albedo_ys;
  std::size_t memory_size =
      (dem.width * dem.height + albedo.width * albedo.height +
       dem.width * dem.height + dem.height + dem.width + albedo.height +
       albedo.width) *
      sizeof(float);

  std::cout << "Total GPU memory required: " << memory_size / (1024.0 * 1024.0)
            << " MB" << std::endl;
  std::size_t dem_size = dem.width * dem.height;
  std::size_t albedo_size = albedo.width * albedo.height;
  cudaMalloc(&d_dem, dem_size * sizeof(float));
  cudaMalloc(&d_albedo, albedo_size * sizeof(float));
  cudaMalloc(&d_aligned_albedo, dem_size * sizeof(float));
  cudaMalloc(&d_dem_ys, dem.height * sizeof(float));
  cudaMalloc(&d_dem_xs, dem.width * sizeof(float));
  cudaMalloc(&d_albedo_ys, albedo.height * sizeof(float));
  cudaMalloc(&d_albedo_xs, albedo.width * sizeof(float));
  cudaMemcpy(d_dem, dem.data.data(), dem_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_dem_ys, dem.y_coords.data(), dem.height * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_dem_xs, dem.x_coords.data(), dem.width * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_albedo, albedo.data.data(), albedo_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_albedo_ys, albedo.y_coords.data(), albedo.height * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_albedo_xs, albedo.x_coords.data(), albedo.width * sizeof(float),
             cudaMemcpyHostToDevice);
  dim3 blockSize(16, 16);
  dim3 gridSize((dem.width + blockSize.x - 1) / blockSize.x,
                (dem.height + blockSize.y - 1) / blockSize.y);
  tools::batch_align_kernel<<<gridSize, blockSize>>>(
      d_dem, d_dem_ys, d_dem_xs, dem.width, dem.height, d_albedo, d_albedo_ys,
      d_albedo_xs, albedo.width, albedo.height, d_aligned_albedo);
  cudaDeviceSynchronize();
  float3 *d_vertices, *d_normals;
  float *d_albs;
  int3 *d_triangles;
  cudaMalloc(&d_vertices, dem_size * sizeof(float3));
  cudaMalloc(&d_normals, dem_size * sizeof(float3));
  cudaMalloc(&d_albs, dem_size * sizeof(float));
  cudaMalloc(&d_triangles,
             (dem.width - 1) * (dem.height - 1) * 2 * sizeof(int3));
  cudaMemset(d_normals, 0, dem_size * sizeof(float3));
  cudaMemset(d_albs, 0, dem_size * sizeof(float));
  int *d_vertice_count, *d_triangle_count;
  cudaMalloc(&d_vertice_count, sizeof(int));
  cudaMalloc(&d_triangle_count, sizeof(int));
  cudaMemset(d_vertice_count, 0, sizeof(int));
  CUDA_CHECK(cudaMemset(d_triangle_count, 0, sizeof(int)),
             "Failed to initialize triangle count");

  tools::build_from_raster(d_dem, d_aligned_albedo, d_dem_ys, d_dem_xs,
                           dem.width, dem.height, d_vertices, d_vertice_count,
                           d_triangles, d_triangle_count, d_normals, d_albs);

  int h_vertice_count, h_triangle_count;
  cudaMemcpy(&h_vertice_count, d_vertice_count, sizeof(int),
             cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaMemcpy(&h_triangle_count, d_triangle_count, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "Failed to copy triangle count to host");
  std::cout << "Generated " << h_vertice_count << " vertices, "
            << h_vertice_count << " normals, and " << h_triangle_count
            << " triangles." << std::endl;
  cudaFree(d_dem);
  cudaFree(d_albedo);
  cudaFree(d_aligned_albedo);
  cudaFree(d_dem_ys);
  cudaFree(d_dem_xs);
  cudaFree(d_albedo_ys);
  cudaFree(d_albedo_xs);
  cudaFree(d_vertice_count);
  cudaFree(d_triangle_count);
  // 计算当前DEM的中心点经纬度、深度范围等信息，作为刚体变换的参数
  float y_cnt = dem.y_coords[dem.height / 2];
  float x_cnt = dem.x_coords[dem.width / 2];
  float depth_cnt = dem.data[dem.width * (dem.height / 2) + dem.width / 2];

  std::cout << "DEM center x: " << x_cnt << ", y: " << y_cnt
            << ", depth: " << depth_cnt << std::endl;
  // 定义一个刚体变换，将 DEM 的平均经纬度位置移动到原点，并保持当地水平
  float3 translation = make_float3(-x_cnt, -y_cnt, -depth_cnt);
  float4 rotation = make_float4(0.0f, 0.0f, 0.0f, 1.0f); // 单位四元数表示无旋转

  // 对各个顶点作刚体变换
  int thread_count = 256;
  int vertex_grid_size = (h_vertice_count + thread_count - 1) / thread_count;
  tools::apply_rigid_transform<<<vertex_grid_size, thread_count>>>(
      d_vertices, h_vertice_count, translation, rotation);
  tools::apply_rigid_transform<<<vertex_grid_size, thread_count>>>(
      d_normals, h_vertice_count, make_float3(0.0f, 0.0f, 0.0f), rotation);
  // 修改为当前dvs版本的三角形数据结构
  dvs::Triangle<float3> *d_tris;
  cudaMalloc(&d_tris, h_triangle_count * sizeof(dvs::Triangle<float3>));
  int tri_grid_size = (h_triangle_count + thread_count - 1) / thread_count;
  tools::trianglize<<<tri_grid_size, thread_count>>>(
      d_vertices, d_normals, d_triangles, h_triangle_count, d_tris);
  cudaFree(d_vertices);
  cudaFree(d_normals);
  cudaFree(d_triangles);

  // // 输出变换后的第一个顶点与法线，检查是否正确应用了刚体变换
  // std::cout << "First vertex after transform: (" << h_vertices[0].x << ", "
  //           << h_vertices[0].y << ", " << h_vertices[0].z << ")" <<
  //           std::endl;
  // std::cout << "First normal after transform: (" << h_normals[0].x << ", "
  //           << h_normals[0].y << ", " << h_normals[0].z << ")" << std::endl;

  h_tris.resize(h_triangle_count);
  h_albs.resize(h_vertice_count);
  cudaMemcpy(h_tris.data(), d_tris,
             h_triangle_count * sizeof(dvs::Triangle<float3>),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_albs.data(), d_albs, h_vertice_count * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(d_tris);
  cudaFree(d_albs);
  std::cout << "Converted triangles to DVS format." << std::endl;
  return true;
}
} // namespace tools