#pragma once
#include "dvs/helper_math.hpp"
#include "dvs/scene_types.hpp"
#include <cuda_runtime.h>

// 从DEM数据中构建三角形、法向量以及必要的反照率信息，并可对当前三维坐标进行必要的刚体变换
namespace tools {

inline __host__ __device__ float3 latlon_to_xyz(const float lat,
                                                const float lon,
                                                const float alt,
                                                const float radii) {
  // 将经纬度转换为三维坐标
  float x = (radii + alt) * cosf(lat) * cosf(lon);
  float y = (radii + alt) * cosf(lat) * sinf(lon);
  float z = (radii + alt) * sinf(lat);
  return make_float3(x, y, z);
}

inline __host__ __device__ float3 rotation_vector(const float4 quaternion,
                                                  const float3 vector) {
  // 使用四元数旋转向量
  float3 rot_mat_row0, rot_mat_row1, rot_mat_row2;
  rot_mat_row0 = make_float3(
      1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
      2 * (quaternion.x * quaternion.y - quaternion.z * quaternion.w),
      2 * (quaternion.x * quaternion.z + quaternion.y * quaternion.w));
  rot_mat_row1 = make_float3(
      2 * (quaternion.x * quaternion.y + quaternion.z * quaternion.w),
      1 - 2 * (quaternion.x * quaternion.x + quaternion.z * quaternion.z),
      2 * (quaternion.y * quaternion.z - quaternion.x * quaternion.w));
  rot_mat_row2 = make_float3(
      2 * (quaternion.x * quaternion.z - quaternion.y * quaternion.w),
      2 * (quaternion.y * quaternion.z + quaternion.x * quaternion.w),
      1 - 2 * (quaternion.x * quaternion.x + quaternion.y * quaternion.y));

  return make_float3(dot(rot_mat_row0, vector), dot(rot_mat_row1, vector),
                     dot(rot_mat_row2, vector));
}
/**
 * @brief 构建三角形顶点，不含反照率
 * @param dem_data DEM数据
 * @param lats DEM数据对应的纬度数组
 * @param lons DEM数据对应的经度数组
 * @param width DEM数据的宽度
 * @param height DEM数据的高度
 * @param vertices 输出的顶点数组
 * @param vertice_count 输出的顶点数量
 * @param triangles 输出的三角形索引数组
 * @param triangle_count 输出的三角形数量
 * @param normals 输出的法向量数组
 */
__host__ void build_from_raster(const float *dem_data, const float *lats,
                                const float *lons, const float radii,
                                const int width, const int height,
                                float3 *vertices, int *vertice_count,
                                int3 *triangles, int *triangle_count,
                                float3 *normals);

/**
 * @brief 构建三角形、法向量，含反照率
 * @param dem_data DEM数据
 * @param albedo_data 反照率数据
 * @param lats DEM数据对应的纬度数组
 * @param lons DEM数据对应的经度数组
 * @param width DEM数据的宽度
 * @param height DEM数据的高度
 * @param vertices 输出的顶点数组
 * @param vertice_count 输出的顶点数量
 * @param triangles 输出的三角形索引数组
 * @param triangle_count 输出的三角形数量
 * @param normals 输出的法向量数组
 * @param albedos 输出的反照率数组
 */
__host__ void build_from_raster(const float *dem_data, const float *albedo_data,
                                const float *lats, const float *lons,
                                const float radii, const int width,
                                const int height, float3 *vertices,
                                int *vertice_count, int3 *triangles,
                                int *triangle_count, float3 *normals,
                                float *albedos);

__global__ void apply_rigid_transform(float3 *vertices, const int vertice_count,
                                      const float3 translation,
                                      const float4 quaternion);

__global__ void trianglize(const float3 *vertices, const float3 *normals,
                           const int3 *indices, const int triangle_count,
                           dvs::Triangle<float3> *triangles);

void transform_normalization(const float lat, const float lon,
                             const float radii, float3 &translation,
                             float4 &quaternion);
} // namespace tools
