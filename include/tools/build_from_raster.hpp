#pragma once
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

/**
 * @brief 构建三角形顶点，不含反照率，加载经纬度坐标
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
 * @brief 构建三角形顶点，不含反照率，加载平面坐标
 * @param dem_data DEM数据
 * @param x_coords DEM数据对应的X坐标数组
 * @param y_coords DEM数据对应的Y坐标数组
 * @param width DEM数据的宽度
 * @param height DEM数据的高度
 * @param vertices 输出的顶点数组
 * @param vertice_count 输出的顶点数量
 * @param triangles 输出的三角形索引数组
 * @param triangle_count 输出的三角形数量
 * @param normals 输出的法向量数组
 */
__host__ void build_from_raster(const float *d_dem_data,
                                const float *d_y_coords,
                                const float *d_x_coords, const int width,
                                const int height, float3 *d_vertices,
                                int *d_vertice_count, int3 *d_triangles,
                                int *d_triangle_count, float3 *d_normals);
/**
 * @brief 构建三角形、法向量，含反照率，加载经纬度坐标
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
/**
 * @brief 构建三角形、法向量，含反照率，加载平面坐标
 * @param dem_data DEM数据
 * @param albedo_data 反照率数据
 * @param x_coords DEM数据对应的X坐标数组
 * @param y_coords DEM数据对应的Y坐标数组
 * @param width DEM数据的宽度
 * @param height DEM数据的高度
 * @param vertices 输出的顶点数组
 * @param vertice_count 输出的顶点数量
 * @param triangles 输出的三角形索引数组
 * @param triangle_count 输出的三角形数量
 * @param normals 输出的法向量数组
 * @param albedos 输出的反照率数组
 */
__host__ void
build_from_raster(const float *d_dem_data, const float *d_albedo_data,
                  const float *d_y_coords, const float *d_x_coords,
                  const int width, const int height, float3 *d_vertices,
                  int *d_vertice_count, int3 *d_triangles,
                  int *d_triangle_count, float3 *d_normals, float *d_albedos);
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
