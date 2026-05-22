#pragma once
#include <cuda_runtime.h>
#include <sys/types.h>

// 从DEM数据中构建三角形、法向量以及必要的反照率信息，并可对当前三维坐标进行必要的刚体变换
namespace tools {

inline __device__ float3 latlon_to_xyz(const float lat, const float lon,
                                       const float alt, const float radii) {
  // 将经纬度转换为三维坐标
  float x = (radii + alt) * cosf(lat) * cosf(lon);
  float y = (radii + alt) * cosf(lat) * sinf(lon);
  float z = (radii + alt) * sinf(lat);
  return make_float3(x, y, z);
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

} // namespace tools
