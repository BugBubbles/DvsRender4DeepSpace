#pragma once

#include <cpl_conv.h>
#include <cuda_runtime.h>
#include <gdal_priv.h>
#include <gdal_rat.h>
#include <gdal_version.h>
#include <iostream>

namespace tools {
struct Imagery {
  std::vector<float> data, lats, lons;
  int width;
  int height;
  float radii;
};
/**
 * @brief 利用Gdal加载数据
 * @param path 数据路径，支持tiff和IMG格式，单个波段
 * @param imagery 加载后的数据格式，形状为(height,
 * width)，每个元素对应一个像素值
 * @param latitude 纬度数据，形状为(height)，每个元素对应一个像素行
 * @param longitude 经度数据，形状为(width)，每个元素对应一个像素列
 * @return 加载成功返回true，失败返回false
 */
inline bool loadImageryData(const std::string &path, Imagery &imagery) {
  GDALAllRegister();
  GDALDataset *dataset =
      static_cast<GDALDataset *>(GDALOpen(path.c_str(), GA_ReadOnly));
  if (dataset == nullptr) {
    std::cerr << "Failed to open dataset: " << path << std::endl;
    return false;
  }

  GDALRasterBand *band = dataset->GetRasterBand(1);
  if (band == nullptr) {
    std::cerr << "Failed to get raster band from dataset: " << path
              << std::endl;
    GDALClose(dataset);
    return false;
  }

  int width = band->GetXSize();
  int height = band->GetYSize();
  imagery.width = width;
  imagery.height = height;
  imagery.data.resize(width * height);
  if (band->RasterIO(GF_Read, 0, 0, width, height, imagery.data.data(), width,
                     height, GDT_Float32, 0, 0) != CE_None) {
    std::cerr << "Failed to read raster data from dataset: " << path
              << std::endl;
    GDALClose(dataset);
    return false;
  }

  double gt[6];
  if (dataset->GetGeoTransform(gt) != CE_None) {
    std::cerr << "Failed to get geotransform from dataset: " << path
              << std::endl;
    GDALClose(dataset);
    return false;
  }

  const OGRSpatialReference *src_srs = dataset->GetSpatialRef();
  OGRSpatialReference src_copy;
  if (src_srs == nullptr) {
    const char *proj = dataset->GetProjectionRef();
    if (proj == nullptr || *proj == '\0') {
      std::cerr << "Missing projection metadata: " << path << std::endl;
      GDALClose(dataset);
      return false;
    }
    if (src_copy.importFromWkt(proj) != OGRERR_NONE) {
      std::cerr << "Failed to import WKT from dataset: " << path << std::endl;
      GDALClose(dataset);
      return false;
    }
    src_copy.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    src_srs = &src_copy;
  }

  OGRSpatialReference *geog_srs = src_srs->CloneGeogCS();
  if (geog_srs == nullptr) {
    std::cerr << "Failed to clone lunar geographic CRS from source: " << path
              << std::endl;
    GDALClose(dataset);
    return false;
  }
  geog_srs->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

  OGRCoordinateTransformation *to_geog =
      OGRCreateCoordinateTransformation(src_srs, geog_srs);
  if (to_geog == nullptr) {
    std::cerr << "Failed to create projected->geographic transformation: "
              << path << std::endl;
    OSRDestroySpatialReference(geog_srs);
    GDALClose(dataset);
    return false;
  }

  imagery.lats.resize(static_cast<std::size_t>(height));
  imagery.lons.resize(static_cast<std::size_t>(width));
#pragma omp parallel for
  for (int i = 0; i < height; ++i) {
    double x = gt[0] + 0.5 * gt[1] + (i + 0.5) * gt[2];
    double y = gt[3] + 0.5 * gt[4] + (i + 0.5) * gt[5];
    to_geog->Transform(1, &x, &y);
    imagery.lats[i] = y;
  }

#pragma omp parallel for
  for (int j = 0; j < width; ++j) {
    double x = gt[0] + (j + 0.5) * gt[1] + 0.5 * gt[2];
    double y = gt[3] + (j + 0.5) * gt[4] + 0.5 * gt[5];
    // 投影坐标 -> 月球经纬度
    to_geog->Transform(1, &x, &y);
    imagery.lons[j] = x;
  }

  // 加载当前投影坐标系的半径
  double a = geog_srs->GetSemiMajor();
  double b = geog_srs->GetSemiMinor();
  imagery.radii = static_cast<float>((a + b) * 0.5);

  OCTDestroyCoordinateTransformation(to_geog);
  OSRDestroySpatialReference(geog_srs);
  GDALClose(dataset);
  return true;
}
// static inline void geodeticToMoonECEF(const OGRSpatialReference *geog,
//                                       double lon_deg, double lat_deg,
//                                       double h, double &x, double &y,
//                                       double &z) {
//   const double a = geog->GetSemiMajor();
//   const double inv_f = geog->GetInvFlattening();
//   const double f = (inv_f > 0.0) ? (1.0 / inv_f) : 0.0;
//   const double e2 = f * (2.0 - f);

//   const double lon = lon_deg * dvs::kPi / 180.0;
//   const double lat = lat_deg * dvs::kPi / 180.0;

//   const double sin_lat = std::sin(lat);
//   const double cos_lat = std::cos(lat);
//   const double sin_lon = std::sin(lon);
//   const double cos_lon = std::cos(lon);

//   const double n = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);

//   x = (n + h) * cos_lat * cos_lon;
//   y = (n + h) * cos_lat * sin_lon;
//   z = (n * (1.0 - e2) + h) * sin_lat;
// }

} // namespace tools