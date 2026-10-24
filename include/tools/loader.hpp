#pragma once

#include <cpl_conv.h>
#include <cuda_runtime.h>
#include <gdal_priv.h>
#include <gdal_rat.h>
#include <gdal_version.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace tools {
struct Imagery {
  std::vector<float> data, lats, lons, x_coords, y_coords;
  int width=0,height=0;
  double transform[6]{};
  std::string projection;
};
/**
 * @brief 利用Gdal加载数据
 * @param path 数据路径，支持tiff和IMG格式，单个波段
 * @param imagery 加载后的数据格式，形状为(height,
 * width)，每个元素对应一个像素值
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
  if(gt[2]!=0 || gt[4]!=0 || gt[1]==0 || gt[5]==0) {
    std::cerr<<"Rotated/sheared or degenerate rasters require reprojection before planar mesh loading: "<<path<<"\n";
    GDALClose(dataset);return false;
  }
  std::copy(gt,gt+6,imagery.transform);
  imagery.projection=dataset->GetProjectionRef();
  int has_nodata=0,has_scale=0,has_offset=0;
  float nodata=band->GetNoDataValue(&has_nodata);
  double scale=band->GetScale(&has_scale),offset=band->GetOffset(&has_offset);
  if(!has_scale) scale=1;if(!has_offset) offset=0;
  size_t missing=0;
  for(auto &value:imagery.data) {
    if(!std::isfinite(value) || (has_nodata && value==nodata)) {value=NAN;++missing;}
    else value=value*scale+offset;
  }
  if(missing) std::cout<<path<<": "<<missing<<" NoData samples (NaN), scale="<<scale<<", offset="<<offset<<"\n";
  // 加载平面坐标系
  imagery.x_coords.resize(static_cast<std::size_t>(width));
  imagery.y_coords.resize(static_cast<std::size_t>(height));
#pragma omp parallel for
  for (int j = 0; j < width; ++j) {
    double x = gt[0] + (j + 0.5) * gt[1] + 0.5 * gt[2];
    imagery.x_coords[j] = static_cast<float>(x);
  }
#pragma omp parallel for
  for (int i = 0; i < height; ++i) {
    double y = gt[3] + 0.5 * gt[4] + (i + 0.5) * gt[5];
    imagery.y_coords[i] = static_cast<float>(y);
  }

  GDALClose(dataset);
  return true;
}

inline float reference_height(const Imagery &dem) {
  float center=dem.data[dem.width*(dem.height/2)+dem.width/2];
  if(std::isfinite(center)) return center;
  throw std::runtime_error("DEM center is NoData; choose a valid crop/reference origin before rendering");
}
inline void require_same_projection(const Imagery &reference,const Imagery &source) {
  if(reference.projection==source.projection) return;
  OGRSpatialReference a,b;
  if(reference.projection.empty() || source.projection.empty() ||
     a.SetFromUserInput(reference.projection.c_str())!=OGRERR_NONE ||
     b.SetFromUserInput(source.projection.c_str())!=OGRERR_NONE || !a.IsSame(&b))
    throw std::runtime_error("Raster CRS differ; reproject to a common CRS before alignment");
}
} // namespace tools
