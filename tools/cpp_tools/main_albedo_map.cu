/**
利用Gdal加载cub文件，逐像素计算Albedo，然后保存为Tiff文件
*/

#include "dvs/hapke.hpp"
#include "dvs/scene_types.hpp"

#include <cpl_conv.h>
#include <cuda_runtime.h>
#include <fstream>
#include <gdal_priv.h>
#include <gdal_rat.h>
#include <gdal_version.h>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace tools {

struct Options {
  std::string input_cub_path;
  std::string output_dir = "../output";
  std::string config_path = "../../config/default.yaml";
  float end_time = std::numeric_limits<float>::max();
};

void print_usage(const char *prog) {
  std::cout << "Usage: " << prog << " --input_cub <path> [options]\n"
            << "Options:\n"
            << "  --output <path>            output images file (default: ./)\n"
            << "  --config <path>            path to YAML config file "
               "(default: ../../config/default.yaml)\n"
            << std::endl;
}

float parse_float_arg(const char *key, const std::string &value) {
  try {
    return std::stof(value);
  } catch (const std::exception &) {
    throw std::runtime_error(std::string("invalid float for ") + key + ": " +
                             value);
  }
}

Options parse_args(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char *key) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + key);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--input_cub") {
      opts.input_cub_path = require_value("--input_cub");
    } else if (arg == "--output") {
      opts.output_dir = require_value("--output");
    } else if (arg == "--config") {
      opts.config_path = require_value("--config");
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (opts.input_cub_path.empty()) {
    throw std::runtime_error("--input_cub is required");
  }
  std::ifstream config_in(opts.config_path);
  if (!config_in.good()) {
    throw std::runtime_error("failed to open config file: " + opts.config_path);
  }
  return opts;
}

struct CubBands {
  int width = 0;
  int height = 0;
  std::vector<float> i_over_f;
  std::vector<float> phase;
  std::vector<float> emi;
  std::vector<float> inc;
  // 保存输入 cub 的仿射变换信息
  bool has_geo_transform = false;
  double geo_transform[6] = {0.0, 1.0, 0.0, 0.0, 0.0, -1.0};
  // 保存输入 cub 的投影 WKT
  bool has_projection = false;
  std::string projection_wkt;
};

__device__ float radians(float degrees) {
  return degrees * (dvs::kPi / 180.0f);
}
__device__ void hapke_constant(const float inc_angle, const float emi_angle,
                               const float phase_angle,
                               const dvs::HapkeConfig &p, float &f, float &p_g,
                               float &mu0, float &mue, float &K) {
  const float cosi = cosf(radians(inc_angle));
  const float cose = cosf(radians(emi_angle));
  const float sini = sinf(radians(inc_angle));
  const float sine = sinf(radians(emi_angle));
  const float cosg = cosf(radians(phase_angle));
  const float tan_g_div_2 = dvs::safe_tan(dvs::safe_acos(cosg) * 0.5f);
  const float psi =
      dvs::safe_acos((cosg - cosi * cose) / (sini * sine + 1e-6f));

  p_g = dvs::HG_double_lobe(cosg, p.b, p.c);
  const float BS = dvs::B_S(tan_g_div_2, p.hS);
  const float BC = dvs::B_C(tan_g_div_2, p.hC);
  K = dvs::porosity_K(p.phi);
  const dvs::RoughnessTerms rst =
      dvs::rough_S(cosi, cose, sini, sine, psi, p.thetap);
  const float LS = rst.mu0 / (rst.mu0 + rst.mue + 1e-6f);
  mu0 = rst.mu0;
  mue = rst.mue;

  f = (LS / cosi) * K * (1.0f + BC * p.B_C0) * rst.S / (4.0f * dvs::kPi);
  p_g = (p_g * (1.0f + BS * p.B_S0));
}

__device__ float compute_albedo(const float albedo, const float f,
                                const float p_g, const float mu0,
                                const float mue, const float K) {
  float M = dvs::Hfunc(mu0 / K, albedo) * dvs::Hfunc(mue / K, albedo) - 1.0f;
  return f * (p_g + M) * albedo;
}

__device__ float grad_wr0(const float w) {
  const float root = sqrtf(fmaxf(0.0f, 1.0f - w));
  const float r0 = (1.0f - root) / (1.0f + root);
  return r0 + (r0 + 1) * (r0 + 1) / 4.0f * w / root;
}

__device__ float grad_Hfunc(const float w, const float x, const float H) {
  const float x_ln_1_plus_x_over_x = x * logf(1 + 1.0f / x);
  float term1 = -x * grad_wr0(w);
  float term2 = -0.5 * x_ln_1_plus_x_over_x;
  float term3 = x * x_ln_1_plus_x_over_x * grad_wr0(w);
  float inv_H = 1.0f / H;

  return -(term1 + term2 + term3) * inv_H * inv_H;
}

__device__ float grad_M(const float w, const float mu0, const float mue,
                        const float K) {
  const float H_mu0 = dvs::Hfunc(mu0 / K, w);
  const float H_mue = dvs::Hfunc(mue / K, w);
  const float grad_H_mu0 = grad_Hfunc(w, mu0 / K, H_mu0);
  const float grad_H_mue = grad_Hfunc(w, mue / K, H_mue);
  return H_mue * grad_H_mu0 + H_mu0 * grad_H_mue;
}

__device__ float grad_residual(const float albedo, const float f,
                               const float p_g, const float mu0,
                               const float mue, const float K) {
  const float M =
      dvs::Hfunc(mu0 / K, albedo) * dvs::Hfunc(mue / K, albedo) - 1.0f;
  const float grad_M_val = grad_M(albedo, mu0, mue, K);
  return f * (p_g + M) + f * albedo * grad_M_val;
}
__global__ void batch_compute_albedo(const std::size_t pixel_count,
                                     const float *inc_angles,
                                     const float *emi_angles,
                                     const float *phase_angles,
                                     const float *i_over_f,
                                     const dvs::HapkeConfig p, float *albedos) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pixel_count) {
    return;
  }
  // 先计算固定不变的部分
  float f, p_g, mu0, mue, K;
  hapke_constant(inc_angles[idx], emi_angles[idx], phase_angles[idx], p, f, p_g,
                 mu0, mue, K);
  // 然后利用牛顿迭代法求解albedo
  float albedo = 0.5f; // 初始猜测值
  for (int iter = 0; iter < 5; ++iter) {
    float computed_i_over_f = compute_albedo(albedo, f, p_g, mu0, mue, K);
    float residual = computed_i_over_f - i_over_f[idx];
    if (fabsf(residual) < 1e-6f) {
      break; // 收敛}
    }
    float grad = grad_residual(albedo, f, p_g, mu0, mue, K);
    if (fabsf(grad) < 1e-6f) {
      break; // 避免除以零
    }
    albedo -= residual / grad; // 牛顿迭代更新
    albedo =
        fmaxf(0.0f, fminf(1.0f - 1e-5, albedo)); // 限制albedo在[0, 1]范围内
  }
  albedos[idx] = albedo;
}

bool load_cub_file(const std::string &path, CubBands &out, std::string &err) {
  GDALAllRegister();

  GDALDataset *ds =
      static_cast<GDALDataset *>(GDALOpen(path.c_str(), GA_ReadOnly));
  if (!ds) {
    err = "failed to open cub file: " + path;
    return false;
  }

  const int band_count = ds->GetRasterCount();
  if (band_count < 4) {
    err = "cub file has fewer than 4 bands";
    GDALClose(ds);
    return false;
  }

  out.width = ds->GetRasterXSize();
  out.height = ds->GetRasterYSize();

  // ============================================================
  // 读取 cub 的 GeoTransform
  // ============================================================
  double gt[6];
  if (ds->GetGeoTransform(gt) == CE_None) {
    out.has_geo_transform = true;
    std::copy(gt, gt + 6, out.geo_transform);

    std::cout << "[GDAL] Input cub GeoTransform:\n"
              << "       Origin X      = " << out.geo_transform[0] << "\n"
              << "       Pixel Width   = " << out.geo_transform[1] << "\n"
              << "       Rotation X    = " << out.geo_transform[2] << "\n"
              << "       Origin Y      = " << out.geo_transform[3] << "\n"
              << "       Rotation Y    = " << out.geo_transform[4] << "\n"
              << "       Pixel Height  = " << out.geo_transform[5] << "\n";
  } else {
    out.has_geo_transform = false;
    std::cerr << "[GDAL] Warning: input cub has no GeoTransform.\n";
  }

  // ============================================================
  // 读取 cub 的投影坐标系
  // 优先使用 GDAL 3 的 GetSpatialRef()
  // ============================================================
#if GDAL_VERSION_MAJOR >= 3
  const OGRSpatialReference *srs = ds->GetSpatialRef();
  if (srs != nullptr) {
    char *wkt = nullptr;
    if (srs->exportToWkt(&wkt) == OGRERR_NONE && wkt != nullptr) {
      out.has_projection = true;
      out.projection_wkt = wkt;
      CPLFree(wkt);
      std::cout << "[GDAL] Input cub projection loaded from SpatialRef.\n";
    } else {
      out.has_projection = false;
      std::cerr << "[GDAL] Warning: failed to export SpatialRef to WKT.\n";
    }
  } else {
    const char *proj = ds->GetProjectionRef();
    if (proj != nullptr && proj[0] != '\0') {
      out.has_projection = true;
      out.projection_wkt = proj;
      std::cout << "[GDAL] Input cub projection loaded from ProjectionRef.\n";
    } else {
      out.has_projection = false;
      std::cerr << "[GDAL] Warning: input cub has no projection.\n";
    }
  }
#else
  const char *proj = ds->GetProjectionRef();
  if (proj != nullptr && proj[0] != '\0') {
    out.has_projection = true;
    out.projection_wkt = proj;
    std::cout << "[GDAL] Input cub projection loaded from ProjectionRef.\n";
  } else {
    out.has_projection = false;
    std::cerr << "[GDAL] Warning: input cub has no projection.\n";
  }
#endif

  const std::size_t pixel_count = static_cast<std::size_t>(out.width) *
                                  static_cast<std::size_t>(out.height);

  out.i_over_f.resize(pixel_count);
  out.phase.resize(pixel_count);
  out.emi.resize(pixel_count);
  out.inc.resize(pixel_count);

  auto read_band = [&](int band_index, std::vector<float> &dst) -> bool {
    GDALRasterBand *band = ds->GetRasterBand(band_index);
    if (!band) {
      err = "failed to get band " + std::to_string(band_index);
      return false;
    }
    CPLErr status =
        band->RasterIO(GF_Read, 0, 0, out.width, out.height, dst.data(),
                       out.width, out.height, GDT_Float32, 0, 0);
    if (status != CE_None) {
      err = "failed to read band " + std::to_string(band_index);
      return false;
    }
    return true;
  };

  if (!read_band(1, out.i_over_f)) {
    GDALClose(ds);
    return false;
  }
  if (!read_band(2, out.phase)) {
    GDALClose(ds);
    return false;
  }
  if (!read_band(3, out.emi)) {
    GDALClose(ds);
    return false;
  }
  if (!read_band(4, out.inc)) {
    GDALClose(ds);
    return false;
  }

  GDALClose(ds);
  return true;
}

} // namespace tools

int main(int argc, char *argv[]) {
  try {
    if (argc <= 1) {
      tools::print_usage(argv[0]);
      return 1;
    }
    const tools::Options opts = tools::parse_args(argc, argv);
    const std::string input_cub_file = opts.input_cub_path;
    // 输出文件名改为输入文件名（去除后缀）+ "-albedo.tiff"
    const std::string output_tiff_file =
        opts.output_dir + "/" +
        input_cub_file.substr(input_cub_file.find_last_of("/\\") + 1,
                              input_cub_file.find_last_of(".") -
                                  input_cub_file.find_last_of("/\\") - 1) +
        "-albedo.tiff";
    const std::string hapke_config_file = opts.config_path;

    // 加载Hapke配置
    std::string error_message;
    dvs::Config config(hapke_config_file, error_message);
    if (!error_message.empty()) {
      throw std::runtime_error("failed to load config: " + error_message);
      return 1;
    }
    // 利用Gdal加载cub文件，获取inc_angles、emi_angles、phase_angles等数据
    tools::CubBands cub_bands;
    if (!tools::load_cub_file(input_cub_file, cub_bands, error_message)) {
      std::cerr << "Error loading cub file: " << error_message << std::endl;
      return 1;
    }

    // 将数据上传到GPU，调用batch_compute_albedo核函数进行计算
    float *d_inc_angles, *d_emi_angles, *d_phase_angles, *d_albedos,
        *d_i_over_f;
    const std::size_t pixel_count = static_cast<std::size_t>(cub_bands.width) *
                                    static_cast<std::size_t>(cub_bands.height);
    std::cout << "Dimensions: (" << cub_bands.width << ", " << cub_bands.height
              << ") = " << pixel_count << "\n";
    std::cout << "device memory needed approximately = "
              << static_cast<double>(pixel_count) * 5.0 * sizeof(float) /
                     1024.0 / 1024.0 / 1024.0
              << " GiB\n";
    cudaMalloc(&d_inc_angles, pixel_count * sizeof(float));
    cudaMalloc(&d_emi_angles, pixel_count * sizeof(float));
    cudaMalloc(&d_phase_angles, pixel_count * sizeof(float));
    cudaMalloc(&d_i_over_f, pixel_count * sizeof(float));
    cudaMalloc(&d_albedos, pixel_count * sizeof(float));

    cudaMemcpy(d_inc_angles, cub_bands.inc.data(), pixel_count * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_emi_angles, cub_bands.emi.data(), pixel_count * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_phase_angles, cub_bands.phase.data(),
               pixel_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_over_f, cub_bands.i_over_f.data(),
               pixel_count * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_albedos, 0, pixel_count * sizeof(float));

    const int threads_per_block = 256;
    const int blocks = static_cast<int>((pixel_count + threads_per_block - 1) /
                                        threads_per_block);

    tools::batch_compute_albedo<<<blocks, threads_per_block>>>(
        pixel_count, d_inc_angles, d_emi_angles, d_phase_angles, d_i_over_f,
        config.hapke, d_albedos);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    std::cerr << "[CUDA Kernel]: " << cudaGetErrorString(err) << "\n";
    // 将计算结果从GPU复制回CPU
    std::vector<float> albedos(pixel_count);
    cudaMemcpy(albedos.data(), d_albedos, pixel_count * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_inc_angles);
    cudaFree(d_emi_angles);
    cudaFree(d_phase_angles);
    cudaFree(d_i_over_f);
    cudaFree(d_albedos);

    // ============================================================
    // 利用 GDAL 将 albedo 保存为 GeoTIFF
    // ============================================================
    GDALAllRegister();

    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) {
      std::cerr << "Error: GTiff driver not available" << std::endl;
      return 1;
    }

    char **creation_options = nullptr;
    creation_options = CSLSetNameValue(creation_options, "TILED", "YES");
    creation_options = CSLSetNameValue(creation_options, "COMPRESS", "LZW");
    creation_options = CSLSetNameValue(creation_options, "BIGTIFF", "IF_SAFER");

    GDALDataset *out_ds =
        driver->Create(output_tiff_file.c_str(), cub_bands.width,
                       cub_bands.height, 1, GDT_Float32, creation_options);

    CSLDestroy(creation_options);

    if (!out_ds) {
      std::cerr << "Error: failed to create output TIFF file" << std::endl;
      return 1;
    }

    // ============================================================
    // 写入 GeoTransform
    // ============================================================
    if (cub_bands.has_geo_transform) {
      CPLErr gt_status = out_ds->SetGeoTransform(cub_bands.geo_transform);
      if (gt_status != CE_None) {
        std::cerr << "[GDAL] Warning: failed to set GeoTransform to TIFF."
                  << std::endl;
      } else {
        std::cout << "[GDAL] GeoTransform copied to output TIFF.\n";
      }
    } else {
      std::cerr << "[GDAL] Warning: no GeoTransform available, output TIFF "
                   "will not be georeferenced by affine transform."
                << std::endl;
    }

    // ============================================================
    // 写入投影坐标系 WKT
    // 这里使用 GDAL 官方 SetProjection API
    // ============================================================
    if (cub_bands.has_projection && !cub_bands.projection_wkt.empty()) {
      CPLErr proj_status =
          out_ds->SetProjection(cub_bands.projection_wkt.c_str());

      if (proj_status != CE_None) {
        std::cerr << "[GDAL] Warning: failed to set projection to TIFF."
                  << std::endl;
      } else {
        std::cout << "[GDAL] Projection copied to output TIFF.\n";
      }
    } else {
      std::cerr << "[GDAL] Warning: no projection available, output TIFF "
                   "will not contain CRS."
                << std::endl;
    }

    GDALRasterBand *out_band = out_ds->GetRasterBand(1);
    out_band->SetDescription("Albedo");
    out_band->SetNoDataValue(0);
    CPLErr status = out_band->RasterIO(
        GF_Write, 0, 0, cub_bands.width, cub_bands.height, albedos.data(),
        cub_bands.width, cub_bands.height, GDT_Float32, 0, 0);
    if (status != CE_None) {
      std::cerr << "Error: failed to write albedo data to TIFF file"
                << std::endl;
      GDALClose(out_ds);
      return 1;
    }
    out_band->FlushCache();
    GDALClose(out_ds);

    std::cout << "Albedo map saved to: " << output_tiff_file << std::endl;
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}