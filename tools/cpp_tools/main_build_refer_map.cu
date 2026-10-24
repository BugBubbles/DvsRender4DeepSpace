#include "dvs/hapke.hpp"
#include "dvs/scene_types.hpp"
#include "tools/build_from_raster.hpp"
#include "tools/cli.hpp"
#include "tools/loader.hpp"

#include <algorithm>
#include <charconv>
#include <cpl_conv.h>
#include <filesystem>
#include <gdal_priv.h>
#include <gdal_rat.h>
#include <gdal_version.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

namespace tools {
class BuildOptions : public tools::Options {
public:
  std::string dom_path;
  int window_size = 512;
  static void print_usage(const char *prog) {
    std::cout
        << "Usage: " << prog << "[options]\n"
        << "Options:\n"
        << "  --dem <path>               path to DEM file\n"
        << "  --dom <path>               path to DOM file\n"
        << "(both --dem and --dom are required)\n"
        << "  --window_size <size>       window size for processing (default: "
           "512)\n"
        << "  --output <path>            output directory (default: ./)\n"
        << "  --config <path>            path to YAML config file (default: "
           "../../config/default.yaml)\n"
        << std::endl;
  }
  BuildOptions(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto require_value = [&](const char *key) -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error(std::string("missing value for ") + key);
        }
        return std::string(argv[++i]);
      };

      if (arg == "--dem") {
        dem_path = require_value("--dem");
      } else if (arg == "--dom") {
        dom_path = require_value("--dom");
      } else if (arg == "--window_size") {
        auto value=require_value("--window_size");
        auto parsed=std::from_chars(value.data(),value.data()+value.size(),window_size);
        if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size() || window_size<=0)
          throw std::runtime_error("window_size must be a positive integer");
      } else if (arg == "--output") {
        output_dir = require_value("--output");
      } else if (arg == "--config") {
        config_path = require_value("--config");
      } else if (arg == "--help" || arg == "-h") {
        print_usage(argv[0]);
        std::exit(0);
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }

    if (dem_path.empty() || dom_path.empty()) {
      throw std::runtime_error("both --dem and --dom are required");
    }
    std::ifstream config_in(config_path);
    if (!config_in.good()) {
      throw std::runtime_error("failed to open config file: " + config_path);
    }
  }
};
} // namespace tools

int main(int argc, char **argv) {
  try {
    if (argc <= 1) {
      tools::BuildOptions::print_usage(argv[0]);
      return 1;
    }
    tools::BuildOptions opts(argc, argv);
    tools::Imagery dem,dom;
    if(!tools::loadImageryData(opts.dem_path,dem) || !tools::loadImageryData(opts.dom_path,dom))
      throw std::runtime_error("failed to load DEM/DOM");
    tools::require_same_projection(dem,dom);
    const double *transform_dem=dem.transform;
    // 如果同时加载了DEM和DOM，将 DOM 对齐至 DEM
    // 将 DEM 和 dom 数据上传到 GPU，执行对齐和法线计算的 CUDA 内核
    float *d_dem, *d_dom, *d_aligned_dom, *d_dem_xs, *d_dem_ys, *d_dom_xs,
        *d_dom_ys;
    std::size_t memory_size = (dem.width * dem.height + dom.width * dom.height +
                               dem.width * dem.height + dem.height + dem.width +
                               dom.height + dom.width) *
                              sizeof(float);

    std::cout << "Total GPU memory required: "
              << memory_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::size_t dem_size = dem.width * dem.height;
    std::size_t dom_size = dom.width * dom.height;
    cudaMalloc(&d_dem, dem_size * sizeof(float));
    cudaMalloc(&d_dom, dom_size * sizeof(float));
    cudaMalloc(&d_aligned_dom, dem_size * sizeof(float));
    cudaMalloc(&d_dem_ys, dem.height * sizeof(float));
    cudaMalloc(&d_dem_xs, dem.width * sizeof(float));
    cudaMalloc(&d_dom_ys, dom.height * sizeof(float));
    cudaMalloc(&d_dom_xs, dom.width * sizeof(float));
    cudaMemcpy(d_dem, dem.data.data(), dem_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dem_ys, dem.y_coords.data(), dem.height * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dem_xs, dem.x_coords.data(), dem.width * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dom, dom.data.data(), dom_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dom_ys, dom.y_coords.data(), dom.height * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dom_xs, dom.x_coords.data(), dom.width * sizeof(float),
               cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((dem.width + blockSize.x - 1) / blockSize.x,
                  (dem.height + blockSize.y - 1) / blockSize.y);
    tools::batch_align_kernel<<<gridSize, blockSize>>>(
        d_dem, d_dem_ys, d_dem_xs, dem.width, dem.height, d_dom, d_dom_ys,
        d_dom_xs, dom.width, dom.height, d_aligned_dom);
    // 将 GPU 上对齐后的 DOM 数据下载回 CPU，保存为 GeoTIFF 文件
    dom.data.resize(dem_size);
    cudaMemcpy(dom.data.data(), d_aligned_dom, dem_size * sizeof(float),
               cudaMemcpyDeviceToHost);
    dom.width = dem.width;
    dom.height = dem.height;
    // 释放 GPU 内存
    cudaFree(d_dem);
    cudaFree(d_dom);
    cudaFree(d_aligned_dom);
    cudaFree(d_dem_ys);
    cudaFree(d_dem_xs);
    cudaFree(d_dom_ys);
    cudaFree(d_dom_xs);
    // 输出图像和文件的路径
    if (!std::filesystem::exists(opts.output_dir)) {
      std::filesystem::create_directories(opts.output_dir);
    }
    std::string dem_image_dir =
        std::filesystem::path(opts.output_dir) / "dem_images";
    std::string dom_image_dir =
        std::filesystem::path(opts.output_dir) / "dom_images";
    std::string labels_path =
        std::filesystem::path(opts.output_dir) / "labels.txt";
    if (!std::filesystem::exists(dem_image_dir)) {
      std::filesystem::create_directories(dem_image_dir);
    }
    if (!std::filesystem::exists(dom_image_dir)) {
      std::filesystem::create_directories(dom_image_dir);
    }

    // 按 window_size 切分影像，将 DEM 和对齐后的 DOM 分别写入为对应的文件中
    std::ofstream labels_file(labels_path);
    if (!labels_file.is_open()) {
      std::cerr << "Failed to open labels file: " << labels_path << std::endl;
      return 1;
    }
    // 写入标题头
    labels_file << "dem_image_path dom_image_path world_x reso_x row_rot "
                   "world_y reso_y col_rot offset_z"
                << std::endl;
    float offset_x = dem.x_coords[dem.width / 2];
    float offset_y = dem.y_coords[dem.height / 2];
    float offset_z = tools::reference_height(dem);
    // 在世界空间中，整个DEM已经被整体平移了(-offset_x, -offset_y,
    // -offset_z)，因此中心点的坐标应该是(0, 0, 0)
    std::cout << "DEM center x: " << offset_x << ", y: " << offset_y
              << ", depth: " << offset_z << std::endl;
    for (int y = 0; y < dem.height; y += opts.window_size) {
      for (int x = 0; x < dem.width; x += opts.window_size) {
        int window_width = std::min(opts.window_size, dem.width - x);
        int window_height = std::min(opts.window_size, dem.height - y);
        std::string idx_name = std::to_string(x) + "_" + std::to_string(y);
        std::string dem_window_path =
            std::filesystem::path(dem_image_dir) / (idx_name + ".tif");
        std::string dom_window_path =
            std::filesystem::path(dom_image_dir) / (idx_name + ".tif");

        cv::Mat dem_window(window_height, window_width, CV_32F);
        cv::Mat dom_window(window_height, window_width, CV_32F);
        for (int j = 0; j < window_height; ++j) {
          for (int i = 0; i < window_width; ++i) {
            dem_window.at<float>(j, i) =
                dem.data[(y + j) * dem.width + (x + i)];
            dom_window.at<float>(j, i) =
                dom.data[(y + j) * dom.width + (x + i)];
          }
        }
        cv::imwrite(dem_window_path, dem_window);
        cv::imwrite(dom_window_path, dom_window);
        // 计算当前窗口的左上角在世界空间中的坐标，作为当前窗口的投影坐标
        float world_x = (x + 0.5) * transform_dem[1] + transform_dem[0] +
                        0.5 * transform_dem[2] - offset_x;
        float world_y = (y + 0.5) * transform_dem[5] + transform_dem[3] +
                        0.5 * transform_dem[4] - offset_y;
        // 写入文件的投影坐标信息，从左到右分别是 offset_x	reso_x	row_rot
        // offset_y	reso_y	col_rot	offset_z
        // 设置输出浮点数精度为 6 位小数，不要使用科学计数法
        labels_file << "dem_images/" + idx_name + ".tif" << " "
                    << "dom_images/" + idx_name + ".tif" << " " << std::fixed
                    << std::setprecision(6) << world_x << " "
                    << transform_dem[1] << " " << transform_dem[2] << " "
                    << world_y << " " << transform_dem[5] << " "
                    << transform_dem[4] << " " << -offset_z << std::endl;
        labels_file.flush();
      }
    }
    labels_file.close();
    std::cout << "Processing completed successfully." << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}