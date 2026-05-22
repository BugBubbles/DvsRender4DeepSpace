#include "tools/build_from_raster.hpp"
#include "tools/loader.hpp"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace tools {
struct Options {
  std::string dem_path;
  std::string albedo_path;
  std::string output_dir = "../output";
  std::string config_path = "../../config/default.yaml";
};

void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog
      << " --dem_path <path> --albedo_path <path> [options]\n"
      << "Required arguments:\n"
      << "  --dem_path <path>          path to input DEM file (e.g. "
         ".tiff/.img)\n"
      << "  --albedo_path <path>       path to input albedo file (e.g. "
         ".tiff/.img)\n"
      << "Options:\n"
      << "  --output <path>            output images file (default: ./output)\n"
      << "  --config <path>            path to YAML config file (default: "
         "../../config/default.yaml)\n"
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

    if (arg == "--dem_path") {
      opts.dem_path = require_value("--dem_path");
    } else if (arg == "--albedo_path") {
      opts.albedo_path = require_value("--albedo_path");
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

  if (opts.dem_path.empty()) {
    throw std::runtime_error("--input_cub is required");
  }
  if (opts.albedo_path.empty()) {
    throw std::runtime_error("--albedo_path is required");
  }
  std::ifstream config_in(opts.config_path);
  if (!config_in.good()) {
    throw std::runtime_error("failed to open config file: " + opts.config_path);
  }
  return opts;
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
  // if (idx ==
  //     ref_height / 2 * ref_width + ref_width / 2) { // 打印中心点的调试信息
  //   printf("src_lat_idx: (%d, %d), src_lon_idx: (%d, %d)\n", src_lat_idx.x,
  //          src_lat_idx.y, src_lon_idx.x, src_lon_idx.y);
  //   printf("ref_lat: %f, ref_lon: %f\n", ref_lat, ref_lon);
  // }
  if (src_lat_idx.x == -1 || src_lon_idx.x == -1) {
    aligned_src_rst[idx] = 0.0f; // 无效位置，设置为0
    return;
  }
  // if (idx ==
  //     ref_height / 2 * ref_width + ref_width / 2) { // 打印中心点的调试信息
  //   printf("src_lat: (%f, %f), src_lon: (%f, %f)\n", src_lats[src_lat_idx.x],
  //          src_lats[src_lat_idx.y], src_lons[src_lon_idx.x],
  //          src_lons[src_lon_idx.y]);
  // }

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
} // namespace tools
int main(int argc, char *argv[]) {
  try {
    if (argc <= 1) {
      tools::print_usage(argv[0]);
      return 1;
    }
    const tools::Options opts = tools::parse_args(argc, argv);
    const std::string dem_path = opts.dem_path;
    const std::string albedo_path = opts.albedo_path;
    // 输出文件名改为输入文件名（去除后缀）+ "-albedo.tiff"
    const std::string output_tiff_file = opts.output_dir;
    const std::string hapke_config_file = opts.config_path;

    // 加载 DEM 和 albedo 数据
    tools::Imagery dem, albedo;
    if (!tools::loadImageryData(dem_path, dem))
      throw std::runtime_error("failed to load DEM data from: " + dem_path);

    std::cout << "DEM dimensions: " << dem.width << "x" << dem.height
              << std::endl;
    if (!tools::loadImageryData(albedo_path, albedo))
      throw std::runtime_error("failed to load albedo data from: " +
                               albedo_path);

    std::cout << "Albedo dimensions: " << albedo.width << "x" << albedo.height
              << std::endl;

    // Albedo 对齐至 DEM，DEM各个顶点生成三角形和表面法线
    // 将 DEM 和 albedo 数据上传到 GPU，执行对齐和法线计算的 CUDA 内核
    float *d_dem, *d_albedo, *d_aligned_albedo, *d_dem_lats, *d_dem_lons,
        *d_albedo_lats, *d_albedo_lons;
    std::size_t memory_size =
        (dem.width * dem.height + albedo.width * albedo.height +
         dem.width * dem.height + dem.height + dem.width + albedo.height +
         albedo.width) *
        sizeof(float);

    std::cout << "Total GPU memory required: "
              << memory_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::size_t dem_size = dem.width * dem.height;
    std::size_t albedo_size = albedo.width * albedo.height;
    cudaMalloc(&d_dem, dem_size * sizeof(float));
    cudaMalloc(&d_albedo, albedo_size * sizeof(float));
    cudaMalloc(&d_aligned_albedo, dem_size * sizeof(float));
    cudaMalloc(&d_dem_lats, dem.height * sizeof(float));
    cudaMalloc(&d_dem_lons, dem.width * sizeof(float));
    cudaMalloc(&d_albedo_lats, albedo.height * sizeof(float));
    cudaMalloc(&d_albedo_lons, albedo.width * sizeof(float));
    cudaMemcpy(d_dem, dem.data.data(), dem_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dem_lats, dem.lats.data(), dem.height * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dem_lons, dem.lons.data(), dem.width * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_albedo, albedo.data.data(), albedo_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_albedo_lats, albedo.lats.data(), albedo.height * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_albedo_lons, albedo.lons.data(), albedo.width * sizeof(float),
               cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((dem.width + blockSize.x - 1) / blockSize.x,
                  (dem.height + blockSize.y - 1) / blockSize.y);
    tools::batch_align_kernel<<<gridSize, blockSize>>>(
        d_dem, d_dem_lats, d_dem_lons, dem.width, dem.height, d_albedo,
        d_albedo_lats, d_albedo_lons, albedo.width, albedo.height,
        d_aligned_albedo);
    cudaDeviceSynchronize();
    // 将对齐后的 albedo 数据从 GPU 复制回 CPU，并保存为 TIFF 文件
    cudaMemcpy(albedo.data.data(), d_aligned_albedo, dem_size * sizeof(float),
               cudaMemcpyDeviceToHost);
    float3 *d_vertices, *d_normals;
    int3 *d_triangles;
    cudaMalloc(&d_vertices, dem_size * sizeof(float3));
    cudaMalloc(&d_normals, dem_size * sizeof(float3));
    cudaMalloc(&d_triangles,
               (dem.width - 1) * (dem.height - 1) * 2 * sizeof(int3));
    cudaMemset(d_normals, 0, dem_size * sizeof(float3));
    int *d_vertice_count, *d_triangle_count;
    cudaMalloc(&d_vertice_count, sizeof(int));
    cudaMalloc(&d_triangle_count, sizeof(int));
    cudaMemset(d_vertice_count, 0, sizeof(int));
    cudaMemset(d_triangle_count, 0, sizeof(int));

    tools::build_from_raster(d_dem, d_dem_lats, d_dem_lons, dem.radii,
                             dem.width, dem.height, d_vertices, d_vertice_count,
                             d_triangles, d_triangle_count, d_normals);

    int h_vertice_count, h_triangle_count;
    cudaMemcpy(&h_vertice_count, d_vertice_count, sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_triangle_count, d_triangle_count, sizeof(int),
               cudaMemcpyDeviceToHost);
    std::cout << "Generated " << h_vertice_count << " vertices and "
              << h_triangle_count << " triangles." << std::endl;

    // 将顶点、法线和三角形数据从 GPU 复制回 CPU，并保存为 OBJ 文件
    std::vector<float3> vertices(h_vertice_count);
    std::vector<float3> normals(h_vertice_count);
    std::vector<int3> triangles(h_triangle_count);
    cudaMemcpy(vertices.data(), d_vertices, h_vertice_count * sizeof(float3),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(normals.data(), d_normals, h_vertice_count * sizeof(float3),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(triangles.data(), d_triangles, h_triangle_count * sizeof(int3),
               cudaMemcpyDeviceToHost);
    cudaFree(d_dem);
    cudaFree(d_albedo);
    cudaFree(d_aligned_albedo);
    cudaFree(d_dem_lats);
    cudaFree(d_dem_lons);
    cudaFree(d_albedo_lats);
    cudaFree(d_albedo_lons);
    cudaFree(d_vertices);
    cudaFree(d_normals);
    cudaFree(d_triangles);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}