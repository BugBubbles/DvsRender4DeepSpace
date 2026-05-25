#include "bvh/build.h"
#include "dvs/math_types.hpp"
#include "tools/build_from_raster.hpp"
#include "tools/cli.hpp"
#include "tools/loader.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace tools {
class BakeOptions : public tools::Options {
public:
  BakeOptions(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto require_value = [&](const char *key) -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error(std::string("missing value for ") + key);
        }
        return std::string(argv[++i]);
      };

      if (arg == "--scene_dir") {
        scene_dir = require_value("--scene_dir");
      } else if (arg == "--dem") {
        dem_path = require_value("--dem");
      } else if (arg == "--albedo") {
        albedo_path = require_value("--albedo");
      } else if (arg == "--traj") {
        traj_path = require_value("--traj");
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

    if (dem_path.empty() || albedo_path.empty()) {
      throw std::runtime_error("both --dem and --albedo are required");
    }
    std::ifstream config_in(config_path);
    if (!config_in.good()) {
      throw std::runtime_error("failed to open config file: " + config_path);
    }
  }
};
} // namespace tools
int main(int argc, char *argv[]) {
  try {
    if (argc <= 1) {
      tools::BakeOptions::print_usage(argv[0]);
      return 1;
    }
    const tools::BakeOptions opts(argc, argv);
    const std::string dem_path = opts.dem_path;
    const std::string albedo_path = opts.albedo_path;
    // 输出文件名改为输入文件名（去除后缀）+ "-albedo.tiff"
    const std::string output_dir = opts.output_dir;
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
    float *d_dem, *d_albedo, *d_aligned_albedo, *d_dem_xs, *d_dem_ys,
        *d_albedo_xs, *d_albedo_ys;
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
    cudaMemcpy(d_albedo_ys, albedo.y_coords.data(),
               albedo.height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_albedo_xs, albedo.x_coords.data(),
               albedo.width * sizeof(float), cudaMemcpyHostToDevice);
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
    std::cout << "Generated " << h_vertice_count << " vertices and "
              << h_triangle_count << " triangles." << std::endl;
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
    float4 rotation =
        make_float4(0.0f, 0.0f, 0.0f, 1.0f); // 单位四元数表示无旋转

    // 对各个顶点作刚体变换
    int thread_count = 256;
    int vertex_grid_size = (h_vertice_count + thread_count - 1) / thread_count;
    tools::apply_rigid_transform<<<vertex_grid_size, thread_count>>>(
        d_vertices, h_vertice_count, translation, rotation);
    tools::apply_rigid_transform<<<vertex_grid_size, thread_count>>>(
        d_normals, h_vertice_count, make_float3(0.0f, 0.0f, 0.0f), rotation);
    // 修改为当前dvs版本的三角形数据结构
    // dvs::Triangle<float3> *d_tris;
    // cudaMalloc(&d_tris, h_triangle_count * sizeof(dvs::Triangle<float3>));
    // int tri_grid_size = (h_triangle_count + thread_count - 1) / thread_count;
    // tools::trianglize<<<tri_grid_size, thread_count>>>(
    //     d_vertices, d_normals, d_triangles, h_triangle_count, d_tris);

    // 将顶点、法线、albedo 从 GPU 复制回 CPU，并保存为输出文件
    std::vector<float3> h_vertices(h_vertice_count);
    std::vector<float3> h_normals(h_vertice_count);
    std::vector<int3> h_triangles(h_triangle_count);
    cudaMemcpy(h_vertices.data(), d_vertices, h_vertice_count * sizeof(float3),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_normals.data(), d_normals, h_vertice_count * sizeof(float3),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_triangles.data(), d_triangles, h_triangle_count * sizeof(int3),
               cudaMemcpyDeviceToHost);
    cudaFree(d_vertices);
    cudaFree(d_normals);
    cudaFree(d_triangles);

    // 输出变换后的第一个顶点与法线，检查是否正确应用了刚体变换
    std::cout << "First vertex after transform: (" << h_vertices[0].x << ", "
              << h_vertices[0].y << ", " << h_vertices[0].z << ")" << std::endl;
    std::cout << "First normal after transform: (" << h_normals[0].x << ", "
              << h_normals[0].y << ", " << h_normals[0].z << ")" << std::endl;

    // std::vector<dvs::Triangle<float3>> h_tris(h_triangle_count);
    std::vector<float> h_albs(h_vertice_count);
    // cudaMemcpy(h_tris.data(), d_tris,
    //            h_triangle_count * sizeof(dvs::Triangle<float3>),
    //            cudaMemcpyDeviceToHost);
    cudaMemcpy(h_albs.data(), d_albs, h_vertice_count * sizeof(float),
               cudaMemcpyDeviceToHost);
    // cudaFree(d_tris);
    cudaFree(d_albs);
    std::cout << "Converted triangles to DVS format." << std::endl;

    // dvs::BVHBuildParams params; // 可使用默认参数
    // std::vector<dvs::BVHNode<float3>> bvh_nodes;
    // int root_index = dvs::BVHBuilder::build(params, bvh_nodes, h_tris);

    // std::cout << "BVH root_index_ = " << root_index
    //           << ", root count = " << bvh_nodes[root_index].count
    //           << ", left = " << bvh_nodes[root_index].left
    //           << ", right = " << bvh_nodes[root_index].right << std::endl;
    // dvs::BVHBuilder::validate(bvh_nodes, root_index);
    // std::cout << "BVH validated: " << root_index << " nodes, "
    //           << bvh_nodes[root_index].count << " triangles." << std::endl;
    // 创建文件夹
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }
    // 将三角形数据和顶点 albedo 保存为输出文件
    std::string indices_output_path = output_dir + "/mesh_indices.bin";
    std::string vertices_output_path = output_dir + "/mesh_vertices.bin";
    std::string normals_output_path = output_dir + "/mesh_normals.bin";
    std::string albedo_output_path = output_dir + "/mesh_albedo.bin";
    std::ofstream indices_out(indices_output_path, std::ios::binary);
    std::ofstream vertices_out(vertices_output_path, std::ios::binary);
    std::ofstream normals_out(normals_output_path, std::ios::binary);
    std::ofstream albedo_out(albedo_output_path, std::ios::binary);
    if (!indices_out.good() || !vertices_out.good() || !normals_out.good() ||
        !albedo_out.good()) {
      throw std::runtime_error("failed to open output files in: " + output_dir);
    }
    for (const auto &tri : h_triangles) {
      indices_out.write(reinterpret_cast<const char *>(&tri), sizeof(int3));
    }
    indices_out.close();
    for (const auto &vert : h_vertices) {
      vertices_out.write(reinterpret_cast<const char *>(&vert), sizeof(float3));
    }
    vertices_out.close();
    for (const auto &normal : h_normals) {
      normals_out.write(reinterpret_cast<const char *>(&normal),
                        sizeof(float3));
    }
    normals_out.close();
    for (const auto &alb : h_albs) {
      albedo_out.write(reinterpret_cast<const char *>(&alb), sizeof(float));
    }
    albedo_out.close();

    std::cout << "Saved output files to: " << output_dir << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}