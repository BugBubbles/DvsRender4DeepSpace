#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <sys/types.h>

#include "bvh/build.h"
#include "dvs/blenderize.hpp"
#include "dvs/progress.hpp"
#include "dvs/raycast.hpp"
#include "dvs/scene_data.hpp"
#include "dvs/scene_types.hpp"
#include "dvs/trajectory.hpp"

namespace {

struct Options {
  std::string scene_dir;
  std::string output_dir = "./output";
  std::string config_path = "./config/default.yaml";
  float end_time = std::numeric_limits<float>::max();
};

void print_usage(const char *prog) {
  std::cout << "Usage: " << prog << " --scene_dir <bundle_dir> [options]\n"
            << "Options:\n"
            << "  --output <path>            output images file (default: ./)\n"
            << "  --end-time <seconds>       end time for simulation (default: "
               "until end of scene)\n"
            << "--config <path>             path to YAML config file "
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

    if (arg == "--scene_dir") {
      opts.scene_dir = require_value("--scene_dir");
    } else if (arg == "--end-time") {
      opts.end_time =
          parse_float_arg("--end-time", require_value("--end-time"));
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

  if (opts.scene_dir.empty()) {
    throw std::runtime_error("--scene_dir is required");
  }
  std::ifstream config_in(opts.config_path);
  if (!config_in.good()) {
    throw std::runtime_error("failed to open config file: " + opts.config_path);
  }
  return opts;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc <= 1) {
      print_usage(argv[0]);
      return 1;
    }

    const Options opts = parse_args(argc, argv);
    dvs::SceneBundle<dvs::Vec3, dvs::Uint3, dvs::Mat3> scene;
    std::string error_message;
    dvs::Config config(opts.config_path, error_message);
    if (!error_message.empty()) {
      throw std::runtime_error("failed to load config: " + error_message);
      return 1;
    }
    if (!dvs::load_scene_bundle(opts.scene_dir, &scene, &error_message)) {
      throw std::runtime_error("failed to load scene bundle: " + error_message);
      return 1;
    }

    std::cout << "Loaded scene bundle from " << opts.scene_dir << "\n";

    // 使用样条曲线获取连续位姿轨迹
    dvs::CameraTrajectory traj(scene.frames, config.scene.fps);
    dvs::BVHBuildParams params; // 可使用默认参数
    int root_index =
        dvs::BVHBuilder::build(params, scene.bvh_nodes, scene.bvh_triangles);

    dvs::CamBlurRayCaster raycaster(scene, config, root_index);
    std::filesystem::path image_output_dir(opts.output_dir + "/blur_images");
    if (!std::filesystem::exists(image_output_dir)) {
      std::filesystem::create_directories(image_output_dir);
    }

    std::size_t total_images = 0;

    float end_time = std::min(opts.end_time, scene.frames.back().timestamp);
    float start_time = scene.frames.front().timestamp;
    std::size_t total_steps = static_cast<std::size_t>(
        std::round((end_time - start_time) / config.scene.blur_resolution));
    float step_time = config.scene.blur_resolution;

    dvs::ProgressBar bar(total_steps);
    const float write_interval = 1.0f / config.scene.cam_fps;
    float next_write_time = start_time + write_interval;

    int num_blur = static_cast<int>(
        std::round(1.0f / config.scene.blur_resolution / config.scene.cam_fps));
    std::vector<float> light_frame_step_float(config.camera.width *
                                              config.camera.height);
    cv::Mat light_frame_step_float_add(config.camera.height,
                                       config.camera.width, CV_32F);
    cv::Mat light_frame_float(config.camera.height, config.camera.width,
                              CV_32F);
    light_frame_float.setTo(0.0f);

    std::vector<u_int8_t> light_frame_uint8(config.camera.width *
                                            config.camera.height);
    for (size_t step = 0; step < total_steps; ++step) {
      float current_time = start_time + step * step_time;
      const dvs::CameraTrajectoryState &state = traj.evaluate(current_time);
      dvs::Vec3 camera_origin = state.position;
      dvs::Mat3 camera_world = state.orientation;
      dvs::Vec3 light_dir = state.light_direction;
      float light_intensity = state.light_intensity;
      raycaster.render_main(camera_origin, camera_world, light_dir,
                            light_intensity, light_frame_step_float);

      // 将当前step的结果累加到light_frame_float中
      light_frame_step_float_add =
          cv::Mat(config.camera.height, config.camera.width, CV_32F,
                  light_frame_step_float.data());
      light_frame_float += light_frame_step_float_add;
      if (current_time >= next_write_time) {
        next_write_time += write_interval;
        light_frame_float /= num_blur; // 取平均值
        // 输出当前渲染所得相机图像，精度为9位小数，以cam_fps频率输出
        // 使用OpenCV将light_frame保存为图像
        std::string image_name =
            image_output_dir / (std::to_string(current_time) + ".png");
        dvs::render_float_to_png_bytes(
            light_frame_float.ptr<float>(), config.camera.width,
            config.camera.height, 0.0, light_frame_uint8);
        cv::Mat img_uint8(config.camera.height, config.camera.width, CV_8U,
                          light_frame_uint8.data());
        // // 归一化图像到0-1范围
        cv::imwrite(image_name, img_uint8);

        total_images++;
        light_frame_float.setTo(0.0f);
      }
      bar.update();
    }

    std::cout << "Finished. Wrote " << total_images << " images to "
              << opts.output_dir << "\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
