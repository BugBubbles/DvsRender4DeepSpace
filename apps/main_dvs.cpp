#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "bvh/build.h"
#include "dvs/dvs_base.hpp"
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
            << "  --output <path>            output events file (default: ./)\n"
            << "  --config <path>            path to YAML config file "
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

    dvs::DvsRayCaster raycaster(scene, config, root_index);
    std::filesystem::path output_dir(opts.output_dir);
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }
    std::string events_output_path = opts.output_dir + "/events.bin";
    std::ofstream events_out(events_output_path, std::ios::binary);
    if (!events_out) {
      throw std::runtime_error("failed to open output file: " +
                               events_output_path);
    }

    std::size_t total_events = 0;

    size_t frame_count = std::min(
        scene.frames.size(), static_cast<size_t>(config.scene.frame_count));
    float end_time = scene.frames[frame_count - 1].timestamp;
    float start_time = scene.frames.front().timestamp;
    std::size_t total_steps = static_cast<std::size_t>(
        std::round((end_time - start_time) / config.scene.time_resolution));

    dvs::ProgressBar bar(total_steps);
    std::vector<dvs::Event> events, events_buffer; // 用于存储当前时间步的事件
                                                   //
    std::vector<float> time_records(5, 0.0f);      // 用于记录不同阶段的时间
#ifdef DEBUG
    float time_start, time_end;
    int test_id = 0;
#endif
    float next_write_time = start_time;
    const float write_interval = 1.0f / config.scene.imu_fps;

    // 渲染第一帧以获取平均光强度，以初始化事件图
    const dvs::CameraTrajectoryState &state = traj.evaluate(start_time);
    dvs::Vec3 camera_origin = state.position;
    dvs::Mat3 camera_world = state.orientation;
    dvs::Vec3 light_dir = state.light_direction;
    dvs::Vec3 linear_velocity = state.linear_velocity;
    dvs::Vec3 angular_velocity = state.angular_velocity;
    float light_intensity = state.light_intensity;
    std::vector<float> light_frame(config.camera.width * config.camera.height);
    raycaster.render_main(camera_origin, camera_world, light_dir,
                          light_intensity, light_frame);
    float mean_light = 0.0f;
    int valid_pixel_count = 0;
    for (float &v : light_frame) {
      if (v > 0.0f) {
        mean_light += v;
        valid_pixel_count++;
      }
    }
    mean_light /= valid_pixel_count;
    raycaster.initialize(mean_light);
    std::cout << "Initialized event generation with mean light intensity: "
              << mean_light << "\n";

    for (size_t step = 0; step < total_steps; ++step) {
      float current_time = start_time + step * config.scene.time_resolution;
      const dvs::CameraTrajectoryState &state = traj.evaluate(current_time);
      camera_origin = state.position;
      camera_world = state.orientation;
      light_dir = state.light_direction;
      light_intensity = state.light_intensity;

      raycaster.render_main(camera_origin, camera_world, light_dir,
                            light_intensity, current_time, events,
                            time_records.data());

      // const std::vector<dvs::Event> events =
      // dvs_sensor.process_frame(current_time, light_frame);
      // 将events中的事件移动到events_buffer中，清空events以便下一次使用
      std::move(events.begin(), events.end(),
                std::back_inserter(events_buffer));
      events.clear();
      // 输出当前时间和相机速度作为轨迹信息，精度为9位小数，以imu_fps频率输出
      if (current_time >= next_write_time) {
        next_write_time += write_interval;
#ifdef DEBUG
        time_start = static_cast<float>(std::clock()) / CLOCKS_PER_SEC;
#endif
        events_out.write(reinterpret_cast<const char *>(events_buffer.data()),
                         events_buffer.size() * sizeof(dvs::Event));
#ifdef DEBUG
        time_end = static_cast<float>(std::clock()) / CLOCKS_PER_SEC;
        time_records[4] += (time_end - time_start);
        time_start = time_end;
#endif
        total_events += events_buffer.size();
        events_buffer.clear();
      }
      bar.update();
#ifdef DEBUG
      test_id++;
      if (test_id > 100) {
        break; // 只测试前100帧，实际使用时可以去掉这个限制
      }
#endif
    }

    std::cout << "Finished. Wrote " << total_events << " events to "
              << opts.output_dir << "\n";
#ifdef DEBUG
    std::cout << "Time for ray casting: " << time_records[0] << " seconds\n";
    std::cout << "Time for intersecting:" << time_records[1] << " seconds\n";
    std::cout << "Time for shading: " << time_records[2] << " seconds\n";
    std::cout << "Time for event generation: " << time_records[3]
              << " seconds\n";
    std::cout << "Time for writing events: " << time_records[4] << " seconds\n";
#endif
    events_out.close();
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
