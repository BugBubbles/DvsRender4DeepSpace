#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "dvs/dvs_base.hpp"
#include "dvs/progress.hpp"
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
            << "  --output <path>            output directory (default:  ./)\n"
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
    dvs::SceneBundle<float3> scene;
    std::vector<dvs::FrameData<dvs::Vec3, dvs::Mat3>> frames;
    std::string error_message;
    dvs::Config config(opts.config_path, error_message);
    if (!error_message.empty()) {
      throw std::runtime_error("failed to load config: " + error_message);
      return 1;
    }
    if (!dvs::load_scene_bundle(opts.scene_dir, scene, frames,
                                &error_message)) {
      throw std::runtime_error("failed to load scene bundle: " + error_message);
      return 1;
    }

    std::cout << "Loaded scene bundle from " << opts.scene_dir << "\n";

    // 使用样条曲线获取连续位姿轨迹
    dvs::CameraTrajectory traj(frames, config.scene.fps);
    std::string motion_output_path = opts.output_dir + "/gt_motion.txt";
    std::string pose_output_path = opts.output_dir + "/gt_pose.txt";
    // 如果路径不存在，则创建目录
    std::filesystem::path output_dir(opts.output_dir);
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }
    std::ofstream motion_out(motion_output_path);
    std::ofstream pose_out(pose_output_path);
    // 输出标题头
    motion_out << "#ts vx vy vz wx wy wz" << std::endl;
    pose_out << "#ts x y z qx qy qz qw" << std::endl;

    size_t frame_count =
        std::min(frames.size(), static_cast<size_t>(config.scene.frame_count));
    float end_time = frames[frame_count - 1].timestamp;
    float start_time = frames.front().timestamp;
    std::size_t total_steps = static_cast<std::size_t>(
        std::round((end_time - start_time) / config.scene.time_resolution));

    dvs::ProgressBar bar(total_steps);
    std::vector<dvs::Event> events, events_buffer; // 用于存储当前时间步的事件
                                                   //
    float next_write_time = start_time;
    int total_trajs = 0;
    const float write_interval = 1.0f / config.scene.imu_fps;

    for (size_t step = 0; step < total_steps; ++step) {
      float current_time = start_time + step * config.scene.time_resolution;
      const dvs::CameraTrajectoryState &state = traj.evaluate(current_time);
      dvs::Vec3 camera_origin = state.position;
      dvs::Mat3 camera_world = state.orientation;
      // dvs::Vec3 light_dir = state.light_direction;
      dvs::Vec3 linear_velocity = state.linear_velocity;
      dvs::Vec3 angular_velocity = state.angular_velocity;
      // 将Mat3转为四元数
      Eigen::Quaternionf quat(camera_world);
      // float light_intensity = state.light_intensity;

      // 输出当前时间和相机速度作为轨迹信息，精度为9位小数，以imu_fps频率输出
      if (current_time >= next_write_time) {
        next_write_time += write_interval;
        motion_out << std::fixed << std::setprecision(9) << current_time
                   << linear_velocity.transpose() << " "
                   << angular_velocity.transpose() << std::endl;
        pose_out << std::fixed << std::setprecision(9) << current_time
                 << camera_origin.transpose() << " " << quat.x() << " "
                 << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
      }
      total_trajs++;
      bar.update();
    }

    std::cout << "Finished. Wrote " << total_trajs << " poses to "
              << opts.output_dir << "\n";
    motion_out.close();
    pose_out.close();
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
