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
#include "tools/cli.hpp"

int main(int argc, char **argv) {
  try {
    if (argc <= 1) {
      tools::Options::print_usage(argv[0]);
      return 1;
    }

    const tools::Options opts(argc, argv);

    dvs::SceneBundle<float3, uint3> scene;
    std::vector<dvs::FrameData<dvs::Vec3, dvs::Mat3>> frames;
    std::string error_message;
    dvs::Config config(opts.config_path, error_message);
    if (!error_message.empty()) {
      throw std::runtime_error("failed to load config: " + error_message);
      return 1;
    }

    if (std::filesystem::exists(opts.dem_path) &&
        std::filesystem::exists(opts.traj_path)) {
      if (std::filesystem::exists(opts.albedo_path)) {
        if (!dvs::load_scene_bundle(opts.dem_path, opts.albedo_path,
                                    opts.traj_path, scene, frames,
                                    &error_message)) {
          throw std::runtime_error("failed to load scene: " + error_message);
          return 1;
        }
      } else {
        if (!dvs::load_scene_bundle(opts.dem_path, opts.traj_path, scene,
                                    frames, &error_message)) {
          throw std::runtime_error("failed to load scene: " + error_message);
          return 1;
        }
      }
    } else {
      if (!dvs::load_scene_bundle(opts.scene_dir, scene, frames,
                                  &error_message)) {
        throw std::runtime_error("failed to load scene bundle: " +
                                 error_message);
        return 1;
      }
    }

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
    float end_time = static_cast<float>(frames[frame_count - 1].frame_index) /
                     config.scene.fps;
    float start_time =
        static_cast<float>(frames.front().frame_index) / config.scene.fps;
    std::size_t total_steps = static_cast<std::size_t>(
        std::round((end_time - start_time) / config.scene.time_resolution));

    dvs::ProgressBar bar(total_steps, "Generating GT Pose");
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
