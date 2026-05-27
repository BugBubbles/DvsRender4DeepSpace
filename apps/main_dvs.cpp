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
    dvs::CameraTrajectory traj(frames, config.scene);
    dvs::BVHBuildParams params; // 可使用默认参数
    int root_index = dvs::BVHBuilder::build(
        params, scene.bvh_nodes, scene.bvh_triangles, scene.vertices);

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

    size_t frame_count =
        std::min(frames.size(), static_cast<size_t>(config.scene.frame_count));
    float end_time = static_cast<float>(frames[frame_count - 1].frame_index) /
                     config.scene.fps;
    float start_time =
        static_cast<float>(frames.front().frame_index) / config.scene.fps;
    std::size_t total_steps = static_cast<std::size_t>(
        std::round((end_time - start_time) / config.scene.time_resolution));

    dvs::ProgressBar bar(total_steps, "Generating events");
    std::vector<dvs::Event> events, events_buffer; // 用于存储当前时间步的事件
                                                   //
    float next_write_time = start_time;
    const float write_interval = 1.0f / config.scene.imu_fps;

    // 渲染第一帧以获取平均光强度，以初始化事件图
    const dvs::CameraTrajectoryState &state = traj.evaluate(start_time);
    float3 camera_origin = dvs::vec3_to_float3(state.position);
    const float *camera_world = state.orientation.data();
    float3 light_dir = dvs::vec3_to_float3(state.light_direction);
    float3 linear_velocity = dvs::vec3_to_float3(state.linear_velocity);
    float3 angular_velocity = dvs::vec3_to_float3(state.angular_velocity);
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
      camera_origin = dvs::vec3_to_float3(state.position);
      camera_world = state.orientation.data();
      light_dir = dvs::vec3_to_float3(state.light_direction);
      light_intensity = state.light_intensity;

      raycaster.render_main(camera_origin, camera_world, light_dir,
                            light_intensity, current_time, events);

      // const std::vector<dvs::Event> events =
      // dvs_sensor.process_frame(current_time, light_frame);
      // 将events中的事件移动到events_buffer中，清空events以便下一次使用
      std::move(events.begin(), events.end(),
                std::back_inserter(events_buffer));
      events.clear();
      // 输出当前时间和相机速度作为轨迹信息，精度为9位小数，以imu_fps频率输出
      if (current_time >= next_write_time) {
        next_write_time += write_interval;
        events_out.write(reinterpret_cast<const char *>(events_buffer.data()),
                         events_buffer.size() * sizeof(dvs::Event));
        total_events += events_buffer.size();
        events_buffer.clear();
      }
      bar.update();
    }

    std::cout << "Finished. Wrote " << total_events << " events to "
              << opts.output_dir << "\n";
    events_out.close();
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
