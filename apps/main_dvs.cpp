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
      dvs::Options::print_usage(argv[0]);
      return 1;
    }

    const dvs::Options opts(argc, argv);
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

    size_t frame_count =
        std::min(frames.size(), static_cast<size_t>(config.scene.frame_count));
    float end_time = frames[frame_count - 1].timestamp;
    float start_time = frames.front().timestamp;
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
