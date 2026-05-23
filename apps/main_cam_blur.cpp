#include <cstddef>
#include <cstdlib>
#include <ctime>
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

    if (std::filesystem::exists(opts.dem_path) &&
        std::filesystem::exists(opts.albedo_path) &&
        std::filesystem::exists(opts.traj_path)) {
      if (!dvs::load_scene_bundle(opts.dem_path, opts.albedo_path,
                                  opts.traj_path, scene, frames,
                                  &error_message)) {
        throw std::runtime_error("failed to load scene: " + error_message);
        return 1;
      }
    } else {
      if (!dvs::load_scene_bundle(opts.scene_dir, scene, frames,
                                  &error_message)) {
        throw std::runtime_error("failed to load scene bundle: " +
                                 error_message);
        return 1;
      }
    }

    std::cout << "Loaded scene bundle from " << opts.scene_dir << "\n";

    // 使用样条曲线获取连续位姿轨迹
    dvs::CameraTrajectory traj(frames, config.scene.fps);
    dvs::BVHBuildParams params; // 可使用默认参数
    int root_index =
        dvs::BVHBuilder::build(params, scene.bvh_nodes, scene.bvh_triangles);

    dvs::CamBlurRayCaster raycaster(scene, config, root_index);
    std::filesystem::path image_output_dir(opts.output_dir + "/blur_images");
    if (!std::filesystem::exists(image_output_dir)) {
      std::filesystem::create_directories(image_output_dir);
    }

    std::size_t total_images = 0;

    size_t frame_count =
        std::min(frames.size(), static_cast<size_t>(config.scene.frame_count));
    float end_time = static_cast<float>(frames[frame_count - 1].frame_index) /
                     config.scene.fps;
    float start_time =
        static_cast<float>(frames.front().frame_index) / config.scene.fps;
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
      float3 camera_origin = dvs::vec3_to_float3(state.position);
      const float *camera_world = state.orientation.data();
      float3 light_dir = dvs::vec3_to_float3(state.light_direction);
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
