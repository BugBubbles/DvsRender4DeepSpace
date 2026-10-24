#include "app_inputs.hpp"
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <sys/types.h>

#include "bvh/build.h"
#include "dvs/blenderize.hpp"
#include "dvs/math_types.hpp"
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

    dvs::load_app_inputs(opts,config,scene,frames);

    // 使用样条曲线获取连续位姿轨迹
    dvs::CameraTrajectory traj(frames, config.scene);
    dvs::BVHBuildParams params;params.max_leaf_size=64; // full supplied terrain fits GPU memory
    int root_index = dvs::BVHBuilder::build(
        params, scene.bvh_nodes, scene.bvh_triangles, scene.vertices);

    dvs::CamRayCaster raycaster(scene, config, root_index);
    std::filesystem::path image_output_dir(opts.output_dir + "/images");
    std::filesystem::path depth_output_dir(opts.output_dir + "/depth");
    if (!std::filesystem::exists(image_output_dir)) {
      std::filesystem::create_directories(image_output_dir);
    }
    if (!std::filesystem::exists(depth_output_dir)) {
      std::filesystem::create_directories(depth_output_dir);
    }

    std::size_t total_images = 0;

    size_t frame_count =
        std::min(frames.size(), static_cast<size_t>(config.scene.frame_count));
    if(frame_count==0) throw std::runtime_error("no trajectory frames selected");
    float end_time = dvs::frame_time(frames[frame_count - 1], config.scene.fps);
    float start_time =
        dvs::frame_time(frames.front(), config.scene.fps);
    std::size_t total_steps = static_cast<std::size_t>(
        std::ceil((double(end_time) - start_time) * config.scene.cam_fps));

    dvs::ProgressBar bar(total_steps, "Rendering frames");
    const float write_interval = 1.0f / config.scene.cam_fps;
    std::vector<float> light_frame_float(config.camera.width *
                                         config.camera.height);
    std::vector<float> depth_frame(config.camera.width * config.camera.height);

    std::vector<u_int8_t> light_frame_uint8(config.camera.width *
                                            config.camera.height);

    for (size_t step = 0; step < total_steps; ++step) {
      float current_time = start_time + step * write_interval;
      const dvs::CameraTrajectoryState &state = traj.evaluate(current_time);
      float3 camera_origin = dvs::vec3_to_float3(state.position);
      const float *camera_world = state.orientation.data();
      float3 light_dir = dvs::vec3_to_float3(state.light_direction);
      float light_intensity = state.light_intensity;
      raycaster.render_main(camera_origin, camera_world, light_dir,
                            light_intensity, light_frame_float, depth_frame);

      // 输出当前渲染所得相机图像，精度为9位小数，以cam_fps频率输出
      // 使用OpenCV将light_frame保存为图像
      std::string image_name =
          image_output_dir / (std::to_string(current_time) + ".png");
      std::string depth_name =
          depth_output_dir / (std::to_string(current_time) + ".tif");
      dvs::render_float_to_png_bytes(light_frame_float.data(),
                                     config.camera.width, config.camera.height,
                                     0.0, light_frame_uint8);
      cv::Mat img_uint8(config.camera.height, config.camera.width, CV_8U,
                        light_frame_uint8.data());
      cv::Mat depth_img(config.camera.height, config.camera.width, CV_32F,
                        depth_frame.data());
      // 归一化图像到0-1范围
      cv::imwrite(image_name, img_uint8);
      cv::imwrite(depth_name, depth_img);

      total_images++;
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
