#include "app_inputs.hpp"
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
    std::filesystem::path image_output_dir(opts.output_dir + "/blur_images");
    if (!std::filesystem::exists(image_output_dir)) {
      std::filesystem::create_directories(image_output_dir);
    }

    std::size_t total_images = 0;

    size_t frame_count =
        std::min(frames.size(), static_cast<size_t>(config.scene.frame_count));
    if(frame_count==0) throw std::runtime_error("no trajectory frames selected");
    float end_time = dvs::frame_time(frames[frame_count - 1], config.scene.fps);
    float start_time =
        dvs::frame_time(frames.front(), config.scene.fps);
    const double interval=1.0/config.scene.cam_fps;
    const size_t exposures=static_cast<size_t>(std::ceil((double(end_time)-start_time)/interval));
    dvs::ProgressBar bar(exposures,"Rendering motion-blur frames");
    std::vector<float> image;
    std::vector<uint8_t> bytes(config.camera.width*config.camera.height);
    for(size_t exposure=0;exposure<exposures;++exposure) {
      double begin=start_time+exposure*interval;
      double end=std::min(double(end_time),start_time+(exposure+1)*interval);
      if(end<=begin) break;
      cv::Mat accumulated(config.camera.height,config.camera.width,CV_32F,cv::Scalar(0));
      size_t cells=static_cast<size_t>(std::ceil((end-begin)/config.scene.blur_resolution));
      for(size_t cell=0;cell<cells;++cell) {
        double left=begin+cell*double(config.scene.blur_resolution);
        double right=std::min(end,begin+(cell+1)*double(config.scene.blur_resolution));
        auto state=traj.evaluate((left+right)*.5);
        raycaster.render_main(dvs::vec3_to_float3(state.position),state.orientation.data(),
                              dvs::vec3_to_float3(state.light_direction),state.light_intensity,image);
        accumulated+=cv::Mat(config.camera.height,config.camera.width,CV_32F,image.data())*float((right-left)/(end-begin));
      }
      dvs::render_float_to_png_bytes(accumulated.ptr<float>(),config.camera.width,config.camera.height,0,bytes);
      if(!cv::imwrite((image_output_dir/(std::to_string(begin)+".png")).string(),
                     cv::Mat(config.camera.height,config.camera.width,CV_8U,bytes.data())))
        throw std::runtime_error("failed to write blur image");
      ++total_images;bar.update();
    }

    std::cout << "Finished. Wrote " << total_images << " images to "
              << opts.output_dir << "\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
