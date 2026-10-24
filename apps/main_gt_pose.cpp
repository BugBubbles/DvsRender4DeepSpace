#include "app_inputs.hpp"
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

    dvs::load_app_inputs(opts,config,scene,frames);

    // 使用样条曲线获取连续位姿轨迹
    dvs::CameraTrajectory traj(frames, config.scene);
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
    if(frame_count==0) throw std::runtime_error("no trajectory frames selected");
    float end_time = dvs::frame_time(frames[frame_count - 1], config.scene.fps);
    float start_time =
        dvs::frame_time(frames.front(), config.scene.fps);
    std::size_t total_steps=static_cast<std::size_t>(std::ceil((double(end_time)-start_time)*config.scene.imu_fps));
    dvs::ProgressBar bar(total_steps,"Generating GT Pose");
    int total_trajs=0;
    for(size_t step=0;step<total_steps;++step) {
      double current_time=start_time+double(step)/config.scene.imu_fps;
      auto state=traj.evaluate(current_time);
      motion_out<<std::fixed<<std::setprecision(9)<<current_time<<" "
                <<state.linear_velocity.transpose()<<" "<<state.angular_velocity.transpose()<<"\n";
      pose_out<<std::fixed<<std::setprecision(9)<<current_time<<" "<<state.position.transpose()<<" "
              <<state.quaternion.x()<<" "<<state.quaternion.y()<<" "<<state.quaternion.z()<<" "<<state.quaternion.w()<<"\n";
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
