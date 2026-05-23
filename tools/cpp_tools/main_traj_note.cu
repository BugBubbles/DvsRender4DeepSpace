#include "bvh/build.h"
#include "dvs/blenderize.hpp"
#include "dvs/raycast.hpp"
#include "dvs/scene_data.hpp"
#include "dvs/scene_types.hpp"
#include "dvs/trajectory.hpp"
#include "tools/cli.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <readline/history.h>
#include <readline/readline.h>
#include <sstream>
#include <string>

static std::string read_command_line(const char *prompt) {
  char *raw = readline(prompt);
  if (raw == nullptr) {
    return {};
  }

  std::string line(raw);
  if (!line.empty()) {
    add_history(raw);
  }
  free(raw);
  return line;
}

static bool parse_three_floats(std::istringstream &iss, float &a, float &b,
                               float &c) {
  return static_cast<bool>(iss >> a >> b >> c);
}

int main(int argc, char **argv) {
  try {
    if (argc <= 1) {
      dvs::Options::print_usage(argv[0]);
      return 1;
    }
    dvs::Options opts(argc, argv);
    dvs::SceneBundle<float3> scene;
    std::string error_message;
    dvs::Config config(opts.config_path, error_message);
    if (!error_message.empty()) {
      throw std::runtime_error("failed to load config: " + error_message);
      return 1;
    }
    if (!dvs::load_scene_bundle(opts.dem_path, opts.albedo_path, scene,
                                &error_message)) {
      throw std::runtime_error("failed to load scene: " + error_message);
      return 1;
    }
    dvs::BVHBuildParams params; // 可使用默认参数
    int root_index =
        dvs::BVHBuilder::build(params, scene.bvh_nodes, scene.bvh_triangles);

    dvs::CamRayCaster raycaster(scene, config, root_index);
    std::filesystem::path image_output_dir(opts.output_dir);
    if (!std::filesystem::exists(image_output_dir)) {
      std::filesystem::create_directories(image_output_dir);
    }
    // 用于保存当前写入的帧参数的文件流，输出文件的格式为 idx x y z qx qy qz qw
    // lx ly lz intensity
    std::ofstream traj_out(opts.traj_path);

    std::size_t record_frame_count = 0;

    std::vector<float> light_frame_float(config.camera.width *
                                         config.camera.height);
    std::vector<u_int8_t> light_frame_uint8(config.camera.width *
                                            config.camera.height);
    // 保持循环，直到用户按下 Ctrl+C
    dvs::Vec3 position, light_dir, rotation;
    int traj_set_flag = 0x0, frame_idx;
    float light_intensity;
    std::string line;
    // 命令行格式：输入一个指令 pos/rot/light/intensity，后面接对应的参数
    std::cout << "Realtime Trajectory Control CLI" << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  pos x y z           - Set camera position" << std::endl;
    std::cout << "  rot x y z           - Set camera rotation (rotation vector)"
              << std::endl;
    std::cout << "  light x y z         - Set light direction" << std::endl;
    std::cout << "  intensity value     - Set light intensity" << std::endl;
    std::cout << "  list                - List current parameters" << std::endl;
    std::cout
        << "  render              - Render current frame with set parameters"
        << std::endl;
    while (true) {
      std::string line = read_command_line("traj> ");
      if (line.empty()) {
        std::cout << "\n";
        break;
      }

      std::istringstream iss(line);
      std::string command;
      iss >> command;
      if (command.empty()) {
        continue;
      }

      if (command == "help") {
        std::cout << "Commands:\n";
        std::cout << "  idx <i>\n";
        std::cout << "  pos <x y z>\n";
        std::cout << "  rot <x y z>\n";
        std::cout << "  light <x y z>\n";
        std::cout << "  intensity <value>\n";
        std::cout << "  list\n";
        std::cout << "  write\n";
        std::cout << "  render\n";
        std::cout << "  quit / exit\n";
        continue;
      }

      if (command == "quit" || command == "exit") {
        break;
      } else if (command == "idx") {
        if (!(iss >> frame_idx)) {
          std::cout << "Usage: idx <frame_index>\n";
          continue;
        }
        std::cout << "Current frame index set to: " << frame_idx << "\n";
        traj_set_flag |= 0x10;
      } else if (command == "pos") {
        if (!parse_three_floats(iss, position.x(), position.y(),
                                position.z())) {
          std::cout << "Usage: pos <x y z>\n";
          continue;
        }
        traj_set_flag |= 0x1;
        std::cout << "Camera position set to: " << position.transpose() << "\n";
      } else if (command == "rot") {
        if (!parse_three_floats(iss, rotation.x(), rotation.y(),
                                rotation.z())) {
          std::cout << "Usage: rot <x y z>\n";
          continue;
        }
        traj_set_flag |= 0x2;
        std::cout << "Camera rotation set to: " << rotation.transpose() << "\n";
      } else if (command == "light") {
        if (!parse_three_floats(iss, light_dir.x(), light_dir.y(),
                                light_dir.z())) {
          std::cout << "Usage: light <x y z>\n";
          continue;
        }
        traj_set_flag |= 0x4;
        std::cout << "Light direction set to: " << light_dir.transpose()
                  << "\n";
      } else if (command == "intensity") {
        if (!(iss >> light_intensity)) {
          std::cout << "Usage: intensity <value>\n";
          continue;
        }
        if (light_intensity < 0.0f) {
          std::cout << "Light intensity must be non-negative.\n";
          continue;
        }
        traj_set_flag |= 0x8;
        std::cout << "Light intensity set to: " << light_intensity << "\n";
      } else if (command == "list") {
        std::cout << "Current parameters:\n";
        std::cout << "Frame Index: " << frame_idx << "\n";
        std::cout << "Position: " << position.transpose() << "\n";
        std::cout << "Rotation: " << rotation.transpose() << "\n";
        std::cout << "Light Direction: " << light_dir.transpose() << "\n";
        std::cout << "Light Intensity: " << light_intensity << "\n";
      } else if (command == "write") {
        if (traj_set_flag != 0x1F) {
          std::cout << "Please set idx/pos/rot/light/intensity first.\n";
          continue;
        }

        float rotation_norm;
        if (rotation.norm() < 1e-6f) {
          rotation_norm = 0.0f;
          rotation = dvs::Vec3(0.0f, 0.0f, 1.0f);
        } else {
          rotation_norm = rotation.norm();
          rotation /= rotation_norm;
        }

        Eigen::AngleAxisf angle_axis(rotation_norm, rotation);
        Eigen::Quaternionf quat(angle_axis);

        traj_out << frame_idx << " " << position.transpose() << " " << quat.x()
                 << " " << quat.y() << " " << quat.z() << " " << quat.w() << " "
                 << light_dir.transpose() << " " << light_intensity << "\n";

        traj_out.flush();
        std::cout << "Frame [" << record_frame_count
                  << "] written to trajectory file.\n";
        ++record_frame_count;
      } else if (command == "render") {
        if (traj_set_flag != 0x1F) {
          std::cout << "Please set idx/pos/rot/light/intensity first.\n";
          continue;
        }

        float3 camera_origin = dvs::vec3_to_float3(position);
        float3 light_direction = dvs::vec3_to_float3(light_dir);

        float rotation_norm;
        if (rotation.norm() < 1e-6f) {
          rotation_norm = 0.0f;
          rotation = dvs::Vec3(0.0f, 0.0f, 1.0f);
        } else {
          rotation_norm = rotation.norm();
          rotation /= rotation_norm;
        }

        dvs::Mat3 rotation_matrix =
            Eigen::AngleAxisf(rotation_norm, rotation).toRotationMatrix();

        raycaster.render_main(camera_origin, rotation_matrix.data(),
                              light_direction, light_intensity,
                              light_frame_float);

        std::string image_name =
            image_output_dir.string() + "/realtime_rendered.png";
        dvs::render_float_to_png_bytes(
            light_frame_float.data(), config.camera.width, config.camera.height,
            0.0f, light_frame_uint8);

        cv::Mat img_uint8(config.camera.height, config.camera.width, CV_8U,
                          light_frame_uint8.data());
        cv::imwrite(image_name, img_uint8);

        std::cout << "Rendered image [" << record_frame_count
                  << "] saved to: " << image_name << "\n";
      } else {
        std::cout << "Unknown command. Type 'help' for help.\n";
      }
    }
    traj_out.close();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}