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

namespace tools {
std::string read_command_line(const char *prompt) {
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

bool parse_three_floats(std::istringstream &iss, float &a, float &b, float &c) {
  return static_cast<bool>(iss >> a >> b >> c);
}

void render_frame(const dvs::Vec3 &position, const dvs::Vec3 &rotation,
                  const dvs::Vec3 &light_dir, const float light_intensity,
                  dvs::CamRayCaster &raycaster, const dvs::Config &config,
                  const std::filesystem::path &image_output_dir,
                  std::vector<float> &light_frame_float,
                  std::vector<u_int8_t> &light_frame_uint8) {
  float3 camera_origin = dvs::vec3_to_float3(position);
  float3 light_direction = dvs::vec3_to_float3(light_dir);

  dvs::Mat3 rotation_matrix;
  if (rotation.norm() < 1e-6f) {
    rotation_matrix = Eigen::Matrix3f::Identity();
  } else {
    rotation_matrix =
        Eigen::AngleAxisf(rotation.norm(), rotation / rotation.norm())
            .toRotationMatrix();
  }

  raycaster.render_main(camera_origin, rotation_matrix.data(), light_direction,
                        light_intensity, light_frame_float);

  std::string image_name = image_output_dir.string() + "/realtime_rendered.png";
  dvs::render_float_to_png_bytes(light_frame_float.data(), config.camera.width,
                                 config.camera.height, 0.0f, light_frame_uint8);

  cv::Mat img_uint8(config.camera.height, config.camera.width, CV_8U,
                    light_frame_uint8.data());
  cv::imwrite(image_name, img_uint8);
}
} // namespace tools
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
    time_t now = std::time(nullptr);
    // 使用当前时间戳创建一个唯一的输出目录
    std::string timestamp = std::to_string(now);
    std::filesystem::path traj_file_name =
        image_output_dir / ("trajectory_" + timestamp + ".txt");
    std::ofstream traj_out(traj_file_name);

    std::vector<float> light_frame_float(config.camera.width *
                                         config.camera.height);
    std::vector<u_int8_t> light_frame_uint8(config.camera.width *
                                            config.camera.height);
    // 保持循环，直到用户按下 Ctrl+C
    dvs::Vec3 position(0, 0, 0), light_dir(0, 0, 1), rotation(0, 0, 1);
    int frame_idx = 0;
    float light_intensity = 1.0f;
    std::string line;
    float velocity = 0.1f; // 默认移动速度
    // 命令行格式：输入一个指令 pos/rot/light/intensity，后面接对应的参数
    // 注意，平移是沿着当前视角的前/后/左/右方向，旋转是绕着当前视角的上下左右轴旋转
    std::cout << "Realtime Trajectory Control CLI" << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  idx <i>             - Set current frame index" << std::endl;
    std::cout << "  pos x y z           - Set camera position" << std::endl;
    std::cout << "  rot x y z           - Set camera rotation (rotation vector)"
              << std::endl;
    std::cout << "  light x y z         - Set light direction" << std::endl;
    std::cout << "  intensity value     - Set light intensity" << std::endl;
    std::cout << "  list                - List current parameters" << std::endl;
    std::cout << "  write               - Write down current parameters"
              << std::endl;
    std::cout << "  render              - Render current frame " << std::endl;
    std::cout
        << "  left/right/up/down  - Move camera in the corresponding direction"
        << std::endl;
    std::cout << "  front/back          - Move camera forward/backward"
              << std::endl;
    std::cout << "  move v              - Move with velocity v" << std::endl;
    std::cout << "  help                - Show this help message" << std::endl;
    std::cout << "  quit / exit         - Exit the program" << std::endl;

    while (true) {
      std::string line = tools::read_command_line("traj> ");
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
        std::cout << "  l(eft)/r(ight)/u(p)/d(own)/f(ront)/b(ack)\n";
        std::cout << "  move <v>\n";
        std::cout << "  quit / exit\n";
        continue;
      }

      if (command == "quit" || command == "exit") {
        break;
      } else if (command == "move") {
        if (!(iss >> velocity)) {
          std::cout << "Usage: move <velocity>\n";
          continue;
        }
        std::cout << "Movement velocity set to: " << velocity << "\n";
      } else if (command == "left" || command == "right" || command == "up" ||
                 command == "down" || command == "front" || command == "back" ||
                 command == "l" || command == "r" || command == "u" ||
                 command == "d" || command == "f" || command == "b") {
        dvs::Mat3 rotation_matrix;
        if (rotation.norm() < 1e-6f) {
          rotation_matrix = Eigen::Matrix3f::Identity();
        } else {
          rotation_matrix =
              Eigen::AngleAxisf(rotation.norm(), rotation / rotation.norm())
                  .toRotationMatrix();
        }
        // 计算在当前相机坐标系下的前/右方向
        dvs::Vec3 forward = rotation_matrix.col(2); // 前方向
        dvs::Vec3 up = rotation_matrix.col(1);      // 上方向
        dvs::Vec3 right = rotation_matrix.col(0);   // 右方向

        if (command == "left" || command == "l") {
          position -= right * velocity;
        } else if (command == "right" || command == "r") {
          position += right * velocity;
        } else if (command == "up" || command == "u") {
          position -= up * velocity;
        } else if (command == "down" || command == "d") {
          position += up * velocity;
        } else if (command == "front" || command == "f") {
          position -= forward * velocity;
        } else if (command == "back" || command == "b") {
          position += forward * velocity;
        }

        std::cout << "Position updated to: " << position.transpose() << "\n";
        tools::render_frame(position, rotation, light_dir, light_intensity,
                            raycaster, config, image_output_dir,
                            light_frame_float, light_frame_uint8);

      } else if (command == "idx") {
        if (!(iss >> frame_idx)) {
          std::cout << "Usage: idx <frame_index>\n";
          continue;
        }
        std::cout << "Current frame index set to: " << frame_idx << "\n";
      } else if (command == "pos") {
        if (!tools::parse_three_floats(iss, position.x(), position.y(),
                                       position.z())) {
          std::cout << "Usage: pos <x y z>\n";
          continue;
        }
        std::cout << "Camera position set to: " << position.transpose() << "\n";
      } else if (command == "rot") {
        if (!tools::parse_three_floats(iss, rotation.x(), rotation.y(),
                                       rotation.z())) {
          std::cout << "Usage: rot <x y z>\n";
          continue;
        }
        std::cout << "Camera rotation set to: " << rotation.transpose() << "\n";
      } else if (command == "light") {
        if (!tools::parse_three_floats(iss, light_dir.x(), light_dir.y(),
                                       light_dir.z())) {
          std::cout << "Usage: light <x y z>\n";
          continue;
        }
        std::cout << "Light direction set to: " << light_dir.transpose()
                  << "\n";
        light_dir.normalize();
      } else if (command == "intensity") {
        if (!(iss >> light_intensity)) {
          std::cout << "Usage: intensity <value>\n";
          continue;
        }
        if (light_intensity < 0.0f) {
          std::cout << "Light intensity must be non-negative.\n";
          continue;
        }
        std::cout << "Light intensity set to: " << light_intensity << "\n";
      } else if (command == "list") {
        std::cout << "Current parameters:\n";
        std::cout << "Frame Index: " << frame_idx << "\n";
        std::cout << "Position: " << position.transpose() << "\n";
        std::cout << "Rotation: " << rotation.transpose() << "\n";
        std::cout << "Light Direction: " << light_dir.transpose() << "\n";
        std::cout << "Light Intensity: " << light_intensity << "\n";
      } else if (command == "write") {

        Eigen::Quaternionf quat;
        if (rotation.norm() < 1e-6f) {
          quat = Eigen::Quaternionf::Identity();
        } else {
          Eigen::AngleAxisf angle_axis(rotation.norm(),
                                       rotation / rotation.norm());
          quat = Eigen::Quaternionf(angle_axis);
        }

        traj_out << frame_idx << " " << position.transpose() << " " << quat.x()
                 << " " << quat.y() << " " << quat.z() << " " << quat.w() << " "
                 << light_dir.transpose() << " " << light_intensity << "\n";

        traj_out.flush();
        std::cout << "Frame [" << frame_idx
                  << "] written to trajectory file.\n";
      } else if (command == "render") {

        tools::render_frame(position, rotation, light_dir, light_intensity,
                            raycaster, config, image_output_dir,
                            light_frame_float, light_frame_uint8);
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