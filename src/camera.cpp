#include "dvs/camera.hpp"

#include <cstdlib>
#include <stdexcept>

#include "dvs/math_types.hpp"
#include "dvs/scene_types.hpp"
#include <Eigen/Core>
#include <Eigen/LU>

namespace dvs {

std::vector<Ray> generate_camera_rays(const FrameData &frame,
                                      const SceneBundle &scene) {
  int width = scene.width;
  int height = scene.height;
  float fx = scene.fx;
  float fy = scene.fy;
  float cx = scene.cx;
  float cy = scene.cy;

  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("width and height must be positive");
  }
  Eigen::Matrix4f proj_eigen, inv_proj_eigen;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      proj_eigen(i, j) = frame.camera_world.m[i * 4 + j];
    }
  }
  inv_proj_eigen = proj_eigen.inverse();
  std::cout << "inv_proj_eigen Matrix: " << std::endl
            << inv_proj_eigen << std::endl;
  std::cout << "Camera Origin: " << frame.camera_origin << std::endl;
  std::vector<Ray> rays(static_cast<std::size_t>(width * height));
  std::cout << (- cx + 0.5f) / fx <<'______' << (- cy + 0.5f) / fy << std::endl;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const float x_ndc = (static_cast<float>(u) - cx + 0.5f) / fx;
      const float y_ndc = (static_cast<float>(v) - cy + 0.5f) / fy;

      // 相机坐标系转向世界坐标系
      Eigen::Vector4f p_world =
          inv_proj_eigen * Eigen::Vector4f(-x_ndc, -y_ndc, -1.0f, 1.0f);

      

      if (fabsf(p_world[3]) > 1e-8f) {
        p_world[0] /= p_world[3];
        p_world[1] /= p_world[3];
        p_world[2] /= p_world[3];
        p_world[3] = 1.0f;
      }
      const Vec3 point_ray{p_world[0], p_world[1], p_world[2]};

      // std::cout << "point_ray: " << point_ray << std::endl;

      const Vec3 ray_dir = normalize(point_ray - frame.camera_origin);
      if ((u == 0 && v == 0) || (u == width - 1 && v == height - 1) ||
          (u == width / 2 && v == height / 2)) {
        std::cout << "Ray :" << ray_dir << std::endl;
        std::exit(0);
      }
      const std::size_t idx = static_cast<std::size_t>(v * width + u);
      rays[idx] = Ray{frame.camera_origin, ray_dir};
    }
  }

  return rays;
}

} // namespace dvs
