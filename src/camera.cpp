#include "dvs/camera.hpp"

#include <stdexcept>

#include "dvs/math_types.hpp"
#include "dvs/scene_types.hpp"
#include <Eigen/Core>

namespace dvs {

std::vector<Ray<Vec3>> generate_camera_rays(const FrameData<Vec3, Mat3>& frame, const SceneBundle<Vec3, Uint3, Mat3>& scene) {
  int width = scene.width;
  int height = scene.height;
  float fx = scene.fx;
  float fy = scene.fy;
  float cx = scene.cx;
  float cy = scene.cy;

  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("width and height must be positive");
  }
  Eigen::Matrix3f camera_world;
  camera_world = frame.camera_world.block<3, 3>(0, 0);

  std::cout << "camera_world: " << std::endl << camera_world << std::endl;
  std::vector<Ray<Vec3>> rays(static_cast<std::size_t>(width * height));
  for (int v = height/2; v <= height/2; ++v) {
    for (int u = width/2; u <=width/2; ++u) {
      const float x_ndc = (static_cast<float>(u) - cx + 0.5f) / fx;
      const float y_ndc = (static_cast<float>(v) - cy + 0.5f) / fy;

      // Blender camera looks along local -Z, so rays must be emitted toward z = -1 in camera space.
      Eigen::Vector3f p_world =
          camera_world * Eigen::Vector3f(x_ndc, y_ndc, -1.0f);

      if (std::fabs(p_world[2]) > 1e-8f) {
        p_world[0] /= p_world[2];
        p_world[1] /= p_world[2];
        p_world[2] = 1.0f;
      }
      const Vec3 point_ray(p_world[0], p_world[1], p_world[2]);
      std::cout << "point_ray: " << point_ray << std::endl;

      const Vec3 ray_dir = point_ray.normalized();
      const std::size_t idx = static_cast<std::size_t>(v * width + u);
      rays[idx] = Ray<Vec3>{frame.camera_origin, ray_dir};
    }
  }

  return rays;
}

} // namespace dvs
