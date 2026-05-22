#pragma once

#include <Eigen/Dense>
#include <cmath>

#define CUDA_CHECK(expr, pos)                                                  \
  do {                                                                         \
    const cudaError_t err__ = (expr);                                          \
    if (err__ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("[CUDA Error] in ") + pos +         \
                               std::string(" : ") +                            \
                               cudaGetErrorString(err__));                     \
    }                                                                          \
  } while (0)

namespace dvs {

constexpr float inf = std::numeric_limits<float>::infinity();
constexpr float kPi = 3.14159265358979323846f;

using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;
using Mat3 = Eigen::Matrix3f;
using Mat4 = Eigen::Matrix4f;
using Uint3 = Eigen::Matrix<unsigned int, 3, 1>;
} // namespace dvs