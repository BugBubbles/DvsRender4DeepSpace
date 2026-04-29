#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <Eigen/Dense>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "dvs/helper_math.h"
#define DVS_HD __host__ __device__
#else
#define DVS_HD
#endif

namespace dvs {

constexpr float inf = std::numeric_limits<float>::infinity();
constexpr float kPi = 3.14159265358979323846f;

using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;
using Mat3 = Eigen::Matrix3f;
using Mat4 = Eigen::Matrix4f;
using Uint3 = Eigen::Matrix<unsigned int, 3, 1>;
}  // namespace dvs