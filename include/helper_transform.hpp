#pragma once

#include "helper_math.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <device_types.h>

inline __host__ __device__ float3 rotation_vector(const float4 quaternion,
                                                  const float3 vector) {
  // 使用四元数旋转向量
  float3 rot_mat_row0, rot_mat_row1, rot_mat_row2;
  rot_mat_row0 = make_float3(
      1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
      2 * (quaternion.x * quaternion.y - quaternion.z * quaternion.w),
      2 * (quaternion.x * quaternion.z + quaternion.y * quaternion.w));
  rot_mat_row1 = make_float3(
      2 * (quaternion.x * quaternion.y + quaternion.z * quaternion.w),
      1 - 2 * (quaternion.x * quaternion.x + quaternion.z * quaternion.z),
      2 * (quaternion.y * quaternion.z - quaternion.x * quaternion.w));
  rot_mat_row2 = make_float3(
      2 * (quaternion.x * quaternion.z - quaternion.y * quaternion.w),
      2 * (quaternion.y * quaternion.z + quaternion.x * quaternion.w),
      1 - 2 * (quaternion.x * quaternion.x + quaternion.y * quaternion.y));

  return make_float3(dot(rot_mat_row0, vector), dot(rot_mat_row1, vector),
                     dot(rot_mat_row2, vector));
}

/**
 * @brief 使用欧拉角旋转向量
 * @param euler_angles 欧拉角，单位为弧度，顺序为(x, y, z)
 * @param vector 待旋转的向量
 * @return 旋转后的向量
 */
inline __host__ __device__ float3 rotation_vector(const float3 euler_angles,
                                                  const float3 vector) {
  // 使用欧拉角旋转向量
  float3 rot_mat_row0, rot_mat_row1, rot_mat_row2;
  float cx = cosf(euler_angles.x);
  float sx = sinf(euler_angles.x);
  float cy = cosf(euler_angles.y);
  float sy = sinf(euler_angles.y);
  float cz = cosf(euler_angles.z);
  float sz = sinf(euler_angles.z);

  rot_mat_row0 = make_float3(cy * cz, -cy * sz, sy);
  rot_mat_row1 =
      make_float3(sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy);
  rot_mat_row2 =
      make_float3(-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy);

  return make_float3(dot(rot_mat_row0, vector), dot(rot_mat_row1, vector),
                     dot(rot_mat_row2, vector));
}