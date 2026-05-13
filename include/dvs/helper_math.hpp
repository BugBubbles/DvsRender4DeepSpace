#pragma once

#include "dvs/math_types.hpp"
#include <cuda_runtime.h>

namespace dvs {

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
constexpr float kRayEps = 1e-6f;
static constexpr int kMaxStackSize = 128; // 足够支持 6900 万三角形的深度

inline __host__ __device__ float clampf(float x, float lo, float hi) {
  return clampf(x, lo, hi);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}
inline __host__ __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float3 operator*(float b, float3 a) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator/(float3 a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ float3 normalize(float3 v) {
  float len = sqrtf(dot(v, v));
  if (len > 0.0f) {
    return v / len;
  }
  return make_float3(0.0f, 0.0f, 0.0f);
}
} // namespace dvs