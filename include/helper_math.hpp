#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <device_types.h>

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
constexpr float kRayEps = 1e-6f;
static constexpr int kMaxStackSize = 128; // 足够支持 6900 万三角形的深度

inline __host__ __device__ float clampf(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}
inline __host__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator-(float3 a) {
  return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float length(float3 v) {
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
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
inline __host__ __device__ float3 operator/(float a, float3 b) {
  return make_float3(a / b.x, a / b.y, a / b.z);
}
inline __host__ __device__ float3 operator/=(float3 &a, float b) {
  a.x /= b, a.y /= b, a.z /= b;
  return a;
}
inline __host__ __device__ float3 operator+=(float3 &a, float3 b) {
  a.x += b.x, a.y += b.y, a.z += b.z;
  return a;
}
inline __host__ __device__ float3 normalize(float3 v) {
  float len = sqrtf(dot(v, v));
  if (len > 0.0f) {
    return v / len;
  }
  return make_float3(0.0f, 0.0f, 0.0f);
}
inline __host__ __device__ float radiansf(float degrees) {
  return degrees * (M_PI / 180.0f);
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b) {
  return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
inline __host__ __device__ float3 fmaxf(float3 a, float b) {
  return make_float3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b) {
  return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
inline __host__ __device__ float3 fminf(float3 a, float b) {
  return make_float3(fminf(a.x, b), fminf(a.y, b), fminf(a.z, b));
}