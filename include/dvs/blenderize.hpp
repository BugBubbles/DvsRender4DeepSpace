#pragma once
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <vector>

using uint32_t = unsigned int;

namespace dvs {
// --- Blender: unit_float_to_uchar_clamp ---
inline u_int8_t to_byte(float v) {
  if (!std::isfinite(v) || v <= 0.0f) {
    return 0;
  }
  if (v >= 1.0f) {
    return 255;
  }
  return static_cast<u_int8_t>(v * 255.0f + 0.5f);
}

// --- Blender: linearrgb_to_srgb (scalar form) ---
// Used by linearrgb_to_srgb_v3_v3 in math_color.cc
static inline float linear_to_srgb(float c) {
  if (c < 0.0031308f) {
    return 12.92f * c;
  }
  return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

inline float display_exposure(float x) { return std::clamp(x / 100, 0.0f, 1.0f); }

inline float hash_iqint3_f(const uint32_t x, const uint32_t y) {
  const uint32_t qx = 1103515245u * ((x >> 1u) ^ (y));
  const uint32_t qy = 1103515245u * ((y >> 1u) ^ (x));
  const uint32_t n = 1103515245u * ((qx) ^ (qy >> 3u));
  return float(n) * (1.0f / float(0xffffffffu));
}

inline float random_dither(int x, int y) {
  float v = hash_iqint3_f(x, y);
  /* Convert uniform distribution into triangle-shaped distribution. Based on
   * "remap_pdf_tri_unity" from https://www.shadertoy.com/view/WldSRf */
  v = v * 2.0f - 1.0f;
  v = std::copysignf(1.0f, v) * (1.0f - sqrtf(1.0f - fabsf(v)));
  return v;
}

// Scalar grayscale preview: fixed exposure, sRGB encoding and optional dither.
inline void render_float_to_png_bytes(const float *linear, int width,
                                      int height, float dither,
                                      std::vector<u_int8_t> &out_bytes) {
  out_bytes.resize(static_cast<size_t>(width)*height);

  // linear -> sRGB + 量化
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // 线性 -> sRGB (与 Blender 的 linearrgb_to_srgb_v3_v3 等价)
      float r = linear[y * width + x];
      r = display_exposure(r);
      r = linear_to_srgb(r);

      // 可选：抖动（Blender 在 IMB_buffer_byte_from_float 中做）
      r += dither * random_dither(x, y);

      out_bytes[y * width + x] = to_byte(r);
    }
  }
}

} // namespace dvs
