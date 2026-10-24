#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace dvs {

#pragma pack(push, 1)
struct Event {
  // Internal event uses double seconds; legacy binary writer explicitly emits 13 bytes.
  double timestamp = 0.0; // 事件发生的时间戳，单位为秒，占8字节
  int16_t x = 0;          // 事件发生的像素位置的x坐标，占2字节
  int16_t y = 0;          // 事件发生的像素位置的y坐标，占2字节
  int8_t polarity = 0;    // 事件的极性，1表示正事件，0表示负事件，占1字节
  float depth = 0.0f;     // 事件发生的深度信息，单位为米，占4字节
};
#pragma pack(pop)

} // namespace dvs
