#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace dvs {

#pragma pack(push, 1)
struct Event {
  // 事件流数据结构，总结占4 + 2 + 2 + 1 = 9字节
  float timestamp = 0.0f; // 事件发生的时间戳，单位为秒，占4字节
  int16_t x = 0;          // 事件发生的像素位置的x坐标，占2字节
  int16_t y = 0;          // 事件发生的像素位置的y坐标，占2字节
  int8_t polarity = 0;    // 事件的极性，1表示正事件，-1表示负事件，占1字节
};
#pragma pack(pop)

/**
 * @brief 理想事件相机的CUDA核函数，用于批量生成事件。
 */
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float threshold_pos, const float threshold_neg,
    float *last_log_intensity, Event *events, int *event_count);

/**
 * @brief 考虑暗电流和不应期的事件相机的CUDA核函数。
 */
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float *threshold_pos_map, const float *threshold_neg_map,
    const float dark_current, const float refractory_time,
    float *refractory_map, float *last_log_intensity, Event *events,
    int *event_count);
} // namespace dvs
