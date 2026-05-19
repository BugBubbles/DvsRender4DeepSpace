#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace dvs {

#pragma pack(push, 1)
struct Event {
  // 事件流数据结构，总结占4 + 4 + 2 + 2 + 1 = 13字节
  float timestamp = 0.0f; // 事件发生的时间戳，单位为秒，占4字节
  int16_t x = 0;          // 事件发生的像素位置的x坐标，占2字节
  int16_t y = 0;          // 事件发生的像素位置的y坐标，占2字节
  int8_t polarity = 0;    // 事件的极性，1表示正事件，-1表示负事件，占1字节
  float depth = 0.0f;     // 事件发生的深度信息，单位为米，占4字节
};
#pragma pack(pop)

/**
 * @brief 理想事件相机的CUDA核函数，用于批量生成事件。
 */
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float threshold_pos, const float threshold_neg,
    const float *depth_map, float *last_log_intensity, Event *events,
    int *event_count);

/**
 * @brief 考虑暗电流和不应期的事件相机的CUDA核函数。
 */
#ifdef ENABLE_PHYSIC_EFFECTS
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float *threshold_pos_map,
    const float *threshold_neg_map, const float dark_current,
    const float refractory_time, const float shot_noise_prob,
    const float shot_intensity, const float *depth_map, float *refractory_map,
    float *last_log_intensity, Event *events, int *event_count,
    curandState *rand_states);
/**
 * @brief 考虑暗电流和不应期以及高通滤波器的事件相机的CUDA核函数。
 */
#ifdef ENABLE_HIGH_PASS_FILTER
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float *threshold_pos_map,
    const float *threshold_neg_map, const float dark_current,
    const float refractory_time, const float alpha, const float shot_noise_prob,
    const float shot_intensity, const float *depth_map, float *refractory_map,
    float *last_log_intensity, float *last_voltage, Event *events,
    int *event_count, curandState *rand_states);
#ifdef ENABLE_AER_DELAY
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float *threshold_pos_map,
    const float *threshold_neg_map, const float dark_current,
    const float refractory_time, const float alpha, const float shot_noise_freq,
    const float shot_intensity, const float *depth_map, float *refractory_map,
    float *last_log_intensity, float *last_voltage, Event *events,
    int *event_count_row, int *event_count, curandState *rand_states);

void row_delay_buffer(const int height, const int *event_count_row,
                      const float unit_row_delay, const float unit_col_delay,
                      float *delay_row, int *last_row_idx);

__global__ void batch_aer_kernel(const int height, Event *events,
                                 float *row_delay);
#endif
#endif
#endif
} // namespace dvs
