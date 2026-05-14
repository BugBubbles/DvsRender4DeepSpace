#include "dvs/dvs_base.hpp"
#include <cuda_runtime.h>
#include <vector_functions.h>

namespace dvs {
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float threshold_pos, const float threshold_neg,
    float *last_log_intensity, Event *events, int *event_count) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  float current = intensity_frame[idx];
  if (current <= 0.0f) {
    if (last_log_intensity[idx] > -INFINITY) {
      last_log_intensity[idx] = -INFINITY;
      int pos = atomicAdd(event_count, 1);
      events[pos] =
          Event{timestamp, static_cast<int16_t>(x), static_cast<int16_t>(y), 0};
    }
    return;
  }
  // 光强转对数值
  float log_current = logf(current);
  if (last_log_intensity[idx] <= log_current - threshold_pos) {
    last_log_intensity[idx] = log_current;
    int pos = atomicAdd(event_count, 1);
    events[pos] =
        Event{timestamp, static_cast<int16_t>(x), static_cast<int16_t>(y), 1};
  } else if (last_log_intensity[idx] >= log_current + threshold_neg) {
    last_log_intensity[idx] = log_current;
    int pos = atomicAdd(event_count, 1);
    events[pos] =
        Event{timestamp, static_cast<int16_t>(x), static_cast<int16_t>(y), 0};
  }
}

__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float threshold_pos, const float threshold_neg,
    const float dark_current, const float refractory_time,
    float *refractory_map, float *last_log_intensity, Event *events,
    int *event_count) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  float current = intensity_frame[idx] + dark_current;
  float refractory_end_time = refractory_map[idx];
  if (timestamp < refractory_end_time)
    return;
  if (current <= 0.0f) {
    if (last_log_intensity[idx] > -INFINITY) {
      last_log_intensity[idx] = -INFINITY;
      int pos = atomicAdd(event_count, 1);
      events[pos] =
          Event{timestamp, static_cast<int16_t>(x), static_cast<int16_t>(y), 0};
      refractory_map[idx] = timestamp + refractory_time;
    }
    return;
  }
  // 光强转对数值
  float log_current = logf(current);
  if (last_log_intensity[idx] <= log_current - threshold_pos) {
    last_log_intensity[idx] = log_current;
    int pos = atomicAdd(event_count, 1);
    events[pos] =
        Event{timestamp, static_cast<int16_t>(x), static_cast<int16_t>(y), 1};
    refractory_map[idx] = timestamp + refractory_time;
  } else if (last_log_intensity[idx] >= log_current + threshold_neg) {
    last_log_intensity[idx] = log_current;
    int pos = atomicAdd(event_count, 1);
    events[pos] =
        Event{timestamp, static_cast<int16_t>(x), static_cast<int16_t>(y), 0};
    refractory_map[idx] = timestamp + refractory_time;
  }
}
} // namespace dvs