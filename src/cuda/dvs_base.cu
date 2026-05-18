#include "dvs/dvs_base.hpp"
#include <cuda_runtime.h>
#include <vector_functions.h>

namespace dvs {
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float threshold_pos, const float threshold_neg,
    const float *depth_map, float *last_log_intensity, Event *events,
    int *event_count) {
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
      events[pos] = Event{timestamp, static_cast<int16_t>(x),
                          static_cast<int16_t>(y), 0, depth_map[idx]};
    }
    return;
  }
  // 光强转对数值
  float log_current = logf(current);
  if (last_log_intensity[idx] <= log_current - threshold_pos) {
    last_log_intensity[idx] = log_current;
    int pos = atomicAdd(event_count, 1);
    events[pos] = Event{timestamp, static_cast<int16_t>(x),
                        static_cast<int16_t>(y), 1, depth_map[idx]};
  } else if (last_log_intensity[idx] >= log_current + threshold_neg) {
    last_log_intensity[idx] = log_current;
    int pos = atomicAdd(event_count, 1);
    events[pos] = Event{timestamp, static_cast<int16_t>(x),
                        static_cast<int16_t>(y), 0, depth_map[idx]};
  }
}
#ifdef ENABLE_PHYSIC_EFFECTS
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float *threshold_pos_map,
    const float *threshold_neg_map, const float dark_current,
    const float refractory_time, const float shot_noise_freq,
    const float shot_intensity, const float *depth_map, float *refractory_map,
    float *last_log_intensity, Event *events, int *event_count,
    curandState *rand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  if (timestamp < refractory_map[idx])
    return;
  float current = intensity_frame[idx] + dark_current;

  // 模拟散粒噪声
  // curandState local_state = rand_states[idx];
  float shot_noise_prob =
      1 - expf(-shot_noise_freq *
               (timestamp - refractory_map[idx])); // 转换为每毫秒的概率
  if (curand_uniform(rand_states + idx) < shot_noise_prob) {
    current += shot_intensity;
  }
  // rand_states[idx] = local_state;
  // 光强转对数值
  float log_current = logf(current);
  if (last_log_intensity[idx] <= log_current - threshold_pos_map[idx]) {
    last_log_intensity[idx] = log_current;
    int pos = atomicAdd(event_count, 1);
    events[pos] = Event{timestamp, static_cast<int16_t>(x),
                        static_cast<int16_t>(y), 1, depth_map[idx]};
    refractory_map[idx] = timestamp + refractory_time;
  } else if (last_log_intensity[idx] >= log_current + threshold_neg_map[idx]) {
    last_log_intensity[idx] = log_current;
    int pos = atomicAdd(event_count, 1);
    events[pos] = Event{timestamp, static_cast<int16_t>(x),
                        static_cast<int16_t>(y), 0, depth_map[idx]};
    refractory_map[idx] = timestamp + refractory_time;
  }
}

#ifdef ENABLE_HIGH_PASS_FILTER
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float *threshold_pos_map,
    const float *threshold_neg_map, const float dark_current,
    const float refractory_time, const float alpha, const float shot_noise_freq,
    const float shot_intensity, const float *depth_map, float *refractory_map,
    float *last_log_intensity, float *last_voltage, Event *events,
    int *event_count, curandState *rand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  if (timestamp < refractory_map[idx])
    return;
  float current = intensity_frame[idx] + dark_current;

  // 模拟散粒噪声
  // curandState local_state = rand_states[idx];
  float shot_noise_prob =
      1 - expf(-shot_noise_freq *
               (timestamp - refractory_map[idx])); // 转换为每毫秒的概率
  if (curand_uniform(rand_states + idx) < shot_noise_prob) {
    current += shot_intensity;
  }
  // rand_states[idx] = local_state;
  // 光强转对数值
  float log_current = logf(current);
  // 高通滤波
  float voltage_current = (1 - alpha) * last_voltage[idx] + alpha * log_current;
  last_voltage[idx] = voltage_current;

  if (last_log_intensity[idx] <= voltage_current - threshold_pos_map[idx]) {
    last_log_intensity[idx] = voltage_current;
    int pos = atomicAdd(event_count, 1);
    events[pos] = Event{timestamp, static_cast<int16_t>(x),
                        static_cast<int16_t>(y), 1, depth_map[idx]};
    refractory_map[idx] = timestamp + refractory_time;
  } else if (last_log_intensity[idx] >=
             voltage_current + threshold_neg_map[idx]) {
    last_log_intensity[idx] = voltage_current;
    int pos = atomicAdd(event_count, 1);
    events[pos] = Event{timestamp, static_cast<int16_t>(x),
                        static_cast<int16_t>(y), 0, depth_map[idx]};
    refractory_map[idx] = timestamp + refractory_time;
  }
}
#endif
#endif
} // namespace dvs