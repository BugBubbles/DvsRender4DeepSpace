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
    const float refractory_time, const float shot_intensity,
    const float *depth_map, float *refractory_map, float *last_log_intensity,
    Event *events, int *event_count, curandState *rand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  if (timestamp < refractory_map[idx])
    return;
  // 模拟散粒噪声
  // curandState local_state = rand_states[idx];
  // float shot_noise_prob =
  //     shot_noise_freq * (timestamp - refractory_map[idx]); //
  //     转换为每秒的概率
  // if (curand_uniform(rand_states + idx) < shot_noise_prob) {
  //   current += shot_intensity;
  // }
  float lambda = intensity_frame[idx] * shot_intensity;
  if (lambda < 0 || isnan(lambda) || isinf(lambda)) {
    // 处理无效的lambda值，避免curand_poisson崩溃
    lambda = 0.0f;
  }
  float shot_frame = curand_poisson(rand_states + idx, lambda) / shot_intensity;
  float current = shot_frame + dark_current;
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
    const float refractory_time, const float alpha, const float shot_intensity,
    const float *depth_map, float *refractory_map, float *last_log_intensity,
    float *last_voltage, Event *events, int *event_count,
    curandState *rand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  if (timestamp < refractory_map[idx])
    return;
  // 模拟散粒噪声
  // curandState local_state = rand_states[idx];
  // float shot_noise_prob =
  //     shot_noise_freq * (timestamp - refractory_map[idx]); //
  //     转换为每秒的概率
  // if (curand_uniform(rand_states + idx) < shot_noise_prob) {
  //   current += shot_intensity;
  // }
  float lambda = intensity_frame[idx] * shot_intensity;
  if (lambda < 0 || isnan(lambda) || isinf(lambda)) {
    // 处理无效的lambda值，避免curand_poisson崩溃
    lambda = 0.0f;
  }
  float shot_frame = curand_poisson(rand_states + idx, lambda) / shot_intensity;
  float current = shot_frame + dark_current;
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
#ifdef ENABLE_AER_DELAY
__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float *threshold_pos_map,
    const float *threshold_neg_map, const float dark_current,
    const float refractory_time, const float alpha, const float shot_intensity,
    const float *depth_map, float *refractory_map, float *last_log_intensity,
    float *last_voltage, Event *events, int *event_count_row, int *event_count,
    curandState *rand_states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int idx = y * width + x;
  if (timestamp < refractory_map[idx])
    return;
  // 模拟散粒噪声
  // float current = intensity_frame[idx] + dark_current;
  // float shot_noise_prob = 10 * (timestamp - refractory_map[idx]);
  // if (curand_uniform(rand_states + idx) < shot_noise_prob) {
  //   current += 3.0f;
  // }
  float lambda = intensity_frame[idx] * shot_intensity;
  if (lambda < 0 || isnan(lambda) || isinf(lambda)) {
    // 处理无效的lambda值，避免curand_poisson崩溃
    lambda = 0.0f;
  }
  float shot_frame = curand_poisson(rand_states + idx, lambda) / shot_intensity;
  float current = shot_frame + dark_current;
  // rand_states[idx] = local_state;
  // 光强转对数值
  float log_current = logf(current);
  // 高通滤波
  float voltage_current = (1 - alpha) * last_voltage[idx] + alpha * log_current;
  last_voltage[idx] = voltage_current;

  if (last_log_intensity[idx] <= voltage_current - threshold_pos_map[idx]) {
    last_log_intensity[idx] = voltage_current;
    int pos = atomicAdd(event_count, 1);
    // 统计第y行的事件数量
    atomicAdd(event_count_row + y, 1);
    events[pos] = Event{timestamp, static_cast<int16_t>(x),
                        static_cast<int16_t>(y), 1, depth_map[idx]};
    refractory_map[idx] = timestamp + refractory_time;
  } else if (last_log_intensity[idx] >=
             voltage_current + threshold_neg_map[idx]) {
    last_log_intensity[idx] = voltage_current;
    int pos = atomicAdd(event_count, 1);
    // 统计第y行的事件数量
    atomicAdd(event_count_row + y, 1);
    events[pos] = Event{timestamp, static_cast<int16_t>(x),
                        static_cast<int16_t>(y), 0, depth_map[idx]};
    refractory_map[idx] = timestamp + refractory_time;
  }
}
void row_delay_buffer(const int height, const int *event_count_row,
                      const float unit_row_delay, const float unit_col_delay,
                      float *delay_row, int *last_row_idx) {
  // last_row_idx:
  // 上一帧产生事件的最后一行索引，当前帧应当从它的下一行开始（如有事件产生》
  // 循环遍历第一行
  int request_count = 0; // 统计请求次数，即有事件产生的行数
  int current_row_idx = last_row_idx[0];
  for (int y = 0; y < height; ++y) {
    int row_idx = (y + last_row_idx[0] + 1) % height; // 当前行索引
    if (event_count_row[row_idx] > 0) {
      delay_row[row_idx] =
          unit_col_delay * event_count_row[row_idx] + // 行内事件处理时间 +
          request_count * unit_row_delay;             // 行间处理延迟总计
      request_count++; // 有事件产生，统计有效行数
      current_row_idx = row_idx;
    } else {
      delay_row[row_idx] = 0.0f; // 没有事件产生，延迟为0
    }
  }

  last_row_idx[0] = current_row_idx; // 更新最后一行索引
}

__global__ void batch_aer_kernel(const int *events_count, Event *events,
                                 float *delay_row) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= events_count[0])
    return;
  int y = events[idx].y;
  events[idx].timestamp += delay_row[y]; // 根据行延迟调整事件时间
}
#endif
#endif
#endif
} // namespace dvs