#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

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

struct DvsParameters {
  float threshold_pos = 0.0f;
  float threshold_neg = 0.0f;
  DvsParameters(float threshold_pos, float threshold_neg)
      : threshold_pos(threshold_pos), threshold_neg(threshold_neg) {}
};

__global__ void batch_generate_events_kernel(
    const float timestamp, const float *intensity_frame, const int width,
    const int height, const float threshold_pos, const float threshold_neg,
    float *last_log_intensity, Event *events, int *event_count);

class DVSBase {
public:
  DVSBase(int height, int width, float threshold_pos,
          float threshold_neg = -1.0f);

  std::vector<Event> process_frame(float timestamp,
                                   const std::vector<float> &intensity_frame);

  int width() const { return width_; }
  int height() const { return height_; }

private:
  int width_ = 0;
  int height_ = 0;
  float threshold_pos_ = 0.0f;
  float threshold_neg_ = 0.0f;
  std::vector<float> last_log_intensity_;
};

} // namespace dvs
