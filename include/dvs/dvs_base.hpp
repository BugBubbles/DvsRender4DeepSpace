#pragma once

#include <vector>

namespace dvs {

struct Event {
    float timestamp = 0.0f;
    int x = 0;
    int y = 0;
    int polarity = 0;
};

class DVSBase {
public:
    DVSBase(int height, int width, float threshold_pos, float threshold_neg = -1.0f);

    std::vector<Event> process_frame(float timestamp, const std::vector<float>& log_intensity_frame);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_ = 0;
    int height_ = 0;
    float threshold_pos_ = 0.0f;
    float threshold_neg_ = 0.0f;
    std::vector<float> last_log_intensity_;
};

}  // namespace dvs
