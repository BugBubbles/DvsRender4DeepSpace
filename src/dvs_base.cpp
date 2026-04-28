#include "dvs/dvs_base.hpp"

#include <stdexcept>

namespace dvs {

DVSBase::DVSBase(int height, int width, float threshold_pos, float threshold_neg)
    : width_(width), height_(height), threshold_pos_(threshold_pos) {
    if (height_ <= 0 || width_ <= 0) {
        throw std::invalid_argument("height and width must be positive");
    }
    if (threshold_pos_ <= 0.0f) {
        throw std::invalid_argument("threshold_pos must be positive");
    }
    threshold_neg_ = (threshold_neg > 0.0f) ? threshold_neg : threshold_pos_;
    last_log_intensity_.assign(static_cast<std::size_t>(height_ * width_), 0.0f);
}

std::vector<Event> DVSBase::process_frame(
    float timestamp, const std::vector<float>& log_intensity_frame) {
    if (log_intensity_frame.size() != last_log_intensity_.size()) {
        throw std::invalid_argument("log_intensity_frame size mismatch");
    }

    std::vector<Event> events;
    events.reserve(log_intensity_frame.size() / 10);

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y * width_ + x);
            const float current = log_intensity_frame[idx];
            if (last_log_intensity_[idx] <= current - threshold_pos_) {
                last_log_intensity_[idx] = current;
                events.push_back(Event{timestamp, x, y, 1});
            } else if (last_log_intensity_[idx] >= current + threshold_neg_) {
                last_log_intensity_[idx] = current;
                events.push_back(Event{timestamp, x, y, -1});
            }
        }
    }

    return events;
}

}  // namespace dvs
