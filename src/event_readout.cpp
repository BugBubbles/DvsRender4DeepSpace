#include "dvs/event_sensor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace dvs {
void EventReadout::submit(std::vector<Event> events, double now, std::vector<Event> &ready) {
  ready.clear();
  std::stable_sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
    if (a.timestamp != b.timestamp) return a.timestamp < b.timestamp;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
  });
  for (const auto &event : events) {
    // Retire completions before each arrival so capacity means pending backlog.
    while (!pending_.empty() && pending_.front().ready_time <= event.timestamp) {
      ready.push_back(pending_.front().event); pending_.pop_front();
    }
    if (bandwidth_ <= 0) { ready.push_back(event); continue; }
    if (pending_.size() >= capacity_) { ++dropped_; continue; }
    double completed = std::max(available_, static_cast<double>(event.timestamp)) + 1.0 / bandwidth_;
    available_ = completed;
    Event delivered = event;
    delivered.timestamp = completed + latency_;
    pending_.push_back({completed + latency_, delivered});
    peak_queue_ = std::max(peak_queue_, pending_.size());
  }
  while (!pending_.empty() && pending_.front().ready_time <= now) {
    ready.push_back(pending_.front().event); pending_.pop_front();
  }
}
void EventReadout::drain(std::vector<Event> &ready) {
  ready.clear();
  while (!pending_.empty()) { ready.push_back(pending_.front().event); pending_.pop_front(); }
}
} // namespace dvs
