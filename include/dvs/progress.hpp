#pragma once

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace dvs {

class ProgressBar {
public:
  explicit ProgressBar(std::size_t total, std::size_t width = 40)
      : total_(total), width_(width), start_(std::chrono::steady_clock::now()) {
  }

  void update(std::size_t completed) {
    if (total_ == 0) {
      return;
    }
    completed_ = completed;
    const double frac =
        static_cast<double>(completed) / static_cast<double>(total_);
    const std::size_t filled =
        static_cast<std::size_t>(frac * static_cast<double>(width_));
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed_sec =
        std::chrono::duration_cast<std::chrono::seconds>(now - start_).count();
    const double eta_sec = frac > 1e-8 ? elapsed_sec / frac - elapsed_sec : 0.0;

    std::ostringstream oss;
    oss << "\r|";
    for (std::size_t i = 0; i < width_; ++i) {
      oss << (i < filled ? '#' : '-');
    }
    oss << "| " << std::fixed << std::setprecision(2) << (frac * 100.0) << "% "
        << "Elapsed: " << to_hms(elapsed_sec)
        << " ETA: " << to_hms(static_cast<long long>(eta_sec));
    std::cout << oss.str() << std::flush;
    if (completed >= total_) {
      std::cout << '\n';
    }
  }

  void update() { update(++completed_); }

private:
  static std::string to_hms(long long total_sec) {
    const long long h = total_sec / 3600;
    const long long m = (total_sec % 3600) / 60;
    const long long s = total_sec % 60;
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << h << ":" << std::setw(2) << m
        << ":" << std::setw(2) << s;
    return oss.str();
  }

  std::size_t total_ = 0, completed_ = 0;
  std::size_t width_ = 0;
  std::chrono::steady_clock::time_point start_;
};

} // namespace dvs
