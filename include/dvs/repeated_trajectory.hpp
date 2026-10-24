#pragma once
#include "dvs/trajectory.hpp"
#include <cmath>
#include <stdexcept>
namespace dvs {
// Repeat the supplied path at its original time scale, with C2 phase reversals.
// The reversal reaches each source endpoint without teleporting camera pose.
class RepeatedTrajectory {
public:
  explicit RepeatedTrajectory(const CameraTrajectory &source,double turn_seconds=.5)
    :source_(source),duration_(source.end_time()-source.start_time()),turn_(std::min(turn_seconds,duration_/4)) {
    if(!std::isfinite(duration_) || !std::isfinite(turn_seconds) || duration_<=0 || turn_seconds<=0)
      throw std::invalid_argument("finite positive trajectory and turn duration required");
  }
  CameraTrajectoryState evaluate(double time) const {
    if(!std::isfinite(time))throw std::invalid_argument("finite trajectory query time required");
    double r=std::fmod(time,2*duration_);if(r<0)r+=2*duration_;
    double phase=r<=duration_?r:2*duration_-r;
    double rate=r<=duration_?1:-1;
    if(phase<turn_) {
      double x=phase/turn_;phase=turn_*(3*x*x-3*x*x*x+x*x*x*x);
      rate*=6*x-9*x*x+4*x*x*x;
    }else if(duration_-phase<turn_) {
      double x=(duration_-phase)/turn_;phase=duration_-turn_*(3*x*x-3*x*x*x+x*x*x*x);
      rate*=6*x-9*x*x+4*x*x*x;
    }
    auto state=source_.evaluate(source_.start_time()+phase);
    state.timestamp=time;state.linear_velocity*=rate;state.angular_velocity*=rate;
    state.light_direction_rate*=rate;state.light_intensity_rate*=rate;
    return state;
  }
  double source_duration()const{return duration_;}
private:
  const CameraTrajectory &source_;double duration_,turn_;
};
}
