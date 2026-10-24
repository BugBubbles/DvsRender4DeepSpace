#pragma once
#include "dvs/dvs_base.hpp"
#include "dvs/pixel_model.hpp"
#include "dvs/optical_sensor.hpp"
#include "dvs/scene_types.hpp"
#include "dvs/optical_sampler.hpp"
#include <deque>
#include <vector>

namespace dvs {
// Diagnostic timings use native camera/config and independent scratch states;
// they do not produce or change simulation outputs.
struct KernelResources {
  int registers=0,local_bytes=0,shared_bytes=0,active_blocks_per_sm=0;
};
struct SensorWorkProfile {
  int pixels=0,circuit_cells=0,multiprocessors=0,max_threads_per_sm=0;
  KernelResources fused,optical,circuit;
  double optical_gpu_ms=0,circuit_gpu_ms=0;
};
SensorWorkProfile profile_sensor_work(const OpticalScene&,const OpticalInterval&,const PhysicConfig&);
// Generic FIFO readout, separate from pixel generation. This does not claim to
// reproduce a particular manufacturer's row arbiter or USB packet protocol.
class EventReadout {
public:
  EventReadout(double bandwidth, double latency, std::size_t capacity)
      : bandwidth_(bandwidth), latency_(latency), capacity_(capacity) {}
  void submit(std::vector<Event> events, double now, std::vector<Event> &ready);
  void drain(std::vector<Event> &ready);
  std::size_t dropped() const { return dropped_; }
  std::size_t peak_queue() const { return peak_queue_; }
private:
  struct Pending { double ready_time; Event event; };
  double bandwidth_, latency_, available_ = 0;
  std::size_t capacity_, dropped_ = 0, peak_queue_ = 0;
  std::deque<Pending> pending_;
};
struct ReceptorProbeState {double exact=0,cached=0;};
struct ReceptorProbeResult {double maximum_log_voltage_error=0,p99_log_voltage_error=0,mean_log_voltage_error=0;int pixels=0,circuit_cells=0;};
class EventSensor {
public:
  EventSensor(int width, int height, const PhysicConfig &config);
  ~EventSensor();
  EventSensor(const EventSensor &) = delete;
  EventSensor &operator=(const EventSensor &) = delete;
  void initialize(const OpticalScene &scene, const OpticalState &state, double timestamp);
  void process(const OpticalScene &scene, const OpticalInterval &interval, std::vector<Event> &events);
  void process_many(const OpticalScene &scene,const std::vector<OpticalInterval> &intervals,std::vector<Event> &events);
  bool process_adaptive(const OpticalScene&,const std::vector<OpticalInterval>&,OpticalSampler&,
                        double log_tolerance,std::vector<Event>&,bool check_convergence=false);
  OpticalProbeResult verify_optics(const OpticalScene& scene,OpticalSampler& sampler,const std::vector<OpticalInterval>& ivs,double tolerance){
    return sampler.verify(scene,intervals_,ivs.size(),ivs.front().t0,ivs.back().t1,photo_,tolerance);
  }
  const ReceptorProbeResult& last_convergence()const{return convergence_;}
  void enable_numerical_diagnostics(){diagnostics_=true;}
  ReceptorProbeResult verify_receptor(const OpticalScene&,OpticalSampler&,const std::vector<OpticalInterval>&);
  void drain(std::vector<Event> &events) { readout_.drain(events); }
  std::size_t dropped() const { return readout_.dropped(); }
  std::size_t peak_queue() const { return readout_.peak_queue(); }
private:
  int width_, height_, count_, max_events_;
  unsigned seed_;
  PixelParameters parameters_;
  PhotoParameters photo_;
  double threshold_std_, threshold_min_, last_time_ = 0;
  bool initialized_ = false;
  PixelState *states_ = nullptr;
  curandState *random_ = nullptr;
  Event *events_ = nullptr;
  int *event_count_ = nullptr, *overflow_ = nullptr;
  void upload_intervals(const std::vector<OpticalInterval>&);
  void collect_events(double end,std::vector<Event>&);
  ReceptorProbeResult convergence_{};bool diagnostics_=false;ReceptorProbeState *probe_states_=nullptr;OpticalKnots diagnostic_knots_{};
  int *fallback_ids_=nullptr,*fallback_count_=nullptr;
  OpticalCell *fallback_cells_=nullptr;
  OpticalInterval *intervals_=nullptr;
  std::size_t interval_capacity_=0;
  EventReadout readout_;
};
} // namespace dvs
