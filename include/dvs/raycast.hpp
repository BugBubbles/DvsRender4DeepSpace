#pragma once

#include <vector>
#include <memory>
#include "dvs/event_sensor.hpp"
#include "dvs/device_scene.hpp"
#include "dvs/trajectory.hpp"

#include "device_types.h"
#include "dvs/dvs_base.hpp"
#include "dvs/scene_types.hpp"
#include <cuda_runtime.h>
#include <vector_types.h>


namespace dvs {
inline OpticalState to_optical_state(const CameraTrajectoryState &s){
  OpticalState p{};p.position=vec3_to_float3(s.position);p.velocity=vec3_to_float3(s.linear_velocity);
  p.light=vec3_to_float3(s.light_direction);p.light_rate=vec3_to_float3(s.light_direction_rate);
  p.intensity=s.light_intensity;p.intensity_rate=s.light_intensity_rate;
  p.rotation=make_float4(s.quaternion.x(),s.quaternion.y(),s.quaternion.z(),s.quaternion.w());return p;
}
class DvsRayCaster {
public:
  DvsRayCaster(const SceneBundle<float3, uint3> &scene, const Config &config, int root_index);
  DvsRayCaster(std::shared_ptr<DeviceScene> scene,const Config &config);
  ~DvsRayCaster();
  DvsRayCaster(const DvsRayCaster &) = delete;
  DvsRayCaster &operator=(const DvsRayCaster &) = delete;
  void initialize(const CameraTrajectoryState &state,double timestamp);
  void advance(const CameraTrajectoryState &start,const CameraTrajectoryState &end,
               double t0,double t1,std::vector<Event> &events);
  void advance_many(const std::vector<OpticalInterval>& intervals,std::vector<Event>& events) {
    sensor_->process_many(scene_->view(),intervals,events);
  }
  int adaptive_cache_capacity(){
    if(!sampler_)sampler_=std::make_unique<OpticalSampler>(scene_->view().width,scene_->view().height,scene_->view().samples);
    return sampler_->knot_capacity();
  }
  bool advance_adaptive(const std::vector<OpticalInterval>& intervals,double tolerance,std::vector<Event>& events,bool check_convergence=false){
    if(!sampler_)sampler_=std::make_unique<OpticalSampler>(scene_->view().width,scene_->view().height,scene_->view().samples);
    return sensor_->process_adaptive(scene_->view(),intervals,*sampler_,tolerance,events,check_convergence);
  }
  ReceptorProbeResult last_convergence()const{return sensor_->last_convergence();}
  void enable_numerical_diagnostics(){sensor_->enable_numerical_diagnostics();}
  ReceptorProbeResult verify_receptor(const std::vector<OpticalInterval>& intervals){
    if(!sampler_)throw std::logic_error("no optical knots for receptor diagnostic");
    return sensor_->verify_receptor(scene_->view(),*sampler_,intervals);
  }
  OpticalProbeResult verify_optics(const std::vector<OpticalInterval>& intervals,double tolerance){
    if(!sampler_)throw std::logic_error("adaptive sensor has no optical knots");
    return sensor_->verify_optics(scene_->view(),*sampler_,intervals,tolerance);
  }
  OpticalRefinementStats refinement_statistics()const{return sampler_?sampler_->statistics():OpticalRefinementStats{};}
  void drain(std::vector<Event> &events) { sensor_->drain(events); }
  std::size_t dropped_events() const { return sensor_->dropped(); }
  std::size_t peak_queue() const { return sensor_->peak_queue(); }
private:
  std::shared_ptr<DeviceScene> scene_;
  std::unique_ptr<EventSensor> sensor_;
  std::unique_ptr<OpticalSampler> sampler_;
};

class CamRayCaster {
public:
  CamRayCaster(const SceneBundle<float3, uint3> &scene, const Config &config,
               const int root_index);
  explicit CamRayCaster(std::shared_ptr<DeviceScene> scene);
  ~CamRayCaster();
  CamRayCaster(const CamRayCaster &)=delete;
  CamRayCaster &operator=(const CamRayCaster &)=delete;

  void render_main(const float3 &camera_origin, const float *camera_world,
                   const float3 &light_dir, const float light_intensity,
                   std::vector<float> &light_frame,
                   std::vector<float> &depth_map);

  void render_main(const float3 &camera_origin, const float *camera_world,
                   const float3 &light_dir, const float light_intensity,
                   std::vector<float> &light_frame);

private:
  std::shared_ptr<DeviceScene> scene_;
  int ray_count_=0;
  std::unique_ptr<OpticalSampler> sampler_;
};

} // namespace dvs
