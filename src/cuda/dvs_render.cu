#include "dvs/raycast.hpp"
namespace dvs {
DvsRayCaster::DvsRayCaster(const SceneBundle<float3,uint3> &scene,const Config &config,int root)
  :DvsRayCaster(std::make_shared<DeviceScene>(scene,config,root),config) {}
DvsRayCaster::DvsRayCaster(std::shared_ptr<DeviceScene> scene,const Config &config)
  :scene_(std::move(scene)),
   sensor_(std::make_unique<EventSensor>(config.camera.width,config.camera.height,config.physic)) {}
DvsRayCaster::~DvsRayCaster()=default;
void DvsRayCaster::initialize(const CameraTrajectoryState &state,double timestamp) {
  sensor_->initialize(scene_->view(),to_optical_state(state),timestamp);
}
void DvsRayCaster::advance(const CameraTrajectoryState &start,const CameraTrajectoryState &end,
                         double t0,double t1,std::vector<Event> &events) {
  sensor_->process(scene_->view(),OpticalInterval{to_optical_state(start),to_optical_state(end),t0,t1},events);
}
} // namespace dvs
