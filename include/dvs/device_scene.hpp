#pragma once
#include "dvs/optical_sensor.hpp"
namespace dvs {
// One owner for BVH, indexed mesh and material storage shared by both renderers.
class DeviceScene {
public:
  DeviceScene(const SceneBundle<float3,uint3> &,const Config &,int root);
  ~DeviceScene();
  DeviceScene(const DeviceScene &)=delete;
  DeviceScene &operator=(const DeviceScene &)=delete;
  const OpticalScene &view() const {return scene_;}
private:
  void release();
  OpticalScene scene_;
};
}
