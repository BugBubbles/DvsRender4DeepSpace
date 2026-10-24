#include "bvh/build.h"
#include "dvs/raycast.hpp"
#include "dvs/device_optics.cuh"
#include <Eigen/Geometry>
#include <iostream>
#include <stdexcept>
namespace dvs {
CamRayCaster::CamRayCaster(const SceneBundle<float3,uint3> &scene,const Config &config,int root)
  :CamRayCaster(std::make_shared<DeviceScene>(scene,config,root)) {}
CamRayCaster::CamRayCaster(std::shared_ptr<DeviceScene> scene)
  :scene_(std::move(scene)),ray_count_(scene_->view().width*scene_->view().height) {
  sampler_=std::make_unique<OpticalSampler>(scene_->view().width,scene_->view().height,scene_->view().samples);
}
CamRayCaster::~CamRayCaster()=default;

void CamRayCaster::render_main(const float3 &origin, const float *world,
                               const float3 &light, float intensity,
                               std::vector<float> &image, std::vector<float> &depth) {
  auto scene=scene_->view();
  Eigen::Map<const Mat3> rotation(world);
  Eigen::Quaternionf q(rotation);q.normalize();
  OpticalState pose{};pose.position=origin;pose.light=light;pose.intensity=intensity;
  pose.rotation=make_float4(q.x(),q.y(),q.z(),q.w());
  sampler_->evaluate(scene,{pose});std::vector<OpticalValue> values;sampler_->download(values);
  image.resize(ray_count_);depth.resize(ray_count_);
  for(int i=0;i<ray_count_;++i){image[i]=values[i].radiance;depth[i]=values[i].depth;}

}
void CamRayCaster::render_main(const float3 &origin,const float *world,
                               const float3 &light,float intensity,std::vector<float> &image) {
  std::vector<float> depth;
  render_main(origin,world,light,intensity,image,depth);
}
} // namespace dvs
