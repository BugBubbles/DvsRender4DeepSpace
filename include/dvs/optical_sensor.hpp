#pragma once
#include "dvs/scene_types.hpp"
#include <cuda_runtime.h>
namespace dvs {
struct OpticalScene {
  const BVHNode<float3> *nodes=nullptr;
  const float3 *vertices=nullptr,*normals=nullptr;
  const uint3 *indices=nullptr;
  const float *albedos=nullptr;
  int root=0,width=0,height=0,samples=1;
  float fx=1,fy=1,cx=0,cy=0;
  HapkeConfig hapke;
};
// Quaternion is (x,y,z,w). Pose/light interpolation is inside the ray kernel.
struct OpticalState {
  float3 position,velocity,light,light_rate;
  float4 rotation;
  float intensity=0,intensity_rate=0;
};
struct OpticalInterval { OpticalState start,end; double t0=0,t1=0; };
struct PhotoParameters {
  double electrons_per_radiance=1, dark_electrons=1;
  double integration_step=1e-6;
  bool shot_noise=true;
};
} // namespace dvs
