#include "dvs/device_scene.hpp"
#include "bvh/build.h"
#include "device_types.h"
#include <stdexcept>
#include <iostream>
namespace dvs {
namespace {
template<class T> void upload(const T *&destination,const std::vector<T> &source) {
  T *storage=nullptr;
  CUDA_CHECK(cudaMalloc(&storage,sizeof(T)*source.size()),"allocate scene array");
  destination=storage;
  CUDA_CHECK(cudaMemcpy(storage,source.data(),sizeof(T)*source.size(),cudaMemcpyHostToDevice),"upload scene array");
}
}
DeviceScene::DeviceScene(const SceneBundle<float3,uint3> &mesh,const Config &config,int root) {
  if(mesh.bvh_nodes.empty() || mesh.bvh_triangles.empty() || mesh.vertices.empty() ||
     root<0 || root>=mesh.bvh_nodes.size() || mesh.normals.size()!=mesh.vertices.size() ||
     (mesh.use_albedo && mesh.albedos.size()!=mesh.vertices.size()))
    throw std::invalid_argument("invalid indexed mesh/BVH/material dimensions");
  BVHBuilder::validate(mesh.bvh_nodes,root);
  for(auto triangle:mesh.bvh_triangles)
    if(triangle.x>=mesh.vertices.size() || triangle.y>=mesh.vertices.size() || triangle.z>=mesh.vertices.size())
      throw std::invalid_argument("mesh triangle index outside vertex array");
  scene_.root=root;scene_.width=config.camera.width;scene_.height=config.camera.height;
  scene_.fx=config.camera.fx;scene_.fy=config.camera.fy;scene_.cx=config.camera.cx;scene_.cy=config.camera.cy;
  scene_.samples=config.scene.supersample_factor;scene_.hapke=config.hapke;
  try {
    upload(scene_.nodes,mesh.bvh_nodes);upload(scene_.vertices,mesh.vertices);
    upload(scene_.normals,mesh.normals);upload(scene_.indices,mesh.bvh_triangles);
    if(mesh.use_albedo) {
      size_t fallback=0;
      for(auto w:mesh.albedos)if(!std::isfinite(w) || w<0 || w>1) ++fallback;
      if(fallback) std::cout<<"Material fallback: "<<fallback<<" vertices use configured homogeneous SSA w="<<config.hapke.w<<"\n";
      upload(scene_.albedos,mesh.albedos);
    }
  } catch(...) {release();throw;}
}
void DeviceScene::release() {
  cudaFree(const_cast<BVHNode<float3>*>(scene_.nodes));cudaFree(const_cast<float3*>(scene_.vertices));
  cudaFree(const_cast<float3*>(scene_.normals));cudaFree(const_cast<uint3*>(scene_.indices));cudaFree(const_cast<float*>(scene_.albedos));
  scene_.nodes=nullptr;scene_.vertices=scene_.normals=nullptr;scene_.indices=nullptr;scene_.albedos=nullptr;
}
DeviceScene::~DeviceScene(){release();}
}
