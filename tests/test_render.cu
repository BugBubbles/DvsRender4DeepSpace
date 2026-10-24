#include "dvs/raycast.hpp"
#include "dvs/scene_data.hpp"
#include "dvs/hapke.hpp"
#include "dvs/blenderize.hpp"
#include "dvs/device_optics.cuh"
#include "bvh/build.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>
using namespace dvs;
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
__global__ void reference_pixels(OpticalScene scene,OpticalState pose,OpticalValue *values){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=scene.width*scene.height)return;
  auto result=sample_pixel(scene,i,pose);values[i]={result.radiance,result.depth,scene.samples};
}
__global__ void check_shadow_origins(OpticalScene scene,int *failures){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=10000)return;
  auto tri=scene.indices[0];float u=(i%100+.5f)/101,v=(i/100+.5f)/101*(1-u);
  auto a=scene.vertices[tri.x],b=scene.vertices[tri.y],c=scene.vertices[tri.z];
  float3 origin=shadow_origin(scene,tri,u,v,make_float3(0,0,1));
  // Assess the geometric plane in double precision, independently of the
  // float offset computation at large planetary-map coordinates.
  double nx=double(b.y-a.y)*(c.z-a.z)-double(b.z-a.z)*(c.y-a.y);
  double ny=double(b.z-a.z)*(c.x-a.x)-double(b.x-a.x)*(c.z-a.z);
  double nz=double(b.x-a.x)*(c.y-a.y)-double(b.y-a.y)*(c.x-a.x);
  double side=nx*(double(origin.x)-a.x)+ny*(double(origin.y)-a.y)+nz*(double(origin.z)-a.z);
  if(!(side>0))atomicAdd(failures,1);
}
int main(int argc,char **argv) {try {
  require(argc==3,"usage: test_render config output");
  std::string error;Config config(argv[1],error);require(error.empty(),error.c_str());
  std::filesystem::path output=argv[2];std::filesystem::create_directories(output);
  std::filesystem::copy_file(argv[1],output/"source_config.yaml",std::filesystem::copy_options::overwrite_existing);
  SceneBundle<float3,uint3> mesh;
  // Half-plane on +Y checks image y-down versus Blender pose y-up.
  mesh.vertices={make_float3(-100,0,0),make_float3(100,0,0),make_float3(100,100,0),make_float3(-100,100,0)};
  mesh.normals=std::vector<float3>(4,make_float3(0,0,1));
  mesh.bvh_triangles={make_uint3(0,1,2),make_uint3(0,2,3)};
  int root=BVHBuilder::build(BVHBuildParams{},mesh.bvh_nodes,mesh.bvh_triangles,mesh.vertices);
  auto device_scene=std::make_shared<DeviceScene>(mesh,config,root);
  CamRayCaster camera(device_scene);
  Mat3 rotation=Mat3::Identity();float3 position=make_float3(0,0,10),light=make_float3(0,0,1);
  std::vector<float> image,depth,next_image,next_depth;
  camera.render_main(position,rotation.data(),light,1000,image,depth);
  camera.render_main(position,rotation.data(),light,1000,next_image,next_depth);
  require(image==next_image && depth==next_depth,"repeat render changed image/depth: stale buffer or sampling noise");
  // Independently retain serial native quadrature as a correctness oracle.
  OpticalState optical{};optical.position=position;optical.rotation=make_float4(0,0,0,1);
  optical.light=light;optical.intensity=1000;
  OpticalSampler sampler(config.camera.width,config.camera.height,config.scene.supersample_factor);
  sampler.evaluate(device_scene->view(),{optical});std::vector<OpticalValue> parallel;sampler.download(parallel);
  OpticalValue *reference_device;CUDA_CHECK(cudaMalloc(&reference_device,sizeof(OpticalValue)*parallel.size()),"reference image");
  reference_pixels<<<(parallel.size()+127)/128,128>>>(device_scene->view(),optical,reference_device);
  std::vector<OpticalValue> reference(parallel.size());
  CUDA_CHECK(cudaMemcpy(reference.data(),reference_device,sizeof(OpticalValue)*reference.size(),cudaMemcpyDeviceToHost),"reference download");cudaFree(reference_device);
  double maximum_radiance_error=0;float maximum_depth_error=0;
  for(size_t i=0;i<parallel.size();++i){
    maximum_radiance_error=std::max(maximum_radiance_error,std::abs(parallel[i].radiance-reference[i].radiance));
    maximum_depth_error=std::max(maximum_depth_error,std::abs(parallel[i].depth-reference[i].depth));
    require(parallel[i].samples==config.scene.supersample_factor,"parallel quadrature changed configured grid");
  }
  std::cout<<"parallel max radiance absolute error="<<maximum_radiance_error<<" depth error="<<maximum_depth_error<<"\n";
  require(maximum_radiance_error<1e-10 && maximum_depth_error==0,"parallel rays changed native radiance or depth");
  std::cout<<"PASS parallel/full serial quadrature matches within floating-point summation rounding at native camera configuration\n";
  int column=config.camera.cx;
  require(image[column]>0 && depth[column]>0,"+world Y must appear in upper image");
  require(image[(config.camera.height-1)*config.camera.width+column]==0 &&
          depth[(config.camera.height-1)*config.camera.width+column]==-1,"miss ray must have zero light and depth -1");
  int row=std::round(config.camera.cy);
  double range=0;int hits=0;int ss=config.scene.supersample_factor;
  for(int sy=0;sy<ss;++sy)for(int sx=0;sx<ss;++sx) {
    double px=(column+(sx+.5)/ss-.5-config.camera.cx)/config.camera.fx;
    double py=(row+(sy+.5)/ss-.5-config.camera.cy)/config.camera.fy;
    if(py<=0){range+=10*std::sqrt(1+px*px+py*py);++hits;}
  }
  require(hits>0 && hits<ss*ss,"fixture must cover only part of boundary pixel");
  require(std::abs(depth[row*config.camera.width+column]-range/hits)<1e-5,"depth must average hits, excluding misses");
  CameraTrajectoryState state;state.position=Vec3(0,0,10);state.orientation=rotation;
  state.quaternion=Eigen::Quaternionf::Identity();state.light_direction=Vec3(0,0,1);state.light_intensity=1000;
  DvsRayCaster dvs(device_scene,config);dvs.initialize(state,0);
  std::vector<Event> events;dvs.advance(state,state,0,config.physic.circuit_step,events);dvs.drain(events);
  require(events.empty(),"configured short static interval produced a startup burst");
  cv::imwrite((output/"depth.tif").string(),cv::Mat(config.camera.height,config.camera.width,CV_32F,depth.data()));
  cv::imwrite((output/"linear_radiance.tif").string(),cv::Mat(config.camera.height,config.camera.width,CV_32F,image.data()));
  std::vector<uint8_t> bytes;render_float_to_png_bytes(image.data(),config.camera.width,config.camera.height,0,bytes);
  cv::imwrite((output/"image.png").string(),cv::Mat(config.camera.height,config.camera.width,CV_8U,bytes.data()));
  std::ofstream metadata(output/"result.txt");
  metadata<<"Full source configuration; no camera or supersampling override.\n"
          <<"width="<<config.camera.width<<" height="<<config.camera.height<<" samples="<<ss*ss
          <<"\nRepeated image/depth identical; partial hit-depth error="<<std::abs(depth[row*config.camera.width+column]-range/hits)
          <<"\nStatic event interval="<<config.physic.circuit_step<<" seconds; count="<<events.size()<<"\n";
  std::cout<<"PASS full configured camera "<<config.camera.width<<"x"<<config.camera.height<<": orientation, misses, partial depth, repeatability, shared DVS initialization\n";
  {
    SceneBundle<float3,uint3> slope;
    slope.vertices={make_float3(13000,-13000,4000),make_float3(13005,-13000,4001.5),make_float3(13000,-12995,4000.75)};
    slope.normals=std::vector<float3>(3,normalize(cross(slope.vertices[1]-slope.vertices[0],slope.vertices[2]-slope.vertices[0])));
    slope.bvh_triangles={make_uint3(0,1,2)};int root=BVHBuilder::build(BVHBuildParams{},slope.bvh_nodes,slope.bvh_triangles,slope.vertices);
    DeviceScene large_coordinates(slope,config,root);int *failures;CUDA_CHECK(cudaMalloc(&failures,sizeof(int)),"shadow origin test status");
    CUDA_CHECK(cudaMemset(failures,0,sizeof(int)),"clear shadow origin test");
    check_shadow_origins<<<(10000+127)/128,128>>>(large_coordinates.view(),failures);int failed;
    CUDA_CHECK(cudaMemcpy(&failed,failures,sizeof(int),cudaMemcpyDeviceToHost),"shadow origin test status");cudaFree(failures);
    require(failed==0,"shadow origins remained inside the geometric surface at km-scale coordinates");
    std::cout<<"PASS 10000 km-coordinate shadow origins lie outside a tilted geometric plane\n";
  }
  // Radiance-factor inverse must recover SSA over distinct viewing geometries.
  for(float w:{0.f,.05f,.324f,.7f,.95f,1.f}) for(float angle:{0.f,15.f,45.f,80.f}) {
    float ci=std::cos(angle*kPi/180),target=kPi*hapke_reflectance(ci,1,ci,w,config.hapke);
    float inverse=invert_hapke_if(angle,0,angle,target,config.hapke);
    require(std::abs(inverse-w)<2e-6f,"I/F forward/inverse mismatch");
  }
  require(std::isnan(invert_hapke_if(20,10,90,.1,config.hapke)),"invalid angular geometry accepted");
  require(std::isnan(invert_hapke_if(90,0,90,.1,config.hapke)),"grazing input accepted");
  // Mount translation/rotation is local to the trajectory frame; world light stays fixed.
  FrameData<Vec3,Mat3> frame{};frame.camera_origin=Vec3(10,20,30);
  frame.camera_world=Eigen::AngleAxisf(kPi/2,Vec3::UnitZ()).toRotationMatrix();
  frame.light_dir=Vec3(0,0,1);frame.light_intensity=1000;frame.timestamp_seconds=.125;
  auto last=frame;last.timestamp_seconds=.5;
  auto scene=config.scene;scene.translation={1,0,0};scene.rotation={kPi/2,0,0};
  CameraTrajectory trajectory({frame,last},scene);
  auto pose=trajectory.evaluate(.125);
  require((pose.position-Vec3(10,21,30)).norm()<1e-5,"local mount translation incorrect");
  Mat3 expected=frame.camera_world*Eigen::AngleAxisf(kPi/2,Vec3::UnitX()).toRotationMatrix();
  require((pose.orientation-expected).norm()<1e-5,"local mount rotation incorrect");
  require(pose.light_direction==frame.light_dir && trajectory.start_time()==.125 && trajectory.end_time()==.5,"world light or seconds converted incorrectly");
  bool failed=false;try{CameraTrajectory duplicate({frame,frame},scene);}catch(const std::invalid_argument&){failed=true;}
  require(failed,"duplicate timestamps were silently invented");
  last.timestamp_seconds=std::numeric_limits<double>::infinity();failed=false;
  try{CameraTrajectory invalid({frame,last},scene);}catch(const std::invalid_argument&){failed=true;}
  require(failed,"infinite seconds silently replaced by frame index");
  std::cout<<"PASS SSA inverse round trips, invalid geometry, local mount and explicit seconds\n";
  return 0;
}catch(const std::exception &e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
