#include "dvs/raycast.hpp"
#include "dvs/scene_data.hpp"
#include "bvh/build.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <algorithm>
int main(int argc,char**argv){try{
 bool optimized=argc==6 && std::string(argv[5])=="--optimized";
 bool profile=argc==6 && std::string(argv[5])=="--profile";
 if(argc!=5 && !profile && !optimized)throw std::runtime_error("benchmark config dem albedo_or_empty trajectory [--profile|--optimized]");
 std::string error;dvs::Config config(argv[1],error);if(!error.empty())throw std::runtime_error(error);
 dvs::SceneBundle<float3,uint3> scene;std::vector<dvs::FrameData<dvs::Vec3,dvs::Mat3>> frames;
 auto start=std::chrono::steady_clock::now();
 bool ok=std::string(argv[3]).empty()?dvs::load_scene_bundle(argv[2],argv[4],scene,frames,&error):dvs::load_scene_bundle(argv[2],argv[3],argv[4],scene,frames,&error);
 if(!ok)throw std::runtime_error(error);
 dvs::BVHBuildParams params;params.max_leaf_size=64;
 int root=dvs::BVHBuilder::build(params,scene.bvh_nodes,scene.bvh_triangles,scene.vertices);
 dvs::CameraTrajectory trajectory(frames,config.scene);
 auto seconds=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
 std::cout<<std::setprecision(10)<<"mesh_ready_wall_s="<<seconds()<<" vertices="<<scene.vertices.size()<<" triangles="<<scene.bvh_triangles.size()<<std::endl;
 auto device=std::make_shared<dvs::DeviceScene>(scene,config,root);
 dvs::DvsRayCaster sensor(device,config);if(optimized)sensor.enable_numerical_diagnostics();double t0=trajectory.start_time();auto initial=trajectory.evaluate(t0);
 start=std::chrono::steady_clock::now();sensor.initialize(initial,t0);std::cout<<"initialization_wall_s="<<seconds()<<std::endl;
 if(optimized){

  double now=t0,dt=config.scene.time_resolution,tolerance=dvs::kDefaultOpticalLogTolerance;size_t step=0;
  std::vector<dvs::Event> ready;std::vector<dvs::OpticalInterval> intervals;
  double times[5];
  for(int run=0;run<5;++run){
   intervals.clear();auto left=dvs::to_optical_state(trajectory.evaluate(now));
   for(int j=0;j<200;++j){double next=t0+(++step)*dt;auto right=dvs::to_optical_state(trajectory.evaluate(next));
    intervals.push_back({left,right,now,next});left=right;now=next;}
   start=std::chrono::steady_clock::now();sensor.advance_adaptive(intervals,tolerance,ready);times[run]=seconds();
   auto stats=sensor.refinement_statistics();auto probe=sensor.verify_optics(intervals,tolerance);auto receptor=sensor.verify_receptor(intervals);
   std::cout<<"cache_capacity="<<sensor.adaptive_cache_capacity()<<" adaptive_run="<<run<<" simulation_s="<<intervals.back().t1-intervals.front().t0<<" wall_s="<<times[run]
    <<" events="<<ready.size()<<" optical_pixel_samples="<<stats.traced_pixel_samples<<" fallback_pixels="<<stats.fallback_pixels
    <<" local_exact_queries="<<stats.native_cell_queries<<" exact_segments="<<stats.exact_segments<<" maximum_knots="<<stats.maximum_knots<<" probe_count="<<probe.probes<<" probe_mean_log_error="<<probe.mean_log_error
    <<" probe_p99_log_error="<<probe.p99_log_error<<" probe_max_log_error="<<probe.maximum_log_error
    <<" probes_over_tolerance="<<probe.over_tolerance<<" fallback_probes="<<probe.exact_fallback_probes<<" mean_receptor_log_error="<<receptor.mean_log_voltage_error
    <<" p99_receptor_log_error="<<receptor.p99_log_voltage_error<<" max_receptor_log_error="<<receptor.maximum_log_voltage_error<<std::endl;
  }
  std::sort(times,times+5);std::cout<<"adaptive_median_wall_s="<<times[2]<<" estimate_60s_wall_s="<<times[2]*60/(200*dt)
    <<" native_width="<<config.camera.width<<" native_height="<<config.camera.height<<" configured_ss="<<config.scene.supersample_factor
    <<" configured_circuit_step="<<config.physic.circuit_step<<std::endl;
  dvs::CamRayCaster camera(device);std::vector<float> image,depth;start=std::chrono::steady_clock::now();
  camera.render_main(dvs::vec3_to_float3(initial.position),initial.orientation.data(),dvs::vec3_to_float3(initial.light_direction),initial.light_intensity,image,depth);
  std::cout<<"parallel_frame_wall_s="<<seconds()<<std::endl;return 0;
 }
 std::vector<dvs::Event> events;double t1=t0+double(config.scene.time_resolution);
 start=std::chrono::steady_clock::now();sensor.advance(initial,trajectory.evaluate(t1),t0,t1,events);
 std::cout<<"horizon_sim_s="<<t1-t0<<" horizon_wall_s="<<seconds()<<" estimate_60s_wall_s="<<seconds()*60/(t1-t0)<<" events="<<events.size()<<std::endl;
 if(profile){
  double warm_wall[3],warm_gpu[3];
  for(int j=0;j<3;++j){
   double next=t0+(j+2)*double(config.scene.time_resolution);
   cudaEvent_t before,after;cudaEventCreate(&before);cudaEventCreate(&after);cudaEventRecord(before);
   start=std::chrono::steady_clock::now();sensor.advance(trajectory.evaluate(t1),trajectory.evaluate(next),t1,next,events);
   warm_wall[j]=seconds();cudaEventRecord(after);cudaEventSynchronize(after);float ms;cudaEventElapsedTime(&ms,before,after);warm_gpu[j]=ms;
   cudaEventDestroy(before);cudaEventDestroy(after);t1=next;
   std::cout<<"warm_run="<<j<<" wall_s="<<warm_wall[j]<<" gpu_timeline_ms="<<warm_gpu[j]<<" events="<<events.size()<<std::endl;
  }

  auto work=dvs::profile_sensor_work(device->view(),{dvs::to_optical_state(initial),dvs::to_optical_state(trajectory.evaluate(t0+config.scene.time_resolution)),t0,t0+double(config.scene.time_resolution)},config.physic);
  auto resource=[](const char*name,const dvs::KernelResources &r){std::cout<<name<<" registers="<<r.registers<<" local_bytes="<<r.local_bytes<<" shared_bytes="<<r.shared_bytes<<" active_blocks_per_sm="<<r.active_blocks_per_sm<<std::endl;};
  resource("fused",work.fused);resource("optical",work.optical);resource("circuit",work.circuit);
  std::sort(warm_wall,warm_wall+3);std::sort(warm_gpu,warm_gpu+3);
  std::cout<<"pixels="<<work.pixels<<" circuit_cells="<<work.circuit_cells<<" sms="<<work.multiprocessors<<" max_threads_per_sm="<<work.max_threads_per_sm
    <<" optical_only_gpu_ms="<<work.optical_gpu_ms<<" circuit_only_gpu_ms="<<work.circuit_gpu_ms<<" fused_median_wall_s="<<warm_wall[1]
    <<" fused_median_gpu_timeline_ms="<<warm_gpu[1]<<std::endl;
 }
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
