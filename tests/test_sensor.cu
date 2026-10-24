#include "dvs/event_sensor.hpp"
#include "dvs/hapke.hpp"
#include "dvs/raycast.hpp"
#include "dvs/photon_noise.cuh"
#include <iostream>
#include <random>
#include <stdexcept>
using namespace dvs;
void check(bool yes,const char *message) {if(!yes)throw std::runtime_error(message);}
struct TestRng {__host__ __device__ double normal(){return 0;}};
PixelState initial(double rate,const PixelParameters &p) {
  PixelState s;s.receptor_log=s.reference_log=log(rate);s.threshold_pos=p.threshold_pos;s.threshold_neg=p.threshold_neg;return s;
}
template<class T> T *upload(const std::vector<T> &data) {
  T *out;CUDA_CHECK(cudaMalloc(&out,sizeof(T)*data.size()),"test allocation");
  CUDA_CHECK(cudaMemcpy(out,data.data(),sizeof(T)*data.size(),cudaMemcpyHostToDevice),"test upload");return out;
}
__global__ void counts(double lambda,double *out,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  curandState rng;curand_init(42,i,0,&rng);out[i]=poisson_electrons(&rng,lambda);
}
int main() {try {
  std::string config_error;Config device(std::string(DVS_SOURCE_DIR)+"/config/davis346.yaml",config_error);
  check(config_error.empty(),"DAVIS config could not load");
  SceneConfig trajectory_config{};trajectory_config.fps=100;
  trajectory_config.translation={0,0,0};trajectory_config.rotation={0,0,0};
  FrameData<Vec3,Mat3> frame{};frame.frame_index=0;frame.camera_origin=Vec3(0,0,300);
  frame.camera_world=Mat3::Identity();frame.light_dir=Vec3(.8660254f,0,.5f);frame.light_intensity=1000;
  auto last=frame;last.frame_index=2;
  CameraTrajectory trajectory({frame,last},trajectory_config);
  auto initial_pose=trajectory.evaluate(0);
  for(int i=0;i<=1000;++i) {
    auto state=trajectory.evaluate(i*.00002f);
    check(state.position==initial_pose.position && state.light_direction==initial_pose.light_direction &&
          state.light_intensity==initial_pose.light_intensity,"static CPU trajectory numerical jitter");
  }
  TestRng rng;PixelEvent events[64];PixelParameters p;p.charge_scale=10;p.threshold_pos=p.threshold_neg=.2;p.refractory_time=0;
  auto s=initial(1000,p);
  check(advance_pixel(s,p,1000,0,1,events,64,rng)==0,"static circuit emits signal");
  int n=advance_pixel(s,p,2000,0,.1,events,64,rng);check(n==3,"ON step count");
  double target=1000*exp(.2),expected=-p.charge_scale/2000*log((1/target-1/2000.0)/(1/1000.0-1/2000.0));
  check(abs(events[0].time-expected)<1e-12,"nonlinear crossing disagrees with analytical step");
  check(events[2].time-events[1].time>events[1].time-events[0].time,"step crossings falsely linear");
  double bright=events[0].time;s=initial(100,p);advance_pixel(s,p,200,0,1,events,64,rng);
  check(abs(events[0].time-10*bright)<1e-12,"current-dependent latency");
  s=initial(2000,p);check(advance_pixel(s,p,1000,0,.1,events,64,rng)==3&&events[0].polarity==0,"OFF response");
  auto a=initial(1000,p),b=a;advance_pixel(a,p,2000,0,.1,events,64,rng);int fine=0;
  for(int i=0;i<100;++i)fine+=advance_pixel(b,p,2000,i*.001,(i+1)*.001,events,64,rng);
  check(fine==3&&abs(a.receptor_log-b.receptor_log)<1e-12&&abs(a.reference_log-b.reference_log)<1e-12,"constant-current partition invariance");
  s=initial(1000,p);check(advance_pixel(s,p,2000,0,.1,events,2,rng)==-1,"silent circuit overflow");
  p.refractory_time=.002;s=initial(1000,p);n=advance_pixel(s,p,100000,0,.01,events,64,rng);
  for(int i=1;i<n;++i)check(events[i].time-events[i-1].time>=p.refractory_time-1e-12,"reset hold violation");
  p.refractory_time=0;p.reference_leak=.4;s=initial(1000,p);
  check(advance_pixel(s,p,1000,0,1.1,events,64,rng)==2&&abs(events[0].time-.5)<1e-12,"capacitor leakage");
  check(std::isfinite(receptor_log_at(log(1000.0),0,10,10)),"zero photocurrent singularity");
  std::cout<<"PASS nonlinear analytical roots, nonuniform timing, illumination latency, both polarities, partition invariance, reset, leakage, overflow\n";
  // Verify the stochastic INPUT, independently of event count tuning.
  const int samples=65536;double *dc;cudaMalloc(&dc,samples*sizeof(double));std::vector<double> values(samples);
  for(double lambda:{.1,10.,1000.}) {
    counts<<<(samples+127)/128,128>>>(lambda,dc,samples);CUDA_CHECK(cudaMemcpy(values.data(),dc,samples*sizeof(double),cudaMemcpyDeviceToHost),"count samples");
    double mean=0,var=0;for(auto v:values){mean+=v;var+=v*v;}mean/=samples;var=var/samples-mean*mean;
    check(abs(mean-lambda)<6*sqrt(lambda/samples),"electron Poisson mean");
    check(abs(var-lambda)<.07*lambda,"electron Poisson variance");
  }cudaFree(dc);
  std::cout<<"PASS photon/dark charge-count mean and variance at three fluxes\n";
  std::vector<float3> vertices={make_float3(-100,-100,0),make_float3(100,-100,0),make_float3(100,100,0),make_float3(-100,100,0)};
  BVHNode<float3> root;root.start=0;root.count=2;
  OpticalScene scene;scene.nodes=upload(std::vector<BVHNode<float3>>{root});scene.vertices=upload(vertices);
  scene.normals=upload(std::vector<float3>(4,make_float3(0,0,1)));scene.indices=upload(std::vector<uint3>{make_uint3(0,1,2),make_uint3(0,2,3)});
  scene.width=device.camera.width;scene.height=device.camera.height;scene.fx=device.camera.fx;scene.fy=device.camera.fy;scene.cx=device.camera.cx;scene.cy=device.camera.cy;scene.samples=1;
  // Circuit unit fixture uses one spatial sample; full configured aperture is checked in test_render.
  OpticalState pose{};pose.position=make_float3(0,0,10);pose.rotation=make_float4(0,0,0,1);pose.light=make_float3(0,0,1);pose.intensity=1000;
  PhysicConfig c=device.physic;c.threshold_std=0;c.photon_shot_noise=0;c.max_events_per_step=64;c.readout_bandwidth=0;c.readout_latency=0;c.random_seed=42;
  std::vector<Event> ready;EventSensor sensor(scene.width,scene.height,c);sensor.initialize(scene,pose,0);
  sensor.process(scene,OpticalInterval{pose,pose,0,.002},ready);check(ready.empty(),"static GPU startup burst");
  // Endpoints have identical radiance; Hermite light has an interior pulse.
  OpticalState start=pose,end=pose;start.intensity_rate=8e6;end.intensity_rate=-8e6;
  sensor.process(scene,OpticalInterval{start,end,.002,.004},ready);
  int on=0,off=0;for(auto e:ready){on+=e.polarity;off+=!e.polarity;check(e.depth>=10&&e.timestamp>.002&&e.timestamp<=.004,"event-time ray depth or time");}
  check(on>0&&off>0,"missed sub-horizon pulse with identical endpoint images");
  for(int shot:{0,1}) {
    auto batch_config=c;batch_config.photon_shot_noise=shot;
    EventSensor grouped(scene.width,scene.height,batch_config),separate(scene.width,scene.height,batch_config);
    grouped.initialize(scene,pose,0);separate.initialize(scene,pose,0);
    auto bright=pose;bright.intensity=2000;
    std::vector<OpticalInterval> intervals={{pose,pose,0,.001},{bright,bright,.001,.002},{pose,pose,.002,.003}};
    std::vector<Event> individual,one,all;
    for(auto interval:intervals) {separate.process(scene,interval,one);individual.insert(individual.end(),one.begin(),one.end());}
    grouped.process_many(scene,intervals,all);
    check(all.size()==individual.size(),"batch changed comparator count");
    for(size_t j=0;j<all.size();++j)check(all[j].timestamp==individual[j].timestamp && all[j].x==individual[j].x && all[j].y==individual[j].y && all[j].polarity==individual[j].polarity,"batch changed pixel trajectory");
    std::cout<<"PASS exact multi-horizon batching preserves events and timestamps, shot_noise="<<shot<<"\n";
  }
  // Static full native camera grid: optical reuse retains the exact per-cell
  // random draw order and charge dynamics, including shot noise.
  OpticalSampler sampler(scene.width,scene.height,scene.samples);
  for(int shot:{0,1}){
    auto model=c;model.photon_shot_noise=shot;model.circuit_step=1e-6;
    EventSensor native(scene.width,scene.height,model),adaptive(scene.width,scene.height,model);
    native.initialize(scene,pose,0);adaptive.initialize(scene,pose,0);
    std::vector<OpticalInterval> ivs={{pose,pose,0,.001},{pose,pose,.001,.002}};
    std::vector<Event> exact,cached;native.process_many(scene,ivs,exact);
    adaptive.process_adaptive(scene,ivs,sampler,.001,cached);
    check(sampler.statistics().fallback_pixels==0 && sampler.statistics().maximum_knots==5,"static pixel refinement did not reuse optics");
    check(exact.size()==cached.size(),"static optical reuse changed event count");
    for(size_t j=0;j<exact.size();++j)check(exact[j].timestamp==cached[j].timestamp && exact[j].x==cached[j].x &&
      exact[j].y==cached[j].y && exact[j].polarity==cached[j].polarity,"static optical reuse changed seeded events");
    std::cout<<"PASS adaptive static optics preserves exact seeded circuit events, shot_noise="<<shot<<"\n";
  }
  {
    auto model=c;model.photon_shot_noise=0;model.circuit_step=1e-6;
    EventSensor native(scene.width,scene.height,model),adaptive(scene.width,scene.height,model);
    native.initialize(scene,pose,0);adaptive.initialize(scene,pose,0);
    std::vector<OpticalInterval> ivs={{start,end,0,.002}};
    std::vector<Event> exact,cached;native.process_many(scene,ivs,exact);adaptive.process_adaptive(scene,ivs,sampler,.0001,cached);
    check(sampler.statistics().fallback_pixels<scene.width*scene.height,"interior-pulse convergence test only exercised full fallback");
    check(!cached.empty(),"adaptive optics missed interior pulse with equal endpoints");
    auto pixel_order=[](const Event&a,const Event&b){if(a.y!=b.y)return a.y<b.y;if(a.x!=b.x)return a.x<b.x;return a.timestamp<b.timestamp;};
    std::sort(exact.begin(),exact.end(),pixel_order);std::sort(cached.begin(),cached.end(),pixel_order);
    double maximum_error=0;int different_pixels=0;size_t first=0,second=0;
    for(int pixel=0;pixel<scene.width*scene.height;++pixel){size_t a=first,b=second;
      while(first<exact.size() && exact[first].y*scene.width+exact[first].x==pixel)++first;
      while(second<cached.size() && cached[second].y*scene.width+cached[second].x==pixel)++second;
      if(first-a!=second-b){++different_pixels;continue;}
      for(size_t j=0;j<first-a;++j){check(exact[a+j].polarity==cached[b+j].polarity,"adaptive pulse polarity mismatch");
        maximum_error=std::max(maximum_error,std::abs(exact[a+j].timestamp-cached[b+j].timestamp));}
    }
    std::cout<<"Adaptive pulse pixel count differences="<<different_pixels<<" max matched time error="<<maximum_error<<" s\n";
    check(different_pixels<scene.width*scene.height*.001,"adaptive pulse count did not converge at 99.9% of native pixels");
    check(maximum_error<3*c.circuit_step,"adaptive pulse crossings exceeded three native circuit cells");
    std::cout<<"PASS adaptive interior optical pulse: "<<cached.size()<<" events, max timestamp error="<<maximum_error<<" s\n";
    EventSensor discontinuous_native(scene.width,scene.height,model),discontinuous_adaptive(scene.width,scene.height,model);
    discontinuous_native.initialize(scene,pose,0);discontinuous_adaptive.initialize(scene,pose,0);
    auto bright=pose;bright.intensity=2000;ivs={{pose,pose,0,.001},{bright,bright,.001,.002}};
    discontinuous_native.process_many(scene,ivs,exact);discontinuous_adaptive.process_adaptive(scene,ivs,sampler,.0001,cached);
    check(sampler.statistics().native_cell_queries>0 && sampler.statistics().fallback_pixels==0,"unresolved discontinuity did not use local exact rays");
    check(exact.size()==cached.size(),"fallback changed event count");
    std::sort(exact.begin(),exact.end(),pixel_order);std::sort(cached.begin(),cached.end(),pixel_order);
    double fallback_time_error=0;
    for(size_t j=0;j<exact.size();++j){check(exact[j].x==cached[j].x && exact[j].y==cached[j].y && exact[j].polarity==cached[j].polarity,"exact fallback changed per-pixel events");
      fallback_time_error=std::max(fallback_time_error,std::abs(exact[j].timestamp-cached[j].timestamp));}
    check(fallback_time_error<2e-12,"parallel fallback crossing precision changed");
    std::cout<<"Parallel fallback max timestamp rounding difference="<<fallback_time_error<<" s\n";
    std::cout<<"PASS unresolved per-pixel discontinuity uses exact native rays and events\n";
  }
  {
    auto model=c;model.photon_shot_noise=1;model.circuit_step=1e-6;
    EventSensor checked(scene.width,scene.height,model),fresh(scene.width,scene.height,model);
    checked.initialize(scene,pose,0);fresh.initialize(scene,pose,0);
    // This short pulse lies strictly between the five initial optical probes.
    // Direct mean-current circuit checks must catch it before state changes.
    auto flash=pose;flash.intensity=20000;
    std::vector<OpticalInterval> ivs={{pose,pose,0,.0003},{flash,flash,.0003,.0004},{pose,pose,.0004,.002}};std::vector<Event> exact,retried;
    check(!checked.process_adaptive(scene,ivs,sampler,.5,retried,true),"poor optical interpolation was not rejected by voltage convergence gate");
    checked.process_many(scene,ivs,retried);fresh.process_many(scene,ivs,exact);
    check(exact.size()==retried.size(),"rejected optical window changed pixel/RNG state");
    for(size_t j=0;j<exact.size();++j)check(exact[j].timestamp==retried[j].timestamp && exact[j].x==retried[j].x && exact[j].y==retried[j].y && exact[j].polarity==retried[j].polarity,"rejected optical window advanced state or random generator");
    std::cout<<"PASS voltage convergence rejects before pixel/RNG mutation; native retry preserves seeded events\n";
  }
  auto pulse=ready;
  c.circuit_step=.5e-6f;EventSensor refined(scene.width,scene.height,c);refined.initialize(scene,pose,0);
  refined.process(scene,OpticalInterval{pose,pose,0,.002},ready);refined.process(scene,OpticalInterval{start,end,.002,.004},ready);
  check(ready.size()==pulse.size(),"optical timestep pulse count not converged");
  double error=0;for(size_t i=0;i<ready.size();++i)error=fmax(error,abs(ready[i].timestamp-pulse[i].timestamp));
  check(error<2e-6,"optical timestep timestamps not converged");
  std::cout<<"PASS CUDA scene/circuit pulse: "<<on<<" ON, "<<off<<" OFF with identical endpoints; refinement error "<<error<<" s\n";
#ifdef ENABLE_PHYSIC_EFFECTS
  c.photon_shot_noise=1;c.threshold_pos=c.threshold_neg=.03;c.refractory_time=0;c.circuit_step=1e-6;
  EventSensor noisy(scene.width,scene.height,c),repeat(scene.width,scene.height,c);noisy.initialize(scene,pose,0);repeat.initialize(scene,pose,0);
  noisy.process(scene,OpticalInterval{pose,pose,0,.001},ready);auto first=ready;
  repeat.process(scene,OpticalInterval{pose,pose,0,.001},ready);check(!ready.empty()&&ready.size()==first.size(),"photon-driven circuit noise/repeatability");
  for(size_t i=0;i<ready.size();++i)check(ready[i].timestamp==first[i].timestamp&&ready[i].polarity==first[i].polarity&&ready[i].x==first[i].x,"seed reproducibility");
  std::cout<<"PASS photon-current-driven comparator events, repeatable seed: "<<ready.size()<<"\n";
#endif
  cudaFree(const_cast<BVHNode<float3>*>(scene.nodes));cudaFree(const_cast<float3*>(scene.vertices));cudaFree(const_cast<float3*>(scene.normals));cudaFree(const_cast<uint3*>(scene.indices));
  EventReadout bus(100,.001,2);bus.submit({{0,0,0,1,1},{0,1,0,1,1},{0,2,0,1,1}},.005,ready);
  check(ready.empty()&&bus.dropped()==1,"FIFO overflow");bus.submit({},.012,ready);check(ready.size()==1&&abs(ready[0].timestamp-.011)<1e-12,"FIFO latency");bus.drain(ready);check(ready.size()==1,"FIFO drain");
  HapkeConfig hp;auto normal=make_float3(0,0,1);float opposition=hapke_directional_reflectance(normal,normal,normal,hp.w,hp),angle=hapke_directional_reflectance(normal,make_float3(.8660254f,0,.5f),normal,hp.w,hp);
  check(std::isfinite(opposition)&&opposition>angle&&angle>0,"Hapke opposition");hp.phi=0;hp.thetap=0;hp.hC=0;hp.B_C0=0;
  check(std::isfinite(hapke_directional_reflectance(normal,normal,normal,hp.w,hp)),"Hapke limits");
  std::cout<<"PASS readout serialization and Hapke limiting behavior\n";return 0;
} catch(const std::exception &e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;} }
