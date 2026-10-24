#include "dvs/event_sensor.hpp"
#include "dvs/device_optics.cuh"
#include "dvs/photon_noise.cuh"
#include "dvs/math_types.hpp"
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cstdlib>
namespace dvs {
namespace {
constexpr double q=1.602176634e-19,k=1.380649e-23,h=6.62607015e-34,c=299792458;
struct DeviceRng {
  curandState *state;
  __device__ double normal() {return curand_normal_double(state);}
};
__global__ void initialize_pixels(int n,OpticalScene scene,OpticalState pose,PixelParameters p,
   PhotoParameters photo,double stddev,double minimum,unsigned seed,PixelState *states,curandState *random,int *error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  curand_init(seed,i,0,random+i);
  auto optical=sample_pixel(scene,i,pose);
  if(!optical.valid || !isfinite(optical.radiance)) {atomicExch(error,2);return;}
  PixelState s;
  s.receptor_log=s.reference_log=log(fmax(1e-30,optical.radiance*photo.electrons_per_radiance+photo.dark_electrons));
  s.threshold_pos=fmax(minimum,p.threshold_pos+stddev*curand_normal_double(random+i));
  s.threshold_neg=fmax(minimum,p.threshold_neg+stddev*curand_normal_double(random+i));states[i]=s;
}
__global__ void generate_pixels(int n,int width,OpticalScene scene,const OpticalInterval *intervals,int interval_count,PixelParameters p,
  PhotoParameters photo,int max_events,PixelState *states,curandState *random,Event *events,int *event_count,int *error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  PixelEvent local[64];int count=0;DeviceRng rng{random+i};
  // Integrate each pixel's analog state using fresh optical queries inside the
  // horizon. No frame buffers or frame-end brightness interpolation exist.
  for(int interval=0;interval<interval_count;++interval) {
  OpticalInterval iv=intervals[interval];
  int steps=static_cast<int>(ceil((iv.t1-iv.t0)/photo.integration_step));
  double dt=(iv.t1-iv.t0)/steps;
  for(int step=0;step<steps;++step) {
    double t0=iv.t0+step*dt,t1=step+1==steps?iv.t1:iv.t0+(step+1)*dt;
    auto optical=sample_pixel(scene,i,optical_state_at(iv,(t0+t1)*.5));
    if(!optical.valid || !isfinite(optical.radiance)) {atomicExch(error,2);return;}
    double rate=optical.radiance*photo.electrons_per_radiance+photo.dark_electrons;
    // Charge-count tau leap: independent photon/dark electron counts feed
    // photocurrent, then the nonlinear capacitor dynamics and comparators.
    if(photo.shot_noise) rate=poisson_electrons(random+i,rate*(t1-t0))/(t1-t0);
    int emitted=advance_pixel(states[i],p,rate,t0,t1,local+count,max_events-count,rng);
    if(emitted<0) {atomicExch(error,1);return;} count+=emitted;
  }
  }
  int offset=atomicAdd(event_count,count);
  int interval=0;
  for(int j=0;j<count;++j) {
    while(interval+1<interval_count && local[j].time>intervals[interval].t1) ++interval;
    auto optical=sample_pixel(scene,i,optical_state_at(intervals[interval],local[j].time));
    if(!optical.valid) {atomicExch(error,2);return;}
    events[offset+j]=Event{local[j].time,static_cast<int16_t>(i%width),static_cast<int16_t>(i/width),
                           static_cast<int8_t>(local[j].polarity),optical.depth};
  }
}
__global__ void profile_optical_cells(int n,OpticalScene scene,OpticalInterval iv,PhotoParameters photo,double *rates,int *error) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int steps=static_cast<int>(ceil((iv.t1-iv.t0)/photo.integration_step));
  double dt=(iv.t1-iv.t0)/steps;
  for(int step=0;step<steps;++step){
    double t0=iv.t0+step*dt,t1=step+1==steps?iv.t1:iv.t0+(step+1)*dt;
    auto optical=sample_pixel(scene,i,optical_state_at(iv,(t0+t1)*.5));
    if(!optical.valid || !isfinite(optical.radiance)){atomicExch(error,2);return;}
    rates[static_cast<size_t>(step)*n+i]=optical.radiance*photo.electrons_per_radiance+photo.dark_electrons;
  }
}
__global__ void profile_circuit_cells(int n,OpticalInterval iv,PhotoParameters photo,PixelParameters p,
    const double *rates,PixelState *states,curandState *random,int *counts) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int steps=static_cast<int>(ceil((iv.t1-iv.t0)/photo.integration_step));
  double dt=(iv.t1-iv.t0)/steps;PixelEvent local[64];int count=0;DeviceRng rng{random+i};
  for(int step=0;step<steps;++step){
    double t0=iv.t0+step*dt,t1=step+1==steps?iv.t1:iv.t0+(step+1)*dt;
    double rate=rates[static_cast<size_t>(step)*n+i];
    if(photo.shot_noise)rate=poisson_electrons(random+i,rate*(t1-t0))/(t1-t0);
    int emitted=advance_pixel(states[i],p,rate,t0,t1,local+count,64-count,rng);
    if(emitted<0){counts[i]=-1;return;}count+=emitted;
  }
  counts[i]=count;
}
// Separate from the ray kernel so the common smooth pixels carry no BVH
// stack or ray registers. Every configured native interval/circuit cell is kept.
__global__ void integrate_optical_knots(int n,int width,OpticalKnots knots,ExactOpticalCells exact,const OpticalInterval *ivs,int iv_count,
    double start,double end,PixelParameters p,PhotoParameters photo,int capacity,PixelState *states,
    curandState *random,Event *events,int *total,int *error){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n || knots.counts[i]<0)return;
  auto values=knots.values+size_t(i)*knots.stride;int size=knots.counts[i],a=0,exact_index=0,cell_index=0;
  PixelEvent local[64];int count=0;DeviceRng rng{random+i};
  for(int interval=0;interval<iv_count;++interval){auto iv=ivs[interval];
    int steps=int(ceil((iv.t1-iv.t0)/photo.integration_step));double dt=(iv.t1-iv.t0)/steps;
    for(int step=0;step<steps;++step,++cell_index){
      double t0=iv.t0+step*dt,t1=step+1==steps?iv.t1:iv.t0+(step+1)*dt;
      double fraction=((t0+t1)*.5-start)/(end-start);
      while(a+2<size && fraction>fabs(values[a+1].fraction))++a;
      double weight=(fraction-fabs(values[a].fraction))/(fabs(values[a+1].fraction)-fabs(values[a].fraction));
      double light=values[a].radiance+(values[a+1].radiance-values[a].radiance)*weight;
      if(exact.counts && exact_index<exact.counts[i] && exact.cells[size_t(i)*exact.stride+exact_index].index==cell_index)
        light=exact.radiance[size_t(i)*exact.stride+exact_index++];
      double rate=fmax(0.0,light)*photo.electrons_per_radiance+photo.dark_electrons;
      if(photo.shot_noise)rate=poisson_electrons(random+i,rate*(t1-t0))/(t1-t0);
      int emitted=advance_pixel(states[i],p,rate,t0,t1,local+count,capacity-count,rng);
      if(emitted<0){atomicExch(error,1);return;}count+=emitted;
    }
  }
  if(!count)return;int offset=atomicAdd(total,count);
  // Dataset HDF5 events carry x,y,t,p. Auxiliary per-event ray depth is not
  // supplied by this interpolating path; full direct-ray frame depths remain.
  for(int j=0;j<count;++j)events[offset+j]={local[j].time,int16_t(i%width),int16_t(i/width),int8_t(local[j].polarity),-1};
}

__global__ void compact_fallback(int n,const int *counts,int *ids,int *total){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && counts[i]<0)ids[atomicAdd(total,1)]=i;
}
__global__ void integrate_fallback_tile(int n,int width,const int *ids,const OpticalCell *cells,int cell_count,
    const double *light,PixelParameters p,PhotoParameters photo,int capacity,PixelState *states,
    curandState *random,Event *events,int *total,int *error){
  int work=blockIdx.x*blockDim.x+threadIdx.x;if(work>=n)return;int i=ids[work];
  PixelEvent local[64];int count=0;DeviceRng rng{random+i};
  for(int cell=0;cell<cell_count;++cell){auto time=cells[cell];
    double rate=light[size_t(cell)*n+work]*photo.electrons_per_radiance+photo.dark_electrons;
    if(photo.shot_noise)rate=poisson_electrons(random+i,rate*(time.end-time.start))/(time.end-time.start);
    int emitted=advance_pixel(states[i],p,rate,time.start,time.end,local+count,capacity-count,rng);
    if(emitted<0){atomicExch(error,1);return;}count+=emitted;
  }
  if(!count)return;int offset=atomicAdd(total,count);
  for(int j=0;j<count;++j)events[offset+j]={local[j].time,int16_t(i%width),int16_t(i/width),int8_t(local[j].polarity),-1};
}

__global__ void capture_mean_probes(int n,int pixels,const PixelState *states,ReceptorProbeState *probes){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int pixel=int((uint64_t(i)*2654435761ULL)%pixels);probes[i]={states[pixel].receptor_log,states[pixel].receptor_log};
}
__global__ void advance_mean_probes(int n,const int *ids,const OpticalCell *cells,int cell_count,const double *light,
    OpticalKnots knots,double start,double end,PhotoParameters photo,double charge,ReceptorProbeState *states){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int pixel=ids[i];auto values=knots.values+size_t(pixel)*knots.stride;
  for(int cell=0;cell<cell_count;++cell){auto time=cells[cell];double exact=light[size_t(cell)*n+i],cached=exact;
    if(knots.counts[pixel]>=0){int a=0;double fraction=((time.start+time.end)*.5-start)/(end-start);
      while(a+2<knots.counts[pixel] && fraction>fabs(values[a+1].fraction))++a;
      double weight=(fraction-fabs(values[a].fraction))/(fabs(values[a+1].fraction)-fabs(values[a].fraction));
      if(values[a+1].fraction>=0)cached=values[a].radiance+(values[a+1].radiance-values[a].radiance)*weight;
    }
    states[i].exact=receptor_log_at(states[i].exact,exact*photo.electrons_per_radiance+photo.dark_electrons,time.end-time.start,charge);
    states[i].cached=receptor_log_at(states[i].cached,fmax(0.0,cached)*photo.electrons_per_radiance+photo.dark_electrons,time.end-time.start,charge);
  }
}
PixelParameters parameters(const PhysicConfig &c) {
  PixelParameters p; p.threshold_pos=c.threshold_pos;p.threshold_neg=c.threshold_neg;
  double voltage=k*c.temperature_k/(q*c.subthreshold_kappa);
  p.charge_scale=c.receptor_capacitance_f*voltage/q;
#ifdef ENABLE_PHYSIC_EFFECTS
  p.refractory_time=c.refractory_time;
  p.reference_leak=c.reset_leakage_a/(c.reset_capacitance_f*voltage);
  p.reset_noise_std=c.reset_thermal_noise?sqrt(k*c.temperature_k/c.reset_capacitance_f)/voltage:0;
#else
  p.refractory_time=0;
#endif
  return p;
}
PhotoParameters photo_parameters(const PhysicConfig &s) {
  PhotoParameters p;
  double pitch=s.pixel_pitch_um*1e-6;
  p.electrons_per_radiance=s.radiance_scale*3.141592653589793/(4*s.f_number*s.f_number)*s.lens_transmission*
                           pitch*pitch*s.fill_factor*s.quantum_efficiency*(s.wavelength_nm*1e-9)/(h*c);
  p.integration_step=s.circuit_step;
#ifdef ENABLE_PHYSIC_EFFECTS
  p.dark_electrons=s.dark_current_a/q;p.shot_noise=s.photon_shot_noise;
#else
  p.dark_electrons=0;p.shot_noise=false;
#endif
  return p;
}
double bandwidth(const PhysicConfig &c) {
#ifdef ENABLE_AER_DELAY
  return c.readout_bandwidth;
#else
  return 0;
#endif
}
double latency(const PhysicConfig &c) {
#ifdef ENABLE_AER_DELAY
  return c.readout_latency;
#else
  return 0;
#endif
}
template<class Kernel> KernelResources resources(Kernel kernel) {
  cudaFuncAttributes info{};int blocks=0;
  CUDA_CHECK(cudaFuncGetAttributes(&info,kernel),"kernel resource attributes");
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,128,0),"static kernel occupancy");
  return {info.numRegs,int(info.localSizeBytes),int(info.sharedSizeBytes),blocks};
}
template<class F> double timed_kernel(F launch) {
  cudaEvent_t start,finish;CUDA_CHECK(cudaEventCreate(&start),"profile timer start");
  CUDA_CHECK(cudaEventCreate(&finish),"profile timer finish");
  CUDA_CHECK(cudaEventRecord(start),"record profile start");launch();
  CUDA_CHECK(cudaGetLastError(),"profile launch");CUDA_CHECK(cudaEventRecord(finish),"record profile finish");
  CUDA_CHECK(cudaEventSynchronize(finish),"profile wait");float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms,start,finish),"profile elapsed time");
  cudaEventDestroy(start);cudaEventDestroy(finish);return ms;
}
}
SensorWorkProfile profile_sensor_work(const OpticalScene &scene,const OpticalInterval &iv,const PhysicConfig &config) {
  auto photo=photo_parameters(config);auto p=parameters(config);
  int n=scene.width*scene.height,steps=int(ceil((iv.t1-iv.t0)/photo.integration_step));
  if(steps<1 || steps>1000)throw std::invalid_argument("invalid diagnostic horizon");
  SensorWorkProfile result;result.pixels=n;result.circuit_cells=steps;
  cudaDeviceProp device{};int id;CUDA_CHECK(cudaGetDevice(&id),"profile device");
  CUDA_CHECK(cudaGetDeviceProperties(&device,id),"profile device attributes");
  result.multiprocessors=device.multiProcessorCount;result.max_threads_per_sm=device.maxThreadsPerMultiProcessor;
  result.fused=resources(generate_pixels);result.optical=resources(profile_optical_cells);result.circuit=resources(profile_circuit_cells);
  double *rates=nullptr;PixelState *initial=nullptr,*states=nullptr;
  curandState *initial_rng=nullptr,*rng=nullptr;int *error=nullptr,*counts=nullptr;
  auto release=[&]{cudaFree(rates);cudaFree(initial);cudaFree(states);cudaFree(initial_rng);cudaFree(rng);cudaFree(error);cudaFree(counts);};
  try {
    CUDA_CHECK(cudaMalloc(&rates,sizeof(double)*size_t(n)*steps),"profile currents");
    CUDA_CHECK(cudaMalloc(&initial,sizeof(PixelState)*n),"profile initial states");
    CUDA_CHECK(cudaMalloc(&states,sizeof(PixelState)*n),"profile states");
    CUDA_CHECK(cudaMalloc(&initial_rng,sizeof(curandState)*n),"profile initial RNG");
    CUDA_CHECK(cudaMalloc(&rng,sizeof(curandState)*n),"profile RNG");
    CUDA_CHECK(cudaMalloc(&error,sizeof(int)),"profile error");CUDA_CHECK(cudaMalloc(&counts,sizeof(int)*n),"profile counts");
    CUDA_CHECK(cudaMemset(error,0,sizeof(int)),"profile clear error");
    initialize_pixels<<<(n+127)/128,128>>>(n,scene,iv.start,p,photo,config.threshold_std,config.threshold_min,config.random_seed,initial,initial_rng,error);
    auto optical_launch=[&]{profile_optical_cells<<<(n+127)/128,128>>>(n,scene,iv,photo,rates,error);};
    timed_kernel(optical_launch);double optical_times[3],circuit_times[3];
    for(int j=0;j<3;++j)optical_times[j]=timed_kernel(optical_launch);
    int status;CUDA_CHECK(cudaMemcpy(&status,error,sizeof(int),cudaMemcpyDeviceToHost),"profile optical status");
    if(status)throw std::runtime_error("invalid optical sample during diagnostic");
    for(int j=0;j<3;++j){
      CUDA_CHECK(cudaMemcpy(states,initial,sizeof(PixelState)*n,cudaMemcpyDeviceToDevice),"reset diagnostic states");
      CUDA_CHECK(cudaMemcpy(rng,initial_rng,sizeof(curandState)*n,cudaMemcpyDeviceToDevice),"reset diagnostic RNG");
      circuit_times[j]=timed_kernel([&]{profile_circuit_cells<<<(n+127)/128,128>>>(n,iv,photo,p,rates,states,rng,counts);});
    }
    std::vector<int> host_counts(n);CUDA_CHECK(cudaMemcpy(host_counts.data(),counts,sizeof(int)*n,cudaMemcpyDeviceToHost),"profile circuit status");
    if(std::any_of(host_counts.begin(),host_counts.end(),[](int value){return value<0;}))throw std::runtime_error("diagnostic comparator overflow");
    std::sort(optical_times,optical_times+3);std::sort(circuit_times,circuit_times+3);
    result.optical_gpu_ms=optical_times[1];result.circuit_gpu_ms=circuit_times[1];
  }catch(...){release();throw;}release();return result;
}
EventSensor::EventSensor(int width,int height,const PhysicConfig &config)
 :width_(width),height_(height),count_(width*height),max_events_(config.max_events_per_step),seed_(config.random_seed),
 parameters_(parameters(config)),photo_(photo_parameters(config)),threshold_std_(config.threshold_std),threshold_min_(config.threshold_min),
 readout_(bandwidth(config),latency(config),config.readout_queue_capacity) {
#ifndef ENABLE_PHYSIC_EFFECTS
  threshold_std_=0;
#endif
  if(width<=0 || height<=0 || width>32767 || height>32767 || max_events_<1 || max_events_>64 ||
     static_cast<long long>(count_)*max_events_>std::numeric_limits<int>::max()) throw std::invalid_argument("invalid sensor capacity");
  CUDA_CHECK(cudaMalloc(&states_,sizeof(PixelState)*count_),"pixel states");
  CUDA_CHECK(cudaMalloc(&random_,sizeof(curandState)*count_),"pixel RNG");
  CUDA_CHECK(cudaMalloc(&events_,sizeof(Event)*static_cast<std::size_t>(count_)*max_events_),"events");
  CUDA_CHECK(cudaMalloc(&event_count_,sizeof(int)),"event count");CUDA_CHECK(cudaMalloc(&overflow_,sizeof(int)),"error flag");
  CUDA_CHECK(cudaMalloc(&fallback_ids_,sizeof(int)*count_),"allocate exact fallback pixel IDs");
  CUDA_CHECK(cudaMalloc(&fallback_count_,sizeof(int)),"allocate exact fallback count");
  CUDA_CHECK(cudaMalloc(&fallback_cells_,sizeof(OpticalCell)*64),"allocate exact fallback cell tile");
}
EventSensor::~EventSensor() {cudaFree(probe_states_);cudaFree(fallback_ids_);cudaFree(fallback_count_);cudaFree(fallback_cells_);cudaFree(intervals_);cudaFree(states_);cudaFree(random_);cudaFree(events_);cudaFree(event_count_);cudaFree(overflow_);}
void EventSensor::initialize(const OpticalScene &scene,const OpticalState &pose,double timestamp) {
  if(!std::isfinite(timestamp) || initialized_) throw std::invalid_argument("invalid sensor initialization");
  CUDA_CHECK(cudaMemset(overflow_,0,sizeof(int)),"clear initialization error");
  initialize_pixels<<<(count_+127)/128,128>>>(count_,scene,pose,parameters_,photo_,threshold_std_,threshold_min_,seed_,states_,random_,overflow_);
  CUDA_CHECK(cudaGetLastError(),"initialize circuit");int error=0;
  CUDA_CHECK(cudaMemcpy(&error,overflow_,sizeof(int),cudaMemcpyDeviceToHost),"initialize circuit status");
  if(error) throw std::runtime_error("invalid optical sample during initialization");
  last_time_=timestamp;initialized_=true;
}
void EventSensor::process(const OpticalScene &scene,const OpticalInterval &iv,std::vector<Event> &events) {
  process_many(scene,{iv},events);
}
void EventSensor::upload_intervals(const std::vector<OpticalInterval>& intervals) {
  if(!initialized_ || intervals.empty()) throw std::invalid_argument("sensor not initialized or no intervals");
  double previous=last_time_,cells=0;
  for(const auto &iv:intervals) {
    if(!std::isfinite(iv.t0) || !std::isfinite(iv.t1) || iv.t0!=previous || iv.t1<=iv.t0)
      throw std::invalid_argument("sensor intervals must be positive and contiguous");
    cells+=(iv.t1-iv.t0)/photo_.integration_step;previous=iv.t1;
  }
  if(cells>1000000) throw std::invalid_argument("optical horizon too long; reduce batch size");
  if(intervals.size()>interval_capacity_) {
    cudaFree(intervals_);intervals_=nullptr;interval_capacity_=0;
    CUDA_CHECK(cudaMalloc(&intervals_,sizeof(OpticalInterval)*intervals.size()),"trajectory interval allocation");
    interval_capacity_=intervals.size();
  }
  CUDA_CHECK(cudaMemcpy(intervals_,intervals.data(),sizeof(OpticalInterval)*intervals.size(),cudaMemcpyHostToDevice),"trajectory interval upload");
}
void EventSensor::process_many(const OpticalScene &scene,const std::vector<OpticalInterval> &intervals,std::vector<Event> &events) {
  upload_intervals(intervals);
  CUDA_CHECK(cudaMemset(event_count_,0,sizeof(int)),"clear events");CUDA_CHECK(cudaMemset(overflow_,0,sizeof(int)),"clear status");
  generate_pixels<<<(count_+127)/128,128>>>(count_,width_,scene,intervals_,static_cast<int>(intervals.size()),parameters_,photo_,max_events_,states_,random_,events_,event_count_,overflow_);
  CUDA_CHECK(cudaGetLastError(),"trace and integrate circuits");collect_events(intervals.back().t1,events);
}
void EventSensor::collect_events(double end,std::vector<Event>& events){
  int count=0,error=0;
  CUDA_CHECK(cudaMemcpy(&count,event_count_,sizeof(int),cudaMemcpyDeviceToHost),"copy count");
  CUDA_CHECK(cudaMemcpy(&error,overflow_,sizeof(int),cudaMemcpyDeviceToHost),"copy status");
  if(error==1)throw std::runtime_error("pixel event capacity exceeded; reduce numerical window");
  if(error)throw std::runtime_error("invalid optical sample or BVH traversal overflow");
  std::vector<Event> generated(count);
  if(count)CUDA_CHECK(cudaMemcpy(generated.data(),events_,sizeof(Event)*count,cudaMemcpyDeviceToHost),"copy events");
  readout_.submit(std::move(generated),end,events);last_time_=end;
}
bool EventSensor::process_adaptive(const OpticalScene &scene,const std::vector<OpticalInterval>& intervals,
    OpticalSampler &sampler,double tolerance,std::vector<Event>& events,bool check_convergence){
  if(intervals.empty())throw std::invalid_argument("empty adaptive optical horizon");
  double start=intervals.front().t0,end=intervals.back().t1;
  if(!std::isfinite(tolerance) || tolerance<=0 || parameters_.refractory_time<=0 ||
     (end-start)/parameters_.refractory_time+2>max_events_)
    throw std::invalid_argument("invalid adaptive tolerance or event capacity bound");
  upload_intervals(intervals);
  // All ray refinement completes and is checked before pixel/RNG state changes.
  auto wall_start=std::chrono::steady_clock::now();
  auto knots=sampler.refine(scene,intervals_,intervals.size(),start,end,photo_,tolerance,states_,parameters_.charge_scale);
  auto exact=sampler.prepare_exact_cells(scene,intervals_,intervals.size(),start,end,photo_);
  if(diagnostics_ || check_convergence){
    int n=std::min(1024,count_);if(!probe_states_)CUDA_CHECK(cudaMalloc(&probe_states_,sizeof(ReceptorProbeState)*n),"allocate mean receptor probes");
    capture_mean_probes<<<(n+127)/128,128>>>(n,count_,states_,probe_states_);diagnostic_knots_=knots;
    CUDA_CHECK(cudaGetLastError(),"capture mean receptor initial state");
  }
  if(check_convergence){
    convergence_=verify_receptor(scene,sampler,intervals);double threshold=std::min(parameters_.threshold_pos,parameters_.threshold_neg);
    if(convergence_.maximum_log_voltage_error>.02*threshold || convergence_.p99_log_voltage_error>.005*threshold){events.clear();return false;}
  }
  if(std::getenv("DVS_PROFILE_ADAPTIVE"))std::cout<<"refine_wall_s="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()
    <<" fallback_pixels="<<sampler.statistics().fallback_pixels<<" local_exact_queries="<<sampler.statistics().native_cell_queries<<" optical_pixel_samples="<<sampler.statistics().traced_pixel_samples<<std::endl;
  CUDA_CHECK(cudaMemset(event_count_,0,sizeof(int)),"clear adaptive event count");
  CUDA_CHECK(cudaMemset(overflow_,0,sizeof(int)),"clear adaptive circuit status");
  integrate_optical_knots<<<(count_+127)/128,128>>>(count_,width_,knots,exact,intervals_,intervals.size(),start,end,
    parameters_,photo_,max_events_,states_,random_,events_,event_count_,overflow_);
  CUDA_CHECK(cudaGetLastError(),"adaptive circuit launch");
  int fallback=sampler.statistics().fallback_pixels;
  if(fallback){
    CUDA_CHECK(cudaMemset(fallback_count_,0,sizeof(int)),"clear fallback ID counter");
    compact_fallback<<<(count_+127)/128,128>>>(count_,knots.counts,fallback_ids_,fallback_count_);
    CUDA_CHECK(cudaGetLastError(),"compact exact fallback pixels");
    std::vector<OpticalCell> cells;cells.reserve(64);
    auto tile=[&]{
      const double *light=sampler.sample_cells(scene,intervals_,fallback_ids_,fallback,cells);
      CUDA_CHECK(cudaMemcpy(fallback_cells_,cells.data(),sizeof(OpticalCell)*cells.size(),cudaMemcpyHostToDevice),"upload exact circuit cell times");
      integrate_fallback_tile<<<(fallback+127)/128,128>>>(fallback,width_,fallback_ids_,fallback_cells_,cells.size(),light,
        parameters_,photo_,max_events_,states_,random_,events_,event_count_,overflow_);
      CUDA_CHECK(cudaGetLastError(),"exact fallback circuit tile launch");cells.clear();
    };
    for(size_t index=0;index<intervals.size();++index){auto iv=intervals[index];
      int steps=int(ceil((iv.t1-iv.t0)/photo_.integration_step));double dt=(iv.t1-iv.t0)/steps;
      for(int step=0;step<steps;++step){cells.push_back({iv.t0+step*dt,step+1==steps?iv.t1:iv.t0+(step+1)*dt,int(index)});
        if(cells.size()==64)tile();}
    }
    if(!cells.empty())tile();
  }
  collect_events(end,events);return true;
}

ReceptorProbeResult EventSensor::verify_receptor(const OpticalScene &scene,OpticalSampler &sampler,const std::vector<OpticalInterval>& intervals){
  if(!probe_states_)throw std::logic_error("enable numerical diagnostics before processing");
  int n=std::min(1024,count_);std::vector<int> ids(n);
  for(int i=0;i<n;++i)ids[i]=int((uint64_t(i)*2654435761ULL)%count_);
  CUDA_CHECK(cudaMemcpy(fallback_ids_,ids.data(),sizeof(int)*n,cudaMemcpyHostToDevice),"upload mean receptor probe IDs");
  std::vector<OpticalCell> cells;cells.reserve(64);ReceptorProbeResult result;result.pixels=n;
  auto tile=[&]{const double *light=sampler.sample_cells(scene,intervals_,fallback_ids_,n,cells);
    CUDA_CHECK(cudaMemcpy(fallback_cells_,cells.data(),sizeof(OpticalCell)*cells.size(),cudaMemcpyHostToDevice),"upload mean receptor probe cells");
    advance_mean_probes<<<(n+127)/128,128>>>(n,fallback_ids_,fallback_cells_,cells.size(),light,diagnostic_knots_,
      intervals.front().t0,intervals.back().t1,photo_,parameters_.charge_scale,probe_states_);
    CUDA_CHECK(cudaGetLastError(),"mean receptor probe integration");result.circuit_cells+=cells.size();cells.clear();};
  for(size_t index=0;index<intervals.size();++index){auto iv=intervals[index];int steps=int(ceil((iv.t1-iv.t0)/photo_.integration_step));double dt=(iv.t1-iv.t0)/steps;
    for(int step=0;step<steps;++step){cells.push_back({iv.t0+step*dt,step+1==steps?iv.t1:iv.t0+(step+1)*dt,int(index)});if(cells.size()==64)tile();}}
  if(!cells.empty())tile();std::vector<ReceptorProbeState> states(n);
  CUDA_CHECK(cudaMemcpy(states.data(),probe_states_,sizeof(ReceptorProbeState)*n,cudaMemcpyDeviceToHost),"download mean receptor convergence");
  std::vector<double> errors(n);for(int i=0;i<n;++i){errors[i]=fabs(states[i].exact-states[i].cached);result.mean_log_voltage_error+=errors[i];}
  std::sort(errors.begin(),errors.end());result.maximum_log_voltage_error=errors.back();result.p99_log_voltage_error=errors[size_t(n*.99)];result.mean_log_voltage_error/=n;return result;
}
} // namespace dvs
