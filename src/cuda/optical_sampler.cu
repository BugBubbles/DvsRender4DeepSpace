#include "dvs/optical_sampler.hpp"
#include "dvs/device_optics.cuh"
#include "dvs/math_types.hpp"
#include <algorithm>
namespace dvs {
namespace {
// Four warps per block. This storage is reused at each quadrature level.
__global__ void sample_warps(OpticalScene scene,const OpticalState *poses,int pose_count,int maximum,OpticalValue *out,int *error) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int work=blockIdx.x*4+warp,n=scene.width*scene.height;if(work>=n*pose_count)return;
  int pixel=work%n,pose_index=work/n;
  __shared__ double radiance[4][256];__shared__ float depth[4][256];
  for(int j=lane;j<maximum*maximum;j+=32){
    auto sample=sample_subray(scene,pixel,poses[pose_index],j%maximum,j/maximum,maximum);
    radiance[warp][j]=sample.radiance;depth[warp][j]=sample.depth;
    if(!sample.valid)atomicExch(error,1);
  }
  __syncwarp();
  if(lane==0){double total=0,ranges=0;int hits=0;
    for(int j=0;j<maximum*maximum;++j){total+=radiance[warp][j];if(depth[warp][j]>=0){ranges+=depth[warp][j];++hits;}}
    out[work]={total/(maximum*maximum),hits?float(ranges/hits):-1,maximum};
  }
}

struct RefinementEntry {double a,b,left,middle,right;};
__device__ OpticalState native_pose(const OpticalInterval *ivs,int count,double t){
  int lo=0,hi=count-1;
  while(lo<hi){int m=(lo+hi)/2;if(t>ivs[m].t1)lo=m+1;else hi=m;}
  return optical_state_at(ivs[lo],t);
}
__device__ double warp_radiance(OpticalScene scene,int pixel,OpticalState pose,double *scratch,int *error){
  int lane=threadIdx.x%32,ss=scene.samples;
  for(int j=lane;j<ss*ss;j+=32){auto r=sample_subray(scene,pixel,pose,j%ss,j/ss,ss);
    scratch[j]=r.radiance;if(!r.valid || !isfinite(r.radiance))atomicExch(error,1);}
  __syncwarp();double mean=0;
  if(lane==0){for(int j=0;j<ss*ss;++j)mean+=scratch[j];mean/=ss*ss;}
  return __shfl_sync(0xffffffff,mean,0);
}
__global__ void refine_pixels(OpticalScene scene,const OpticalInterval *ivs,int iv_count,double start,double end,
    PhotoParameters photo,double tolerance,const PixelState *sensor_states,double charge,int capacity,OpticalKnot *knots,int *counts,OpticalRefinementStats *stats,int *error){
  int lane=threadIdx.x%32,warp=threadIdx.x/32,pixel=blockIdx.x*4+warp;
  if(pixel>=scene.width*scene.height)return;
  __shared__ double scratch[4][256];__shared__ RefinementEntry stack[4][16];
  double initial[5];unsigned long long queries=0;
  for(int j=0;j<5;++j){initial[j]=warp_radiance(scene,pixel,native_pose(ivs,iv_count,start+(end-start)*j*.25),scratch[warp],error);++queries;}
  if(lane==0){stack[warp][0]={.5,1,initial[2],initial[3],initial[4]};
    stack[warp][1]={0,.5,initial[0],initial[1],initial[2]};
    knots[size_t(pixel)*capacity]={0,initial[0]};}
  __syncwarp();int top=2,written=1;bool fallback=false;
  while(top){auto entry=stack[warp][--top];
    double actual=entry.middle*photo.electrons_per_radiance+photo.dark_electrons;
    double estimate=(entry.left+entry.right)*.5*photo.electrons_per_radiance+photo.dark_electrons;
    double residual=fabs(log(fmax(1e-30,actual)/fmax(1e-30,estimate)));
    double voltage_start=sensor_states[pixel].receptor_log;
    double leaf_duration=(entry.b-entry.a)*(end-start);
    double voltage_error=fabs(receptor_log_at(voltage_start,actual,leaf_duration,charge)-
                              receptor_log_at(voltage_start,estimate,leaf_duration,charge));
    double pixel_budget=kOpticalVoltageThresholdFraction*fmin(sensor_states[pixel].threshold_pos,sensor_states[pixel].threshold_neg);
    if(voltage_error<=pixel_budget && (residual<=tolerance || fmax(actual,estimate)*leaf_duration<1.0)){
      if(written+2>capacity){fallback=true;break;}
      // Keep the direct midpoint as well as the endpoint: it already cost a
      // ray query and substantially reduces integrated photocurrent error.
      if(lane==0){knots[size_t(pixel)*capacity+written]={(entry.a+entry.b)*.5,entry.middle};
        knots[size_t(pixel)*capacity+written+1]={entry.b,entry.right};}
      written+=2;
    }else{
      // Unresolved features at the electrical sampling scale use exact tracing.
      if((entry.b-entry.a)*(end-start)<=photo.integration_step || top+2>16){
        if(written>=capacity){fallback=true;break;}
        if(lane==0){knots[size_t(pixel)*capacity+written]={-entry.b,entry.right};atomicAdd(&stats->exact_segments,1);}
        ++written;__syncwarp();continue;
      }
      double mid=(entry.a+entry.b)*.5;
      double left_mid=warp_radiance(scene,pixel,native_pose(ivs,iv_count,start+(end-start)*(entry.a+mid)*.5),scratch[warp],error);
      double right_mid=warp_radiance(scene,pixel,native_pose(ivs,iv_count,start+(end-start)*(mid+entry.b)*.5),scratch[warp],error);queries+=2;
      if(lane==0){stack[warp][top]={mid,entry.b,entry.middle,right_mid,entry.right};
        stack[warp][top+1]={entry.a,mid,entry.left,left_mid,entry.middle};}
      top+=2;
    }
    __syncwarp();
  }
  if(lane==0){counts[pixel]=fallback?-1:written;atomicAdd(&stats->traced_pixel_samples,queries);
    if(fallback)atomicAdd(&stats->fallback_pixels,1);else atomicMax(&stats->maximum_knots,written);}
}

__global__ void verify_knots(OpticalScene scene,const OpticalInterval *ivs,int iv_count,double start,double end,
    PhotoParameters photo,int stride,const OpticalKnot *knots,const int *counts,int positions,int times,double *errors,int *status){
  int lane=threadIdx.x%32,warp=threadIdx.x/32,work=blockIdx.x*4+warp;if(work>=positions*times)return;
  int n=scene.width*scene.height,pixel=int((uint64_t(work%positions)*2654435761ULL)%n),probe=work/positions;
  if(counts[pixel]<0){if(lane==0)errors[work]=-1;return;}
  double fraction=(probe+.5)/times,t=start+(end-start)*fraction;
  __shared__ double scratch[4][256];
  double light=warp_radiance(scene,pixel,native_pose(ivs,iv_count,t),scratch[warp],status);
  if(lane==0){auto values=knots+size_t(pixel)*stride;int a=0;
    while(a+2<counts[pixel] && fraction>fabs(values[a+1].fraction))++a;
    if(values[a+1].fraction<0){errors[work]=-1;return;}
    double weight=(fraction-fabs(values[a].fraction))/(fabs(values[a+1].fraction)-fabs(values[a].fraction));
    double estimate=values[a].radiance+(values[a+1].radiance-values[a].radiance)*weight;
    double actual=light*photo.electrons_per_radiance+photo.dark_electrons;
    double predicted=estimate*photo.electrons_per_radiance+photo.dark_electrons;
    errors[work]=fabs(log(fmax(1e-30,actual)/fmax(1e-30,predicted)));
  }
}

__global__ void sample_native_cells(OpticalScene scene,const OpticalInterval *ivs,const int *pixels,int pixel_count,
    const OpticalCell *cells,int cell_count,double *radiance,int *error){
  int lane=threadIdx.x%32,warp=threadIdx.x/32,work=blockIdx.x*4+warp;if(work>=pixel_count*cell_count)return;
  auto cell=cells[work/pixel_count];int pixel=pixels[work%pixel_count];
  __shared__ double scratch[4][256];
  double light=warp_radiance(scene,pixel,optical_state_at(ivs[cell.interval],(cell.start+cell.end)*.5),scratch[warp],error);
  if(lane==0)radiance[work]=light;
}

__global__ void identify_exact_cells(int n,const OpticalInterval *ivs,int iv_count,double start,double end,PhotoParameters photo,
    int knot_stride,const OpticalKnot *knots,int *knot_counts,int capacity,ExactOpticalCell *queries,int *counts,int *queue,int *total,
    OpticalRefinementStats *stats){
  int pixel=blockIdx.x*blockDim.x+threadIdx.x;if(pixel>=n)return;counts[pixel]=0;if(knot_counts[pixel]<0)return;
  auto values=knots+size_t(pixel)*knot_stride;int a=0,cell_index=0,used=0;
  for(int interval=0;interval<iv_count;++interval){auto iv=ivs[interval];int steps=int(ceil((iv.t1-iv.t0)/photo.integration_step));double dt=(iv.t1-iv.t0)/steps;
    for(int step=0;step<steps;++step,++cell_index){double left=iv.t0+step*dt,right=step+1==steps?iv.t1:iv.t0+(step+1)*dt;
      double fraction=((left+right)*.5-start)/(end-start);
      while(a+2<knot_counts[pixel] && fraction>fabs(values[a+1].fraction))++a;
      if(values[a+1].fraction<0){
        if(used==capacity){knot_counts[pixel]=-1;atomicAdd(&stats->fallback_pixels,1);return;}
        queries[size_t(pixel)*capacity+used++]={left,right,interval,cell_index};
      }
    }
  }
  counts[pixel]=used;if(!used)return;
  int offset=atomicAdd(total,used);for(int j=0;j<used;++j)queue[offset+j]=pixel*capacity+j;
}
__global__ void sample_exact_queries(OpticalScene scene,const OpticalInterval *ivs,const ExactOpticalCell *cells,
    const int *queue,int count,int stride,double *radiance,int *error){
  int lane=threadIdx.x%32,warp=threadIdx.x/32,work=blockIdx.x*4+warp;if(work>=count)return;
  int slot=queue[work],pixel=slot/stride;auto time=cells[slot];__shared__ double scratch[4][256];
  double value=warp_radiance(scene,pixel,optical_state_at(ivs[time.interval],(time.start+time.end)*.5),scratch[warp],error);
  if(lane==0)radiance[slot]=value;
}

}
OpticalSampler::OpticalSampler(int w,int h,int maximum):count_(w*h),maximum_(maximum){
  if(w<=0 || h<=0 || maximum<1 || maximum>16)throw std::invalid_argument("parallel sampler supports configured grids 1..16");
  size_t available,total;CUDA_CHECK(cudaMemGetInfo(&available,&total),"query optical cache memory budget");
  // Cache sizes change only overflow frequency. A full native trace is the
  // fallback, so this never changes camera geometry or spatial quadrature.
  for(int candidate:{512,256,128}){
    size_t needed=size_t(count_)*candidate*(sizeof(OpticalKnot)+sizeof(ExactOpticalCell)+sizeof(double)+sizeof(int));
    if(available>needed+1024ULL*1024*1024){knot_capacity_=exact_capacity_=candidate;break;}
    // An image-only sampler needs no optical cache. Allocation is checked in refine.
  }
  CUDA_CHECK(cudaMalloc(&error_,sizeof(int)),"allocate optical status");
}
OpticalSampler::~OpticalSampler(){cudaFree(exact_cells_);cudaFree(exact_radiance_);cudaFree(exact_counts_);cudaFree(exact_queue_);cudaFree(exact_total_);cudaFree(cell_radiance_);cudaFree(cells_);cudaFree(knots_);cudaFree(knot_counts_);cudaFree(device_statistics_);cudaFree(poses_);cudaFree(values_);cudaFree(error_);}
const OpticalValue* OpticalSampler::evaluate(const OpticalScene &scene,const std::vector<OpticalState>& states){
  if(states.empty() || scene.width*scene.height!=count_ || scene.samples!=maximum_)throw std::invalid_argument("invalid parallel optical request");
  if(states.size()>capacity_){
    cudaFree(poses_);cudaFree(values_);poses_=nullptr;values_=nullptr;capacity_=0;
    CUDA_CHECK(cudaMalloc(&poses_,sizeof(OpticalState)*states.size()),"allocate optical poses");
    CUDA_CHECK(cudaMalloc(&values_,sizeof(OpticalValue)*std::size_t(count_)*states.size()),"allocate optical values");capacity_=states.size();
  }
  poses_count_=states.size();
  CUDA_CHECK(cudaMemcpy(poses_,states.data(),sizeof(OpticalState)*states.size(),cudaMemcpyHostToDevice),"upload optical poses");
  CUDA_CHECK(cudaMemset(error_,0,sizeof(int)),"clear optical status");
  sample_warps<<<(std::size_t(count_)*states.size()+3)/4,128>>>(scene,poses_,states.size(),maximum_,values_,error_);
  CUDA_CHECK(cudaGetLastError(),"parallel optical launch");int status;
  CUDA_CHECK(cudaMemcpy(&status,error_,sizeof(int),cudaMemcpyDeviceToHost),"parallel optical status");
  if(status)throw std::runtime_error("invalid optical sample/BVH stack overflow");return values_;
}
void OpticalSampler::download(std::vector<OpticalValue>& values)const{
  values.resize(std::size_t(count_)*poses_count_);
  CUDA_CHECK(cudaMemcpy(values.data(),values_,sizeof(OpticalValue)*values.size(),cudaMemcpyDeviceToHost),"download optical values");
}
}

namespace dvs {
OpticalKnots OpticalSampler::refine(const OpticalScene &scene,const OpticalInterval *ivs,int iv_count,
    double start,double end,PhotoParameters photo,double tolerance,const PixelState *states,double charge){
  if(!states || !std::isfinite(charge) || charge<=0 || !ivs || iv_count<1 || scene.width*scene.height!=count_ || scene.samples!=maximum_ ||
     !std::isfinite(tolerance) || tolerance<=0 || !(end>start))throw std::invalid_argument("invalid pixel optical refinement");
  const int capacity=knot_capacity_;
  if(!knots_){
    CUDA_CHECK(cudaMalloc(&knots_,sizeof(OpticalKnot)*size_t(count_)*capacity),"allocate pixel optical knots");
    CUDA_CHECK(cudaMalloc(&knot_counts_,sizeof(int)*count_),"allocate optical knot counts");
    CUDA_CHECK(cudaMalloc(&device_statistics_,sizeof(OpticalRefinementStats)),"allocate refinement statistics");
  }
  CUDA_CHECK(cudaMemset(error_,0,sizeof(int)),"clear refinement errors");
  CUDA_CHECK(cudaMemset(device_statistics_,0,sizeof(OpticalRefinementStats)),"clear refinement statistics");
  refine_pixels<<<(count_+3)/4,128>>>(scene,ivs,iv_count,start,end,photo,tolerance,states,charge,capacity,knots_,knot_counts_,device_statistics_,error_);
  CUDA_CHECK(cudaGetLastError(),"refine pixel optical launch");int status;
  CUDA_CHECK(cudaMemcpy(&status,error_,sizeof(int),cudaMemcpyDeviceToHost),"refinement status");
  if(status)throw std::runtime_error("invalid ray during optical refinement");
  CUDA_CHECK(cudaMemcpy(&statistics_,device_statistics_,sizeof(statistics_),cudaMemcpyDeviceToHost),"refinement statistics");
  return {knots_,knot_counts_,capacity};
}
}

namespace dvs {
OpticalProbeResult OpticalSampler::verify(const OpticalScene &scene,const OpticalInterval *ivs,int iv_count,
    double start,double end,PhotoParameters photo,double tolerance){
  if(!knots_)throw std::logic_error("refine before validating optical knots");
  constexpr int positions=1024,times=17,n=positions*times;double *errors=nullptr;
  CUDA_CHECK(cudaMalloc(&errors,sizeof(double)*n),"allocate optical convergence probes");
  try{
    CUDA_CHECK(cudaMemset(error_,0,sizeof(int)),"clear probe status");
    verify_knots<<<(n+3)/4,128>>>(scene,ivs,iv_count,start,end,photo,knot_capacity_,knots_,knot_counts_,positions,times,errors,error_);
    CUDA_CHECK(cudaGetLastError(),"optical convergence probe launch");std::vector<double> values(n);
    CUDA_CHECK(cudaMemcpy(values.data(),errors,sizeof(double)*n,cudaMemcpyDeviceToHost),"download optical convergence probes");
    int status;CUDA_CHECK(cudaMemcpy(&status,error_,sizeof(int),cudaMemcpyDeviceToHost),"probe status");
    if(status)throw std::runtime_error("invalid ray in optical convergence probe");
    OpticalProbeResult result;result.probes=n;
    for(auto value:values){if(value<0){++result.exact_fallback_probes;continue;}
      result.maximum_log_error=std::max(result.maximum_log_error,value);result.mean_log_error+=value;
      if(value>tolerance)++result.over_tolerance;}
    result.mean_log_error/=n-result.exact_fallback_probes>0?n-result.exact_fallback_probes:1;
    std::sort(values.begin(),values.end());result.p99_log_error=values[size_t(n*.99)];cudaFree(errors);return result;
  }catch(...){cudaFree(errors);throw;}
}
}

namespace dvs {
const double* OpticalSampler::sample_cells(const OpticalScene &scene,const OpticalInterval *ivs,const int *pixels,
    int pixel_count,const std::vector<OpticalCell>& cells){
  if(pixel_count<1 || cells.empty() || cells.size()>64)throw std::invalid_argument("invalid fallback optical cell tile");
  size_t needed=size_t(pixel_count)*64;
  if(needed>cell_capacity_){cudaFree(cell_radiance_);cell_radiance_=nullptr;cell_capacity_=0;
    CUDA_CHECK(cudaMalloc(&cell_radiance_,sizeof(double)*needed),"allocate fallback optical tile");cell_capacity_=needed;}
  if(!cells_)CUDA_CHECK(cudaMalloc(&cells_,sizeof(OpticalCell)*64),"allocate fallback cell times");
  CUDA_CHECK(cudaMemcpy(cells_,cells.data(),sizeof(OpticalCell)*cells.size(),cudaMemcpyHostToDevice),"upload fallback cell times");
  CUDA_CHECK(cudaMemset(error_,0,sizeof(int)),"clear fallback optical status");
  sample_native_cells<<<(size_t(pixel_count)*cells.size()+3)/4,128>>>(scene,ivs,pixels,pixel_count,cells_,cells.size(),cell_radiance_,error_);
  CUDA_CHECK(cudaGetLastError(),"parallel exact cell optics launch");int status;
  CUDA_CHECK(cudaMemcpy(&status,error_,sizeof(int),cudaMemcpyDeviceToHost),"parallel exact cell optics status");
  if(status)throw std::runtime_error("invalid ray in native fallback tile");return cell_radiance_;
}
}

namespace dvs {
ExactOpticalCells OpticalSampler::prepare_exact_cells(const OpticalScene &scene,const OpticalInterval *ivs,int iv_count,
    double start,double end,PhotoParameters photo){
  if(!knots_)throw std::logic_error("refine before preparing local exact cells");const int capacity=exact_capacity_;
  if(!statistics_.exact_segments){statistics_.native_cell_queries=0;return {nullptr,nullptr,nullptr,capacity};}
  if(!exact_cells_){
    CUDA_CHECK(cudaMalloc(&exact_cells_,sizeof(ExactOpticalCell)*size_t(count_)*capacity),"allocate local exact optical cells");
    CUDA_CHECK(cudaMalloc(&exact_radiance_,sizeof(double)*size_t(count_)*capacity),"allocate local exact radiance");
    CUDA_CHECK(cudaMalloc(&exact_counts_,sizeof(int)*count_),"allocate local exact cell counts");
    CUDA_CHECK(cudaMalloc(&exact_queue_,sizeof(int)*size_t(count_)*capacity),"allocate compact exact ray queue");
    CUDA_CHECK(cudaMalloc(&exact_total_,sizeof(int)),"allocate exact query count");
  }
  CUDA_CHECK(cudaMemset(exact_total_,0,sizeof(int)),"clear exact query queue");
  identify_exact_cells<<<(count_+127)/128,128>>>(count_,ivs,iv_count,start,end,photo,knot_capacity_,knots_,knot_counts_,capacity,
    exact_cells_,exact_counts_,exact_queue_,exact_total_,device_statistics_);
  CUDA_CHECK(cudaGetLastError(),"identify local exact optical cells");int total;
  CUDA_CHECK(cudaMemcpy(&total,exact_total_,sizeof(int),cudaMemcpyDeviceToHost),"local exact query count");
  if(total){
    CUDA_CHECK(cudaMemset(error_,0,sizeof(int)),"clear local exact ray status");
    sample_exact_queries<<<(total+3)/4,128>>>(scene,ivs,exact_cells_,exact_queue_,total,capacity,exact_radiance_,error_);
    CUDA_CHECK(cudaGetLastError(),"trace local exact ray cells");int status;
    CUDA_CHECK(cudaMemcpy(&status,error_,sizeof(int),cudaMemcpyDeviceToHost),"local exact ray status");
    if(status)throw std::runtime_error("invalid ray in local exact optical cell");
  }
  CUDA_CHECK(cudaMemcpy(&statistics_,device_statistics_,sizeof(statistics_),cudaMemcpyDeviceToHost),"local exact refinement statistics");
  statistics_.native_cell_queries=total;
  return {exact_cells_,exact_radiance_,exact_counts_,capacity};
}
}
