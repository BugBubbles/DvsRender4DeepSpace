#pragma once
#include "dvs/optical_sensor.hpp"
#include "dvs/device_ray.hpp"
#include "dvs/hapke.hpp"
namespace dvs {
__device__ inline float hermite(float a,float va,float b,float vb,float u,float dt) {
  float u2=u*u,u3=u2*u;
  return a+(-2*u3+3*u2)*(b-a)+(u3-2*u2+u)*dt*va+(u3-u2)*dt*vb;
}
__device__ inline float3 hermite3(float3 a,float3 va,float3 b,float3 vb,float u,float dt) {
  return make_float3(hermite(a.x,va.x,b.x,vb.x,u,dt),hermite(a.y,va.y,b.y,vb.y,u,dt),hermite(a.z,va.z,b.z,vb.z,u,dt));
}
__device__ inline OpticalState optical_state_at(const OpticalInterval &iv,double t) {
  float dt=iv.t1-iv.t0,u=dt>0?fmin(1.0,fmax(0.0,(t-iv.t0)/dt)):0;
  OpticalState s=iv.start;
  s.position=hermite3(iv.start.position,iv.start.velocity,iv.end.position,iv.end.velocity,u,dt);
  s.light=hermite3(iv.start.light,iv.start.light_rate,iv.end.light,iv.end.light_rate,u,dt);
  s.intensity=fmaxf(0,hermite(iv.start.intensity,iv.start.intensity_rate,iv.end.intensity,iv.end.intensity_rate,u,dt));
  float4 a=iv.start.rotation,b=iv.end.rotation;float d=a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
  if(d<0) {b=make_float4(-b.x,-b.y,-b.z,-b.w);d=-d;}
  float wa=1-u,wb=u;
  if(d<.9995f) {float angle=acosf(fminf(1,d)),inv=1/sinf(angle);wa=sinf((1-u)*angle)*inv;wb=sinf(u*angle)*inv;}
  float4 q=make_float4(wa*a.x+wb*b.x,wa*a.y+wb*b.y,wa*a.z+wb*b.z,wa*a.w+wb*b.w);
  float norm=rsqrtf(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
  s.rotation=make_float4(q.x*norm,q.y*norm,q.z*norm,q.w*norm);return s;
}
__device__ inline float3 rotate_ray(float4 q,float3 v) {
  float3 qv=make_float3(q.x,q.y,q.z);float3 c=2*cross(qv,v);return v+q.w*c+cross(qv,c);
}
// Missing material data uses the configured homogeneous SSA, preserving terrain geometry.
__device__ inline float material_albedo(float value,float fallback) {
  return isfinite(value) && value>=0 && value<=1?value:fallback;
}
// Reconstruct on the triangle instead of O+tD, whose cancellation at km
// scale can place the origin beneath the surface. Offset in geometric-normal
// direction using a float-arithmetic error allowance, then round outward.
__device__ inline float3 shadow_origin(const OpticalScene &scene,uint3 tri,float u,float v,float3 light){
  float3 a=scene.vertices[tri.x],e1=scene.vertices[tri.y]-a,e2=scene.vertices[tri.z]-a;
  float3 point=a+u*e1+v*e2,normal=normalize(cross(e1,e2));
  if(dot(normal,light)<0)normal=-normal;
  constexpr float epsilon=5.960464477539063e-8f,gamma=8*epsilon/(1-8*epsilon);
  float3 error=make_float3(gamma*(fabsf(a.x)+fabsf(u*e1.x)+fabsf(v*e2.x)),
                          gamma*(fabsf(a.y)+fabsf(u*e1.y)+fabsf(v*e2.y)),
                          gamma*(fabsf(a.z)+fabsf(u*e1.z)+fabsf(v*e2.z)));
  float distance=fabsf(normal.x)*error.x+fabsf(normal.y)*error.y+fabsf(normal.z)*error.z+1e-6f;
  float3 offset=normal*distance;point=point+offset;
  if(offset.x!=0)point.x=nextafterf(point.x,copysignf(INFINITY,offset.x));
  if(offset.y!=0)point.y=nextafterf(point.y,copysignf(INFINITY,offset.y));
  if(offset.z!=0)point.z=nextafterf(point.z,copysignf(INFINITY,offset.z));
  return point;
}
struct OpticalSample {double radiance=0;float depth=-1;bool valid=true;};
struct RayContribution {double radiance=0;float depth=-1;bool valid=true;};
__device__ inline RayContribution sample_subray(const OpticalScene &scene,int pixel,const OpticalState &pose,int sx,int sy,int ss) {
  RayContribution out;
  float px=(pixel%scene.width+(sx+.5f)/ss-.5f-scene.cx)/scene.fx;
  float py=(pixel/scene.width+(sy+.5f)/ss-.5f-scene.cy)/scene.fy;
  float3 ray=normalize(rotate_ray(pose.rotation,make_float3(px,-py,-1)));
  auto hit=trace_nearest(scene.nodes,scene.root,scene.vertices,scene.indices,pose.position,ray);
  if(hit.hit<0){out.valid=false;return out;}
  if(!hit.hit)return out;
  out.depth=hit.t;auto tri=scene.indices[hit.triangle_index];float w=1-hit.u-hit.v;
  float3 normal=normalize(scene.normals[tri.x]*w+scene.normals[tri.y]*hit.u+scene.normals[tri.z]*hit.v);
  float3 light=normalize(pose.light);
  if(dot(normal,light)<=0 || dot(normal,-ray)<=0)return out;
  float3 origin=shadow_origin(scene,tri,hit.u,hit.v,light);
  if(trace_shadow_ray_any_hit(scene.nodes,scene.root,scene.vertices,scene.indices,origin,light,hit.triangle_index,kRayEps,1e30f))return out;
  float albedo=scene.albedos?material_albedo(scene.albedos[tri.x],scene.hapke.w)*w+material_albedo(scene.albedos[tri.y],scene.hapke.w)*hit.u+material_albedo(scene.albedos[tri.z],scene.hapke.w)*hit.v:scene.hapke.w;
  double cos4=1.0/((1.0+px*px+py*py)*(1.0+px*px+py*py));
  out.radiance=fmaxf(0,hapke_directional_reflectance(normal,light,-ray,albedo,scene.hapke)*pose.intensity)*cos4;
  return out;
}
__device__ inline OpticalSample sample_pixel(const OpticalScene &scene,int pixel,const OpticalState &pose) {
  OpticalSample out;double depth=0;int valid_depth=0;
  float3 light=normalize(pose.light);int ss=scene.samples;
  for(int sy=0;sy<ss;++sy) for(int sx=0;sx<ss;++sx) {
    float px=(pixel%scene.width+(sx+.5f)/ss-.5f-scene.cx)/scene.fx;
    float py=(pixel/scene.width+(sy+.5f)/ss-.5f-scene.cy)/scene.fy;
    float3 ray=normalize(rotate_ray(pose.rotation,make_float3(px,-py,-1)));
    auto hit=trace_nearest(scene.nodes,scene.root,scene.vertices,scene.indices,pose.position,ray);
    if(hit.hit<0) {out.valid=false;return out;}
    if(hit.hit==0) continue;
    depth+=hit.t;++valid_depth;
    auto tri=scene.indices[hit.triangle_index];float w=1-hit.u-hit.v;
    float3 normal=normalize(scene.normals[tri.x]*w+scene.normals[tri.y]*hit.u+scene.normals[tri.z]*hit.v);
    if(dot(normal,light)<=0 || dot(normal,-ray)<=0) continue;
    float3 origin=shadow_origin(scene,tri,hit.u,hit.v,light);
    if(trace_shadow_ray_any_hit(scene.nodes,scene.root,scene.vertices,scene.indices,origin,light,hit.triangle_index,kRayEps,1e30f)) continue;
    float albedo=scene.albedos?material_albedo(scene.albedos[tri.x],scene.hapke.w)*w+material_albedo(scene.albedos[tri.y],scene.hapke.w)*hit.u+material_albedo(scene.albedos[tri.z],scene.hapke.w)*hit.v:scene.hapke.w;
    double cos4=1.0/((1.0+px*px+py*py)*(1.0+px*px+py*py));
    out.radiance+=fmaxf(0,hapke_directional_reflectance(normal,light,-ray,albedo,scene.hapke)*pose.intensity)*cos4;
  }
  out.radiance/=ss*ss;out.depth=valid_depth?depth/valid_depth:-1;return out;
}
} // namespace dvs
