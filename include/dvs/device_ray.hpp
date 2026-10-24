#pragma once
#include "device_types.h"
#include "dvs/scene_types.hpp"
#include "helper_math.hpp"
namespace dvs {
__device__ __forceinline__ float3 bvh_inverse_direction(const float3 dir) {
  float3 inv;
  inv.x = 1.0f / (fabsf(dir.x) > 1e-20f ? dir.x : copysignf(1e-20f, dir.x));
  inv.y = 1.0f / (fabsf(dir.y) > 1e-20f ? dir.y : copysignf(1e-20f, dir.y));
  inv.z = 1.0f / (fabsf(dir.z) > 1e-20f ? dir.z : copysignf(1e-20f, dir.z));
  return inv;
}
__device__ inline int intersect_two_children(const float3 P, const float3 idir,
                                      const float t_min, const float t_max,
                                      const AABB<float3> &box_left,
                                      const AABB<float3> &box_right,
                                      float *dist) {
  // 左子
  float3 l_min = make_float3(box_left.min.x, box_left.min.y, box_left.min.z);
  float3 l_max = make_float3(box_left.max.x, box_left.max.y, box_left.max.z);
  float t1x0 = (l_min.x - P.x) * idir.x, t1x1 = (l_max.x - P.x) * idir.x;
  float t1y0 = (l_min.y - P.y) * idir.y, t1y1 = (l_max.y - P.y) * idir.y;
  float t1z0 = (l_min.z - P.z) * idir.z, t1z1 = (l_max.z - P.z) * idir.z;
  float t1min =
      fmaxf(fmaxf(fminf(t1x0, t1x1), fminf(t1y0, t1y1)), fminf(t1z0, t1z1));
  float t1max =
      fminf(fminf(fmaxf(t1x0, t1x1), fmaxf(t1y0, t1y1)), fmaxf(t1z0, t1z1));
  t1min = fmaxf(t_min, t1min);
  t1max = fminf(t_max, t1max);

  // 右子
  float3 r_min = make_float3(box_right.min.x, box_right.min.y, box_right.min.z);
  float3 r_max = make_float3(box_right.max.x, box_right.max.y, box_right.max.z);
  float t2x0 = (r_min.x - P.x) * idir.x, t2x1 = (r_max.x - P.x) * idir.x;
  float t2y0 = (r_min.y - P.y) * idir.y, t2y1 = (r_max.y - P.y) * idir.y;
  float t2z0 = (r_min.z - P.z) * idir.z, t2z1 = (r_max.z - P.z) * idir.z;
  float t2min =
      fmaxf(fmaxf(fminf(t2x0, t2x1), fminf(t2y0, t2y1)), fminf(t2z0, t2z1));
  float t2max =
      fminf(fminf(fmaxf(t2x0, t2x1), fmaxf(t2y0, t2y1)), fmaxf(t2z0, t2z1));
  t2min = fmaxf(t_min, t2min);
  t2max = fminf(t_max, t2max);

  dist[0] = t1min;
  dist[1] = t2min;

  int mask = 0;
  if (t1max >= t1min)
    mask |= 1;
  if (t2max >= t2min)
    mask |= 2;
  return mask;
}
__device__ inline bool intersect_triangle(const float3 &ray, const float3 &ray_origin,
                                   const float3 &v0, const float3 &v1,
                                   const float3 &v2, float *out_t, float *out_u,
                                   float *out_v) {
  const float3 edge1 = v1 - v0;
  const float3 edge2 = v2 - v0;
  const float3 pvec = cross(ray, edge2);
  const float det = dot(edge1, pvec);
  if (fabsf(det) < kRayEps)
    return false;
  const float inv_det = 1.0f / det;
  const float3 tvec = ray_origin - v0;
  const float u = dot(tvec, pvec) * inv_det;
  if (u < 0.0f || u > 1.0f)
    return false;
  const float3 qvec = cross(tvec, edge1);
  const float v = dot(ray, qvec) * inv_det;
  if (v < 0.0f || (u + v) > 1.0f)
    return false;
  const float t = dot(edge2, qvec) * inv_det;
  if (t <= kRayEps)
    return false;
  *out_t = t;
  *out_u = u;
  *out_v = v;
  return true;
}
__device__ inline bool trace_shadow_ray_any_hit(
    const BVHNode<float3> *nodes, const int root_index, const float3 *vertices,
    const uint3 *indices, const float3 shadow_origin, const float3 shadow_dir,
    const int self_triangle_index, const float t_min, const float t_max) {
  float3 idir = bvh_inverse_direction(shadow_dir);
  int stack[kMaxStackSize];
  int sp = 0;
  stack[sp++] = root_index;
  while (sp > 0) {
    int ni = stack[--sp];
    const BVHNode<float3> &node = nodes[ni];
    if (node.count > 0) {
      for (int i = 0; i < node.count; ++i) {
        int tri_id = node.start + i;
        // 可选：跳过当前相机命中的同一个三角形，减少自相交风险
        if (tri_id == self_triangle_index) {
          continue;
        }
        float t, u, v;
        auto v0 = vertices[indices[tri_id].x];
        auto v1 = vertices[indices[tri_id].y];
        auto v2 = vertices[indices[tri_id].z];
        if (intersect_triangle(shadow_dir, shadow_origin, v0, v1, v2, &t, &u,
                               &v)) {
          if (t > t_min && t < t_max) {
            return true;
          }
        }
      }
      continue;
    }
    float dist[2];
    int mask = intersect_two_children(shadow_origin, idir, t_min, t_max,
                                      nodes[node.left].box,
                                      nodes[node.right].box, dist);
    if (mask == 0) {
      continue;
    }
    if (sp + (mask==3 ? 2 : 1) > kMaxStackSize) return true;
    if (mask == 3) {
      if (dist[0] < dist[1]) {
        stack[sp++] = node.right;
        stack[sp++] = node.left;
      } else {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    } else if (mask == 1) {
      stack[sp++] = node.left;
    } else {
      stack[sp++] = node.right;
    }
  }
  return false;
}
__device__ inline Intersection<float3> trace_nearest(const BVHNode<float3> *nodes,int root,
   const float3 *vertices,const uint3 *indices,float3 origin,float3 ray) {
  Intersection<float3> hit{}; float closest=1e30f;
  float3 idir=bvh_inverse_direction(ray); int stack[kMaxStackSize],sp=0;stack[sp++]=root;
  while(sp>0) {
    const auto &node=nodes[stack[--sp]];
    if(node.count>0) {
      for(int i=0;i<node.count;++i) {
        int id=node.start+i;uint3 tri=indices[id];float t,u,v;
        if(intersect_triangle(ray,origin,vertices[tri.x],vertices[tri.y],vertices[tri.z],&t,&u,&v) && t<closest) {
          closest=t;hit.hit=1;hit.triangle_index=id;hit.t=t;hit.u=u;hit.v=v;
        }
      }
    } else {
      float dist[2];int mask=intersect_two_children(origin,idir,kRayEps,closest,nodes[node.left].box,nodes[node.right].box,dist);
      if(!mask) continue;
      if(sp+(mask==3?2:1)>kMaxStackSize) { hit.hit=-1;return hit; }
      if(mask==3) {bool left=dist[0]<dist[1];stack[sp++]=left?node.right:node.left;stack[sp++]=left?node.left:node.right;}
      else stack[sp++]=mask==1?node.left:node.right;
    }
  }
  return hit;
}
} // namespace dvs
