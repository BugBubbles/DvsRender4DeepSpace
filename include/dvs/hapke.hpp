#pragma once

#include "dvs/math_types.hpp"
#include "dvs/scene_types.hpp"
#include "helper_math.hpp"

namespace dvs {
struct RoughnessTerms {
  float mu0 = 0.0f;
  float mue = 0.0f;
  float S = 0.0f;
};

inline __host__ __device__ float safe_tan(float x) {
  return tanf(clampf(x, -kPi / 2.0f + 1e-6f, kPi / 2.0f - 1e-6f));
}

inline __host__ __device__ float safe_acos(float x) {
  return acosf(clampf(x, -1.0f, 1.0f));
}

inline __host__ __device__ float HG_double_lobe(float cosg, float b, float c) {
  const float one_minus_bb = 1.0f - b * b;
  const float term1 = ((1.0f + c) * 0.5f) * one_minus_bb /
                      powf(1.0f - 2.0f * b * cosg + b * b, 1.5f);
  const float term2 = ((1.0f - c) * 0.5f) * one_minus_bb /
                      powf(1.0f + 2.0f * b * cosg + b * b, 1.5f);
  return term1 + term2;
}

inline __host__ __device__ float B_S(float tan_g_div_2, float h_S) {
  return 1.0f / (1.0f + tan_g_div_2 / h_S);
}

inline __host__ __device__ float B_C(float tan_g_div_2, float h_C) {
  const float T = tan_g_div_2 / h_C;
  if (T < 1e-8f) {
    return 1.0f;
  }
  return (1.0f + (1.0f - expf(-T)) / T) / (2.0f * (1.0f + T) * (1.0f + T));
}

inline __host__ __device__ float Hfunc(float x, float w) {
  const float root = sqrtf(fmaxf(0.0f, 1.0f - w));
  const float r0 = (1.0f - root) / (1.0f + root);
  const float term = r0 + (1.0f - 2.0f * r0 * x) * 0.5f * logf(1.0f + 1.0f / x);
  return 1.0f / (1.0f - w * x * term);
}

inline __host__ __device__ float porosity_K(float phi) {
  const float p = 1.209f * powf(phi, 0.667f);
  return -logf(1.0f - p) / p;
}

inline __host__ __device__ float chi(float thetap) {
  const float t = tanf(thetap);
  return 1.0f / sqrtf(1.0f + kPi * t * t);
}

inline __host__ __device__ float E1(float thetap, float cosy, float siny) {
  const float safe_siny = fmaxf(siny, 1e-6f);
  return expf((-2.0f / kPi) * (1.0f / tanf(thetap)) * cosy / safe_siny);
}

inline __host__ __device__ float E2(float thetap, float cosy, float siny) {
  const float safe_siny = fmaxf(siny, 1e-6f);
  const float v = (1.0f / tanf(thetap)) * cosy / safe_siny;
  return expf((-1.0f / kPi) * v * v);
}

inline __host__ __device__ float eta(float thetap, float cosy, float siny) {
  const float ch = chi(thetap);
  const float e1 = E1(thetap, cosy, siny);
  const float e2 = E2(thetap, cosy, siny);
  return ch * (cosy + siny * tanf(thetap) * e2 / (2.0f - e1));
}

inline __host__ __device__ float f_psi(float psi) {
  return expf(-2.0f * tanf(psi * 0.5f));
}

inline __host__ __device__ RoughnessTerms rough_S(float cosi, float cose,
                                                  float sini, float sine,
                                                  float psi, float thetap) {
  const float ch = chi(thetap);
  const float tan_thetap = tanf(thetap);
  const float sin_psi_half_sq = sinf(psi * 0.5f);
  const float sin_psi_half_sq_v = sin_psi_half_sq * sin_psi_half_sq;

  RoughnessTerms out{};
  if (cosi >= cose) {
    out.mu0 = ch * (cosi + sini * tan_thetap *
                               (cosf(psi) * E2(thetap, cose, sine) +
                                sin_psi_half_sq_v * E2(thetap, cosi, sini)) /
                               (2.0f - E1(thetap, cose, sine) -
                                (psi / kPi) * E1(thetap, cosi, sini)));
    out.mue = ch * (cose + sine * tan_thetap *
                               (E2(thetap, cose, sine) -
                                sin_psi_half_sq_v * E2(thetap, cosi, sini)) /
                               (2.0f - E1(thetap, cose, sine) -
                                (psi / kPi) * E1(thetap, cosi, sini)));
    out.S = out.mue / eta(thetap, cose, sine) * cosi / eta(thetap, cosi, sini) *
            ch /
            (1.0f - f_psi(psi) +
             f_psi(psi) * ch * (cosi / eta(thetap, cosi, sini)));
  } else {
    out.mu0 = ch * (cosi + sini * tan_thetap *
                               (E2(thetap, cosi, sini) -
                                sin_psi_half_sq_v * E2(thetap, cose, sine)) /
                               (2.0f - E1(thetap, cosi, sini) -
                                (psi / kPi) * E1(thetap, cose, sine)));
    out.mue = ch * (cose + sine * tan_thetap *
                               (cosf(psi) * E2(thetap, cosi, sini) +
                                sin_psi_half_sq_v * E2(thetap, cose, sine)) /
                               (2.0f - E1(thetap, cosi, sini) -
                                (psi / kPi) * E1(thetap, cose, sine)));
    out.S = out.mue / eta(thetap, cose, sine) * cosi / eta(thetap, cosi, sini) *
            ch /
            (1.0f - f_psi(psi) +
             f_psi(psi) * ch * (cose / eta(thetap, cose, sine)));
  }

  return out;
}
inline __host__ __device__ float
hapke_brdf(const float3 &normal, const float3 &light_dir,
           const float3 &view_dir, const float albedo, const HapkeConfig &p) {
  const float3 n = normalize(normal);
  const float3 l = normalize(light_dir);
  const float3 v = normalize(view_dir);
  float cosi = dot(n, l);
  float cose = dot(n, v);
  if (cosi <= 0.0f || cose <= 0.0f) {
    return 0.0f;
  }

  cosi = fmaxf(cosi, 0.0f);
  cose = fmaxf(cose, 0.0f);
  const float sini = fmaxf(length(cross(n, l)), 1e-6f);
  const float sine = fmaxf(length(cross(n, v)), 1e-6f);
  const float cosg = clampf(dot(l, v), -1.0f, 1.0f);
  const float tan_g_div_2 = safe_tan(safe_acos(cosg) * 0.5f);
  const float psi = safe_acos((cosg - cosi * cose) / (sini * sine + 1e-6f));

  const float p_g = HG_double_lobe(cosg, p.b, p.c);
  const float BS = B_S(tan_g_div_2, p.hS);
  const float BC = B_C(tan_g_div_2, p.hC);
  const float K = porosity_K(p.phi);
  const RoughnessTerms rst = rough_S(cosi, cose, sini, sine, psi, p.thetap);
  const float M =
      Hfunc(rst.mu0 / K, albedo) * Hfunc(rst.mue / K, albedo) - 1.0f;
  const float LS = rst.mu0 / (rst.mu0 + rst.mue + 1e-6f);

  const float brdf = (albedo / (4.0f * kPi)) * (LS / cosi) * K *
                     (p_g * (1.0f + BS * p.B_S0) + M) * (1.0f + BC * p.B_C0) *
                     rst.S;
  return brdf * cosi;
}

} // namespace dvs
