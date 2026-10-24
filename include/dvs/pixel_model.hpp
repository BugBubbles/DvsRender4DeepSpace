#pragma once
#include <cuda_runtime.h>
#include <cmath>
namespace dvs {
// Reduced subthreshold current-balance receptor; rates are electrons/s.
// u = V/(kT/(q*kappa)), du/dt = (rate-exp(u))/charge_scale.
struct PixelParameters {
  double threshold_pos=.3, threshold_neg=.3;
  double charge_scale=200, refractory_time=.0002;
  double reference_leak=0, reset_noise_std=0;
};
struct PixelState {
  double receptor_log=0, reference_log=0, refractory_until=-1;
  double threshold_pos=.3, threshold_neg=.3;
  bool reset_pending=false;
};
struct PixelEvent { double time; int polarity; };
__host__ __device__ inline double receptor_log_at(double u, double rate, double dt, double charge) {
  if (dt<=0) return u;
  rate=fmax(0.0,rate);
  double x=exp(u);
  if (rate==0) return u-log1p(x*dt/charge);
  // z=1/x obeys a linear ODE. expm1 retains small increments.
  double a=exp(-rate*dt/charge), b=-expm1(-rate*dt/charge);
  return -log(a/x+b/rate);
}
__host__ __device__ inline double contrast_at(double u, double rate, double dt,
                                             double ref, const PixelParameters &p) {
  return receptor_log_at(u,rate,dt,p.charge_scale)-ref+p.reference_leak*dt;
}
// Exact receptor evolution for a constant photocurrent integration cell;
// comparator roots are solved on the nonlinear voltage trajectory, not an
// interpolation of intensity frames. RNG provides normal() for reset kTC noise.
template<class Rng>
__host__ __device__ int advance_pixel(PixelState &s,const PixelParameters &p,double rate,
                                     double t0,double t1,PixelEvent *out,int capacity,Rng &rng) {
  if(t1<=t0) return 0;
  int count=0; double cursor=t0, u=s.receptor_log;
  while(cursor<t1) {
    if(cursor<s.refractory_until) {
      double end=fmin(t1,s.refractory_until);
      u=receptor_log_at(u,rate,end-cursor,p.charge_scale);
      cursor=end; s.reference_log=u;
      if(cursor<t1 || cursor==s.refractory_until) {
        if(cursor>=s.refractory_until && s.reset_pending) {
          if(p.reset_noise_std>0) s.reference_log+=p.reset_noise_std*rng.normal();
          s.reset_pending=false;
        }
      }
      if(cursor>=t1) break;
    } else if(s.reset_pending) {
      if(p.reset_noise_std>0) s.reference_log+=p.reset_noise_std*rng.normal();
      s.reset_pending=false;
    }
    double remaining=t1-cursor;
    double c0=u-s.reference_log;
    double boundary=remaining;
    // Falling receptor plus ON leakage may have an interior minimum. Split
    // there so opposite-polarity crossings cannot cancel at the endpoints.
    double x=exp(u), target=rate+p.charge_scale*p.reference_leak;
    if(p.reference_leak>0 && x>target && target>rate) {
      double extremum;
      if(rate>0) extremum=-p.charge_scale/rate*log((1.0/target-1.0/rate)/(1.0/x-1.0/rate));
      else extremum=p.charge_scale*(1.0/target-1.0/x);
      if(extremum>1e-15 && extremum<boundary) boundary=extremum;
    }
    double ub=receptor_log_at(u,rate,boundary,p.charge_scale);
    double cb=ub-s.reference_log+p.reference_leak*boundary;
    int polarity=-1; double level=0;
    if(c0>=s.threshold_pos) {polarity=1;level=s.threshold_pos;}
    else if(c0<=-s.threshold_neg) {polarity=0;level=-s.threshold_neg;}
    else if(cb>=s.threshold_pos) {polarity=1;level=s.threshold_pos;}
    else if(cb<=-s.threshold_neg) {polarity=0;level=-s.threshold_neg;}
    if(polarity<0) {
      u=ub;
      s.reference_log-=p.reference_leak*boundary; cursor+=boundary;
      continue;
    }
    if(count==capacity) return -1;
    double lo=0,hi=boundary;
    if(c0>=s.threshold_pos || c0<=-s.threshold_neg) hi=0;
    else if(p.reference_leak==0) {
      double target_inverse=exp(-(s.reference_log+level));
      double initial_inverse=exp(-u);
      if(rate>0) {
        double stationary_inverse=1.0/rate;
        hi=-p.charge_scale/rate*log1p((target_inverse-initial_inverse)/(initial_inverse-stationary_inverse));
      } else hi=p.charge_scale*(target_inverse-initial_inverse);
      hi=fmin(boundary,fmax(0.0,hi));
    }
    else for(int j=0;j<48;++j) {
      double mid=(lo+hi)*.5, c=contrast_at(u,rate,mid,s.reference_log,p);
      if(polarity ? c>=level : c<=level) hi=mid; else lo=mid;
    }
    u=receptor_log_at(u,rate,hi,p.charge_scale); cursor+=hi;
    out[count++]={cursor,polarity};
    // Self-timed reset balances the change amplifier at its actual voltage.
    s.reference_log=u; s.refractory_until=cursor+p.refractory_time;
    s.reset_pending=true;
  }
  s.receptor_log=u; return count;
}
} // namespace dvs
