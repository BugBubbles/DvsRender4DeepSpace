#pragma once
#include <curand_kernel.h>
namespace dvs {
__device__ inline double poisson_electrons(curandState *rng,double lambda) {
  if(lambda<=0) return 0;
  // Normal approximation only at high count where relative error is small.
  if(lambda>1e6) return fmax(0.0,floor(lambda+sqrt(lambda)*curand_normal_double(rng)+.5));
  return curand_poisson(rng,lambda);
}
}
