#pragma once
#include "dvs/optical_sensor.hpp"
#include "dvs/pixel_model.hpp"
#include <vector>
namespace dvs {
constexpr double kDefaultOpticalLogTolerance=1e-2;
constexpr double kOpticalVoltageThresholdFraction=1e-3;
struct OpticalValue {double radiance=0;float depth=-1;int samples=0;};
struct OpticalCell {double start=0,end=0;int interval=0;};
struct ExactOpticalCell {double start=0,end=0;int interval=0,index=0;};
struct ExactOpticalCells {const ExactOpticalCell *cells=nullptr;const double *radiance=nullptr;const int *counts=nullptr;int stride=64;};
// A negative endpoint fraction marks its preceding segment for direct tracing.
struct OpticalKnot { double fraction=0,radiance=0; };
struct OpticalKnots {const OpticalKnot *values=nullptr;const int *counts=nullptr;int stride=0;};
struct OpticalProbeResult {double maximum_log_error=0,p99_log_error=0,mean_log_error=0;int probes=0,over_tolerance=0,exact_fallback_probes=0;};
struct OpticalRefinementStats {unsigned long long traced_pixel_samples=0;int fallback_pixels=0,maximum_knots=0,exact_segments=0,native_cell_queries=0;};
class OpticalSampler {
public:
  OpticalSampler(int width,int height,int maximum_samples);
  ~OpticalSampler();
  OpticalSampler(const OpticalSampler&)=delete;
  OpticalSampler& operator=(const OpticalSampler&)=delete;
  // A warp cooperates on one pixel; ordered reduction retains native quadrature.
  const OpticalValue* evaluate(const OpticalScene&,const std::vector<OpticalState>&);
  // Each pixel independently refines direct-ray photocurrent probes. Failed
  // convergence/overflow is marked for exact native-cell tracing, never clipped.
  OpticalKnots refine(const OpticalScene&,const OpticalInterval *device_intervals,int interval_count,
                      double start,double end,PhotoParameters,double log_tolerance,const PixelState*,double charge_scale);
  ExactOpticalCells prepare_exact_cells(const OpticalScene&,const OpticalInterval *device_intervals,int interval_count,
                                        double start,double end,PhotoParameters);
  const double* sample_cells(const OpticalScene&,const OpticalInterval *device_intervals,
                            const int *device_pixels,int pixel_count,const std::vector<OpticalCell>&);
  OpticalProbeResult verify(const OpticalScene&,const OpticalInterval *device_intervals,int interval_count,
                            double start,double end,PhotoParameters,double tolerance);
  const OpticalRefinementStats& statistics()const{return statistics_;}
  int knot_capacity()const{return knot_capacity_;}
  int exact_capacity()const{return exact_capacity_;}
  void download(std::vector<OpticalValue>&)const;
private:
  int knot_capacity_=128,exact_capacity_=128;
  int count_,maximum_,poses_count_=0;std::size_t capacity_=0;
  ExactOpticalCell *exact_cells_=nullptr;double *exact_radiance_=nullptr;
  int *exact_counts_=nullptr,*exact_queue_=nullptr,*exact_total_=nullptr;
  double *cell_radiance_=nullptr;OpticalCell *cells_=nullptr;std::size_t cell_capacity_=0;
  OpticalKnot *knots_=nullptr;int *knot_counts_=nullptr;
  OpticalRefinementStats *device_statistics_=nullptr,statistics_{};
  OpticalState *poses_=nullptr;OpticalValue *values_=nullptr;int *error_=nullptr;
};
}
