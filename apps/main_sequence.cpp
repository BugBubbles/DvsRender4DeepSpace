#include "bvh/build.h"
#include "dvs/dataset_budget.hpp"
#include "dvs/dvs_writer.hpp"
#include "dvs/frame_writer.hpp"
#include "dvs/repeated_trajectory.hpp"
#include "dvs/raycast.hpp"
#include "dvs/scene_data.hpp"
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

namespace {
constexpr double duration=60;
struct GeographicReference {
  using Dataset=std::unique_ptr<GDALDataset,void(*)(GDALDataset*)>;
  Dataset dem;
  OGRSpatialReference projected;
  std::unique_ptr<OGRSpatialReference> geographic;
  std::unique_ptr<OGRCoordinateTransformation,void(*)(OGRCoordinateTransformation*)> transform;
  double x=0,y=0,z=0;
  explicit GeographicReference(const std::filesystem::path &path)
    :dem(nullptr,[](GDALDataset*p){if(p)GDALClose(p);}),
     transform(nullptr,[](OGRCoordinateTransformation*p){if(p)OCTDestroyCoordinateTransformation(p);}) {
    GDALAllRegister();dem.reset(static_cast<GDALDataset*>(GDALOpen(path.c_str(),GA_ReadOnly)));
    if(!dem)throw std::runtime_error("cannot open source DEM for geographic poses");
    double gt[6];if(dem->GetGeoTransform(gt)!=CE_None)throw std::runtime_error("DEM has no affine transform");
    int col=dem->GetRasterXSize()/2,row=dem->GetRasterYSize()/2;
    x=gt[0]+(col+.5)*gt[1]+(row+.5)*gt[2];y=gt[3]+(col+.5)*gt[4]+(row+.5)*gt[5];
    auto*band=dem->GetRasterBand(1);float value;
    if(band->RasterIO(GF_Read,col,row,1,1,&value,1,1,GDT_Float32,0,0)!=CE_None)
      throw std::runtime_error("cannot read DEM reference height");
    int has_nodata,has_scale,has_offset;double nodata=band->GetNoDataValue(&has_nodata);
    double scale=band->GetScale(&has_scale),offset=band->GetOffset(&has_offset);
    if(!std::isfinite(value) || (has_nodata && value==float(nodata)))throw std::runtime_error("DEM center is NoData");
    z=value*(has_scale?scale:1)+(has_offset?offset:0);
    if(projected.SetFromUserInput(dem->GetProjectionRef())!=OGRERR_NONE)throw std::runtime_error("invalid DEM CRS");
    projected.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    geographic.reset(projected.CloneGeogCS());if(!geographic)throw std::runtime_error("missing source planetary geographic CRS");
    geographic->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    transform.reset(OGRCreateCoordinateTransformation(&projected,geographic.get()));
    if(!transform)throw std::runtime_error("cannot convert source map to lunar longitude/latitude");
  }
};
void prepare_poses(const dvs::RepeatedTrajectory &trajectory,const dvs::Config &c,
                   const std::filesystem::path &dem,const std::filesystem::path &out,const dvs::DatasetBudget &budget){
  GeographicReference map(dem);
  std::ostringstream poses,motion,geo,camera;
  poses<<std::setprecision(12)<<"t_s,x_m,y_m,z_m,qx,qy,qz,qw\n";
  motion<<std::setprecision(12)<<"t_s,vx_world_m_s,vy_world_m_s,vz_world_m_s,wx_world_rad_s,wy_world_rad_s,wz_world_rad_s,sun_x,sun_y,sun_z,sun_intensity\n";
  geo<<std::setprecision(15)<<"t_s,projected_x_m,projected_y_m,source_altitude_m,longitude_deg,latitude_deg\n";
  auto cv_from_source=dvs::Mat3::Identity().eval();cv_from_source(1,1)=cv_from_source(2,2)=-1;
  Eigen::Quaternionf previous(1,0,0,0);
  std::size_t n=std::ceil(duration*c.scene.imu_fps);
  for(std::size_t i=0;i<=n;++i){
    double t=std::min(duration,i/double(c.scene.imu_fps));auto s=trajectory.evaluate(t);
    Eigen::Quaternionf q(s.orientation*cv_from_source);q.normalize();
    if(i && q.coeffs().dot(previous.coeffs())<0)q.coeffs()*=-1;previous=q;
    poses<<t<<','<<s.position.x()<<','<<s.position.y()<<','<<s.position.z()<<','<<q.x()<<','<<q.y()<<','<<q.z()<<','<<q.w()<<'\n';
    motion<<t<<','<<s.linear_velocity.x()<<','<<s.linear_velocity.y()<<','<<s.linear_velocity.z()<<','
          <<s.angular_velocity.x()<<','<<s.angular_velocity.y()<<','<<s.angular_velocity.z()<<','
          <<s.light_direction.x()<<','<<s.light_direction.y()<<','<<s.light_direction.z()<<','<<s.light_intensity<<'\n';
    double px=map.x+s.position.x(),py=map.y+s.position.y(),longitude=px,latitude=py;
    if(!map.transform->Transform(1,&longitude,&latitude))throw std::runtime_error("geographic pose conversion failed");
    geo<<t<<','<<px<<','<<py<<','<<map.z+s.position.z()<<','<<longitude<<','<<latitude<<'\n';
  }
  camera<<std::setprecision(9)<<"model: pinhole\nwidth: "<<c.camera.width<<"\nheight: "<<c.camera.height
        <<"\nfx: "<<c.camera.fx<<"\nfy: "<<c.camera.fy<<"\ncx: "<<c.camera.cx<<"\ncy: "<<c.camera.cy
        <<"\nsupersample_factor: "<<c.scene.supersample_factor<<"\ntime_resolution: "<<c.scene.time_resolution
        <<"\ncircuit_step: "<<c.physic.circuit_step<<"\nthreshold_pos: "<<c.physic.threshold_pos<<"\nthreshold_neg: "<<c.physic.threshold_neg
        <<"\ncam_fps: "<<c.scene.cam_fps<<"\npose_fps: "<<c.scene.imu_fps<<"\nduration_seconds: 60"
        <<"\npixel_centers: integer\npose_axes: opencv_x_right_y_down_z_forward\nsource_model: source_config.yaml"
        <<"\ndistortion: ideal_pinhole_zero_distortion_not_hardware_calibration"
        <<"\nphysics_calibration: generic_uncalibrated_optical_and_circuit_parameters"
        <<"\ngeographic_axes: local_projected_raster_x_y_and_source_height_relative_to_center"
        <<"\nvertical_datum: as_provided_not_independently_verified"
        <<"\nimage_model: instantaneous_fixed_exposure_sRGB_preview_not_calibrated_APS"
        <<"\nimage_encoding: sRGB_of_linear_render_units_divided_by_100"
        <<"\ndepth_model: mean_hit_ray_range_m_minus_one_on_miss"
        <<"\nevent_timestamp: int64_microseconds_transport_completion"
        <<"\nframe_timestamp: int64_microseconds_instantaneous_sample"
        <<"\nmotion: original_speed_ping_pong_with_0.5_second_C2_reversals\n";
  budget.write_text(out/"poses.csv",poses.str());budget.write_text(out/"motion.csv",motion.str());
  budget.write_text(out/"pose_geographic.csv",geo.str());budget.write_text(out/"camera_used.yaml",camera.str());
}
}

int main(int argc,char **argv){
  try{
    bool poses_only=argc>1 && std::string(argv[1])=="--prepare-poses";
    bool optimized=argc>1 && std::string(argv[1])=="--optimized";int shift=(poses_only || optimized)?1:0;
    if(argc!=5+shift)throw std::runtime_error("sequence [--prepare-poses|--optimized] dataset_root sequence_dir dem albedo_or_empty");
    std::filesystem::path root=argv[1+shift],out=argv[2+shift],dem=argv[3+shift];std::string albedo=argv[4+shift];
    dvs::DatasetBudget budget(root);budget.require_path(out/"sequence.json");
    std::string error;dvs::Config config((out/"source_config.yaml").string(),error);
    if(!error.empty())throw std::runtime_error(error);
    if(config.scene.time_resolution<=0 || config.scene.cam_fps<=0 || config.scene.imu_fps<=0)
      throw std::runtime_error("invalid source timing parameters");
    std::vector<dvs::FrameData<dvs::Vec3,dvs::Mat3>> frames;
    if(!dvs::load_frames_from_text(out/"source_trajectory.txt",frames,&error) || frames.size()<2)
      throw std::runtime_error("failed to load source trajectory: "+error);
    dvs::CameraTrajectory source(frames,config.scene);dvs::RepeatedTrajectory trajectory(source);
    if(std::filesystem::exists(out/"events.h5") || std::filesystem::exists(out/"frames.h5"))
      throw std::runtime_error("refusing to overwrite an existing rendered or partial sequence");
    prepare_poses(trajectory,config,dem,out,budget);
    if(poses_only){std::cout<<"Prepared 60 s native-camera poses and geographic coordinates: "<<out<<'\n';return 0;}
#if !defined(ENABLE_PHYSIC_EFFECTS) || !defined(ENABLE_AER_DELAY)
    throw std::runtime_error("physical sequence generation requires enabled circuit effects and configured FIFO readout");
#endif
    budget.write(0,[]{});
    dvs::SceneBundle<float3,uint3> mesh;
    bool loaded=albedo.empty()?dvs::load_scene_bundle(dem,mesh,&error):dvs::load_scene_bundle(dem,albedo,mesh,&error);
    if(!loaded)throw std::runtime_error("failed to load exact source DEM/material: "+error);
    dvs::BVHBuildParams params;params.max_leaf_size=64;
    int bvh_root=dvs::BVHBuilder::build(params,mesh.bvh_nodes,mesh.bvh_triangles,mesh.vertices);
    // One GPU copy of the exact indexed terrain shared by image and event rays.
    auto scene=std::make_shared<dvs::DeviceScene>(mesh,config,bvh_root);
    mesh={};std::cout<<"SCENE_GPU_READY"<<std::endl;
    dvs::DvsRayCaster sensor(scene,config);dvs::CamRayCaster camera(scene);
    dvs::Hdf5EventStreamWriter events((out/"events.h5").string(),&budget);
    dvs::Hdf5FrameWriter images(out/"frames.h5",config.camera.width,config.camera.height,budget);
    sensor.initialize(trajectory.evaluate(0),0);
    std::vector<float> radiance,depth;
    auto frame_at=[&](double t){auto p=trajectory.evaluate(t);
      camera.render_main(dvs::vec3_to_float3(p.position),p.orientation.data(),dvs::vec3_to_float3(p.light_direction),p.light_intensity,radiance,depth);
      images.append(t,radiance,depth);
    };
    frame_at(0);std::size_t frame_index=1,step=0;
    double now=0,last_report=0,dt=config.scene.time_resolution;
    auto start=std::chrono::steady_clock::now();
    std::vector<dvs::Event> ready;std::vector<dvs::OpticalInterval> intervals;intervals.reserve(64);
    // Keep every configured 10 us trajectory interval and every 1 us circuit
    // cell. Optimized mode changes only optical time sampling, with direct-ray
    // fallback and checks before changing pixel/RNG state.
    constexpr double optical_tolerance=dvs::kDefaultOpticalLogTolerance;
    unsigned long long optical_queries=0,fallback_queries=0,fallback_pixel_windows=0,windows=0;
    int maximum_knots=0;unsigned convergence_checks=0,convergence_rejections=0,native_guard_windows=0;
    double next_convergence_check=0,maximum_receptor_error=0,maximum_p99_receptor_error=0;
    unsigned batch=optimized?200:64;
    if(config.physic.refractory_time<=0)batch=1;
    else batch=std::max(1U,std::min(batch,unsigned(std::max(1.0,(config.physic.max_events_per_step-2)*double(config.physic.refractory_time)/dt))));
    const unsigned maximum_batch=batch;unsigned stable_quality_checks=0;
    if(optimized){
      std::ostringstream model;model<<std::setprecision(12)
        <<"{\"model\":\"ray_photocurrent_adaptive_optics_native_circuit\",\"maximum_window_seconds\":"<<batch*dt
        <<",\"empirical_midpoint_log_photocurrent_tolerance\":"<<optical_tolerance
        <<",\"dim_leaf_max_expected_electrons\":1.0"
        <<",\"leaf_voltage_error_threshold_fraction\":"<<dvs::kOpticalVoltageThresholdFraction
        <<",\"maximum_knots_per_pixel\":"<<sensor.adaptive_cache_capacity()<<",\"maximum_local_exact_cells_per_pixel\":"<<sensor.adaptive_cache_capacity()<<",\"native_spatial_grid\":"<<config.scene.supersample_factor
        <<",\"native_trajectory_interval_seconds\":"<<dt<<",\"native_circuit_step_seconds\":"<<config.physic.circuit_step
        <<",\"unresolved_pixel_action\":\"direct_rays_at_every_native_circuit_midpoint\""
        <<",\"optical_error_guarantee\":\"empirical_probe_convergence_not_global_visibility_bound\""
        <<",\"spatial_adaptation\":false,\"per_event_auxiliary_depth\":\"not_available_HDF5_uses_x_y_t_p_only\""
        <<",\"camera_parameters\":\"source_config.yaml_unchanged\",\"physical_parameters\":\"generic_uncalibrated\""
        <<",\"quality_check_interval_seconds\":0.25,\"mean_current_receptor_probe_pixels\":1024,\"quality_max_error_threshold_fraction\":0.02,\"quality_p99_error_threshold_fraction\":0.005"
        <<",\"shadow_origin\":\"barycentric_geometry_float_error_offset_and_outward_rounding\"}";
      budget.write_text(out/"numerical_model.json",model.str());
    }
    auto left=dvs::to_optical_state(trajectory.evaluate(0));
    while(now<duration){
      double window_start=now;size_t window_step=step;auto window_left=left;
      intervals.clear();
      for(unsigned i=0;i<batch && now<duration;++i){
        double next=std::min(duration,(++step)*dt);if(next<=now)throw std::runtime_error("non-increasing integration grid");
        auto right=dvs::to_optical_state(trajectory.evaluate(next));intervals.push_back({left,right,now,next});left=right;now=next;
      }
      if(optimized){
        bool check=window_start>=next_convergence_check;
        bool accepted=sensor.advance_adaptive(intervals,optical_tolerance,ready,check);
        if(check){auto result=sensor.last_convergence();++convergence_checks;
          maximum_receptor_error=std::max(maximum_receptor_error,result.maximum_log_voltage_error);
          maximum_p99_receptor_error=std::max(maximum_p99_receptor_error,result.p99_log_voltage_error);
        }
        if(!accepted){++convergence_rejections;stable_quality_checks=0;
          if(batch>1){now=window_start;step=window_step;left=window_left;batch=std::max(1U,batch/2);continue;}
          sensor.advance_many(intervals,ready);++native_guard_windows;
        }
        if(check){next_convergence_check=now+.25;
          auto quality=sensor.last_convergence();double threshold=std::min(config.physic.threshold_pos,config.physic.threshold_neg);
          if(accepted && quality.maximum_log_voltage_error<.005*threshold && quality.p99_log_voltage_error<.001*threshold)++stable_quality_checks;
          else stable_quality_checks=0;
          if(stable_quality_checks>=4 && batch<maximum_batch){batch=std::min(maximum_batch,batch*2);stable_quality_checks=0;}
        }
        auto statistics=sensor.refinement_statistics();
        optical_queries+=statistics.traced_pixel_samples+statistics.native_cell_queries;fallback_pixel_windows+=statistics.fallback_pixels;
        unsigned long long cells=0;for(auto iv:intervals)cells+=std::ceil((iv.t1-iv.t0)/config.physic.circuit_step);
        fallback_queries+=cells*statistics.fallback_pixels;maximum_knots=std::max(maximum_knots,statistics.maximum_knots);++windows;
      }else sensor.advance_many(intervals,ready);
      events.append(std::move(ready));ready.clear();
      while(frame_index/double(config.scene.cam_fps)<=now+1e-12 && frame_index/double(config.scene.cam_fps)<=duration){
        frame_at(frame_index/double(config.scene.cam_fps));++frame_index;
      }
      double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      if(elapsed-last_report>=30){
        std::ostringstream progress;progress<<std::setprecision(12)<<"{\"status\":\"rendering\",\"observed_seconds\":"<<now
          <<",\"wall_seconds\":"<<elapsed<<",\"events\":"<<events.event_count()<<",\"frames\":"<<images.count()
          <<",\"adaptive_optics\":"<<(optimized?"true":"false")<<",\"optical_pixel_queries\":"<<optical_queries
          <<",\"exact_fallback_pixel_queries\":"<<fallback_queries<<",\"fallback_pixel_windows\":"<<fallback_pixel_windows<<",\"convergence_checks\":"<<convergence_checks
          <<",\"convergence_rejections\":"<<convergence_rejections<<",\"current_batch_intervals\":"<<batch<<"}";
        budget.write_text(out/"render_progress.json",progress.str());std::cout<<progress.str()<<std::endl;last_report=elapsed;
      }
    }
    if(std::abs((frame_index-1)/double(config.scene.cam_fps)-duration)>1e-9)frame_at(duration);
    sensor.drain(ready);double transport_end=ready.empty()?duration:std::max(duration,ready.back().timestamp);
    events.append(std::move(ready));events.close(transport_end);images.close();
    std::ostringstream result;result<<std::setprecision(12)<<"{\"status\":\"rendered_pending_validation\",\"duration_seconds\":60,\"event_transport_end_seconds\":"<<transport_end
      <<",\"events\":"<<events.event_count()<<",\"frames\":"<<images.count()<<",\"readout_dropped_events\":"<<sensor.dropped_events()
      <<",\"peak_readout_queue\":"<<sensor.peak_queue()<<",\"numerical_model\":\""<<(optimized?"adaptive_optics_native_circuit_cells":"native_rays_and_circuit_cells")
      <<"\",\"optical_pixel_queries\":"<<optical_queries<<",\"exact_fallback_pixel_queries\":"<<fallback_queries
      <<",\"fallback_pixel_windows\":"<<fallback_pixel_windows<<",\"windows\":"<<windows<<",\"maximum_knots\":"<<maximum_knots
      <<",\"convergence_checks\":"<<convergence_checks<<",\"convergence_rejections\":"<<convergence_rejections
      <<",\"native_guard_windows\":"<<native_guard_windows<<",\"maximum_probed_mean_current_receptor_log_error\":"<<maximum_receptor_error
      <<",\"maximum_probed_p99_mean_current_receptor_log_error\":"<<maximum_p99_receptor_error<<"}";
    budget.write_text(out/"render_progress.json",result.str());
    std::cout<<result.str()<<'\n';return 0;
  }catch(const dvs::DatasetSizeLimit&e){std::cerr<<"STOP_SIZE_LIMIT: "<<e.what()<<'\n';return 42;}
   catch(const std::exception&e){std::cerr<<"FAILED: "<<e.what()<<'\n';return 1;}
}
