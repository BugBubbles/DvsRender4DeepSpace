#include "app_inputs.hpp"
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "bvh/build.h"
#include "dvs/dvs_base.hpp"
#include "dvs/dvs_writer.hpp"
#include "dvs/progress.hpp"
#include "dvs/raycast.hpp"
#include "dvs/scene_data.hpp"
#include "dvs/scene_types.hpp"
#include "dvs/trajectory.hpp"
#include "tools/cli.hpp"

namespace dvs {
std::unique_ptr<dvs::EventStreamWriter>
make_event_writer(const tools::Options &opts) {
  if (opts.event_format == "hdf5") {
    return std::make_unique<dvs::Hdf5EventStreamWriter>(opts.output_dir +
                                                        "/events.h5");
  } else if (opts.event_format == "bin") {
    return std::make_unique<dvs::BinaryEventStreamWriter>(opts.output_dir +
                                                          "/events.bin");
  } else {
    throw std::runtime_error("unsupported event format: " + opts.event_format);
  }
}
} // namespace dvs

int main(int argc, char **argv) {
  try {
    if (argc <= 1) {
      tools::Options::print_usage(argv[0]);
      return 1;
    }

    const tools::Options opts(argc, argv);
    dvs::SceneBundle<float3, uint3> scene;
    std::vector<dvs::FrameData<dvs::Vec3, dvs::Mat3>> frames;
    std::string error_message;
    dvs::Config config(opts.config_path, error_message);
    if (!error_message.empty()) {
      throw std::runtime_error("failed to load config: " + error_message);
      return 1;
    }

    bool adaptive=opts.event_solver=="adaptive-optics" || (opts.event_solver=="auto" && opts.event_format=="hdf5" && config.physic.refractory_time>0);
#ifndef ENABLE_PHYSIC_EFFECTS
    if(opts.event_solver=="adaptive-optics")throw std::runtime_error("adaptive optics requires enabled physical reset effects");
    adaptive=false;
#endif
    if(adaptive && opts.event_format!="hdf5")throw std::runtime_error("adaptive optics exports HDF5 x/y/t/p; per-event auxiliary binary depth requires --solver native");
    dvs::load_app_inputs(opts,config,scene,frames);

    // 使用样条曲线获取连续位姿轨迹
    dvs::CameraTrajectory traj(frames, config.scene);
    dvs::BVHBuildParams params;params.max_leaf_size=64; // full supplied terrain fits GPU memory
    int root_index = dvs::BVHBuilder::build(
        params, scene.bvh_nodes, scene.bvh_triangles, scene.vertices);

    dvs::DvsRayCaster raycaster(scene, config, root_index);
    std::filesystem::path output_dir(opts.output_dir);
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }
    std::unique_ptr<dvs::EventStreamWriter> events_out =
        dvs::make_event_writer(opts);

    size_t frame_count =
        std::min(frames.size(), static_cast<size_t>(config.scene.frame_count));
    if(frame_count==0) throw std::runtime_error("no trajectory frames selected");
    float end_time = dvs::frame_time(frames[frame_count - 1], config.scene.fps);
    float start_time =
        dvs::frame_time(frames.front(), config.scene.fps);
    std::size_t total_steps = static_cast<std::size_t>(
        std::ceil((static_cast<double>(end_time) - static_cast<double>(start_time)) /
                  static_cast<double>(config.scene.time_resolution)));

    dvs::ProgressBar bar(total_steps, "Generating events");
    std::vector<dvs::Event> events, events_buffer; // 用于存储当前时间步的事件
                                                   //
    double next_write_time = start_time;
    const float write_interval = 1.0f / config.scene.imu_fps;

    auto previous_state=traj.evaluate(start_time);
    raycaster.initialize(previous_state,start_time);
    std::cout << "Initialized continuous photocurrent/circuit state from scene rays.\n";
    double previous_time=start_time;
    unsigned batch=adaptive?200:64;
    if(config.physic.refractory_time<=0)batch=1;
    else batch=std::max(1U,std::min(batch,unsigned(std::max(1.0,(config.physic.max_events_per_step-2)*double(config.physic.refractory_time)/config.scene.time_resolution))));
    double next_check=start_time;size_t step=0;
    std::vector<dvs::OpticalInterval> intervals;intervals.reserve(batch);
    while(step<total_steps){
      size_t first_step=step;double first_time=previous_time;auto first_state=previous_state;intervals.clear();
      for(unsigned j=0;j<batch && step<total_steps;++j){
        double time=std::min(double(end_time),double(start_time)+(++step)*double(config.scene.time_resolution));
        if(time<=previous_time)break;auto current=traj.evaluate(time);
        intervals.push_back({dvs::to_optical_state(previous_state),dvs::to_optical_state(current),previous_time,time});
        previous_state=current;previous_time=time;
      }
      if(intervals.empty())break;
      if(adaptive){bool check=first_time>=next_check;
        if(!raycaster.advance_adaptive(intervals,dvs::kDefaultOpticalLogTolerance,events,check)){
          if(batch>1){batch=std::max(1U,batch/2);step=first_step;previous_time=first_time;previous_state=first_state;continue;}
          raycaster.advance_many(intervals,events);
        }
        if(check)next_check=previous_time+.25;
      }else raycaster.advance_many(intervals,events);
      std::move(events.begin(),events.end(),std::back_inserter(events_buffer));events.clear();
      if(previous_time>=next_write_time){next_write_time=previous_time+write_interval;events_out->append(std::move(events_buffer));events_buffer.clear();}
      for(size_t j=first_step;j<step;++j)bar.update();
    }
    if (!events_buffer.empty()) {
      events_out->append(std::move(events_buffer));
      events_buffer.clear();
    }
    // Finish the last interval and flush the transport backlog without losing events.
    raycaster.drain(events);
    double output_end = events.empty() ? end_time : std::max(static_cast<double>(end_time), static_cast<double>(events.back().timestamp));
    events_out->append(std::move(events));
    events_out->close(output_end);
    std::ifstream source_config(opts.config_path);
    std::ofstream copied_config(opts.output_dir + "/sensor_config.yaml");
    copied_config << source_config.rdbuf();
    std::ofstream model_metadata(opts.output_dir + "/sensor_model.txt");
    model_metadata << std::setprecision(12)<<"model=ray_photocurrent_circuit_v2\n"
                   << "timestamp="<<(opts.event_format=="hdf5"?"int64_microseconds_transport_completion":"float32_seconds_transport_completion")<<"\n"
                   << "config=" << opts.config_path << "\n"
                   << "random_seed=" << config.physic.random_seed << "\n"
                   << "dropped_readout_events=" << raycaster.dropped_events() << "\n"
                   << "peak_readout_queue=" << raycaster.peak_queue() << "\n";
#ifdef ENABLE_PHYSIC_EFFECTS
    model_metadata << "physical_effects=on\n";
#else
    model_metadata << "physical_effects=off\n";
#endif
    model_metadata << "photoreceptor=nonlinear_current_balance\n"
                   << "optical_sampling="<<(adaptive?"adaptive_direct_ray_nodes_with_native_cell_fallback":"midpoint_rays_within_circuit_cells")<<"\n"
                   << "optical_log_tolerance="<<(adaptive?dvs::kDefaultOpticalLogTolerance:0)<<"\n"
                   << "leaf_voltage_error_threshold_fraction="<<(adaptive?dvs::kOpticalVoltageThresholdFraction:0)<<"\n"
                   << "optical_cache_capacity="<<(adaptive?raycaster.adaptive_cache_capacity():0)<<"\n"
                   << "camera_width="<<config.camera.width<<"\ncamera_height="<<config.camera.height<<"\n"
                   << "fx="<<config.camera.fx<<"\nfy="<<config.camera.fy<<"\ncx="<<config.camera.cx<<"\ncy="<<config.camera.cy<<"\n"
                   << "spatial_grid="<<config.scene.supersample_factor<<"\n"
                   << "distortion=ideal_pinhole_not_hardware_calibration\n"
                   << "physics_calibration=generic_uncalibrated\n"
                   << "photon_noise=charge_count_tau_leap\n"
                   << "circuit_step_seconds=" << config.physic.circuit_step << "\n"
                   << "radiance_scale=" << config.physic.radiance_scale << "\n";
#ifdef ENABLE_AER_DELAY
    model_metadata << "readout_queue=on\n";
#else
    model_metadata << "readout_queue=off\n";
#endif
    std::cout << "Readout dropped " << raycaster.dropped_events() << " events; peak queue " << raycaster.peak_queue() << "\n";

    std::cout << "Finished. Wrote " << events_out->event_count()
              << " events to " << events_out->path() << "\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
