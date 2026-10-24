#pragma once
#include "dvs/scene_data.hpp"
#include "tools/cli.hpp"
#include <iomanip>
namespace dvs {
inline void load_app_inputs(const tools::Options &opts,const Config &config,
                            SceneBundle<float3,uint3> &scene,std::vector<FrameData<Vec3,Mat3>> &frames) {
  std::string error;
  bool loaded=false;
  if(!opts.dem_path.empty() || !opts.traj_path.empty() || !opts.albedo_path.empty()) {
    if(opts.dem_path.empty() || opts.traj_path.empty())
      throw std::runtime_error("raster input requires both --dem and --traj");
    loaded=opts.albedo_path.empty()?load_scene_bundle(opts.dem_path,opts.traj_path,scene,frames,&error):
                                  load_scene_bundle(opts.dem_path,opts.albedo_path,opts.traj_path,scene,frames,&error);
  } else loaded=load_scene_bundle(opts.scene_dir,scene,frames,&error);
  if(!loaded) throw std::runtime_error("failed to load scene: "+error);
  if(frames.empty()) throw std::runtime_error("empty trajectory");
  frames.resize(std::min(frames.size(),static_cast<size_t>(config.scene.frame_count)));
  std::filesystem::create_directories(opts.output_dir);
  auto source=std::filesystem::absolute(opts.config_path);
  auto destination=std::filesystem::absolute(std::filesystem::path(opts.output_dir)/"source_config.yaml");
  if(source.lexically_normal()!=destination.lexically_normal())
    std::filesystem::copy_file(source,destination,std::filesystem::copy_options::overwrite_existing);
  std::ofstream out(std::filesystem::path(opts.output_dir)/"camera_used.yaml");
  if(!out) throw std::runtime_error("cannot write camera metadata");
  out<<std::setprecision(9)<<"source: "<<source<<"\nwidth: "<<config.camera.width<<"\nheight: "<<config.camera.height
     <<"\nfx: "<<config.camera.fx<<"\nfy: "<<config.camera.fy<<"\ncx: "<<config.camera.cx<<"\ncy: "<<config.camera.cy
     <<"\nsupersample_factor: "<<config.scene.supersample_factor
     <<"\npixel_centers: integer\nimage_axes: x_right_y_down\npose_axes: Blender_x_right_y_up_z_backward"
     <<"\ndepth: mean_hit_ray_range_m; miss=-1"
     <<"\ndistortion: not_implemented"
     <<"\nphysics_calibration: generic_uncalibrated_optical_and_circuit_parameters\n";
}
}
