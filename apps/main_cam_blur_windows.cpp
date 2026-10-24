#include <cmath>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bvh/build.h"
#include "dvs/blenderize.hpp"
#include "dvs/raycast.hpp"
#include "dvs/repeated_trajectory.hpp"
#include "dvs/scene_data.hpp"
#include "dvs/scene_types.hpp"
#include "dvs/trajectory.hpp"

int main(int argc,char **argv) {
  try {
    if(argc!=8) {
      std::cerr<<"Usage: cam_blur_windows <sequence_dir> <dem> <albedo_or_empty> "
                  "<output_dir> <center_s> <center_s> <center_s>\n";
      return 2;
    }
    const std::filesystem::path sequence=argv[1],dem=argv[2],output=argv[4];
    const std::string albedo=argv[3];
    std::string error;
    dvs::Config config((sequence/"source_config.yaml").string(),error);
    if(!error.empty())throw std::runtime_error("cannot load source camera config: "+error);
    if(config.scene.blur_resolution<=0 || config.scene.supersample_factor<=0)
      throw std::runtime_error("invalid native blur sampling configuration");

    std::vector<dvs::FrameData<dvs::Vec3,dvs::Mat3>> frames;
    if(!dvs::load_frames_from_text(sequence/"source_trajectory.txt",frames,&error) || frames.size()<2)
      throw std::runtime_error("cannot load source trajectory: "+error);
    dvs::CameraTrajectory source(frames,config.scene);
    dvs::RepeatedTrajectory trajectory(source);

    dvs::SceneBundle<float3,uint3> mesh;
    const bool loaded=albedo.empty()?dvs::load_scene_bundle(dem,mesh,&error)
                                    :dvs::load_scene_bundle(dem,albedo,mesh,&error);
    if(!loaded)throw std::runtime_error("cannot load source terrain/material: "+error);
    dvs::BVHBuildParams params;params.max_leaf_size=64;
    const int root=dvs::BVHBuilder::build(params,mesh.bvh_nodes,mesh.bvh_triangles,mesh.vertices);
    dvs::CamRayCaster camera(mesh,config,root);
    mesh={};
    std::cout<<"SCENE_GPU_READY"<<std::endl;
    std::filesystem::create_directories(output);

    const int width=config.camera.width,height=config.camera.height;
    const std::size_t pixels=std::size_t(width)*height;
    const double exposure=0.050;
    const double sample_step=config.scene.blur_resolution;
    for(int i=5;i<8;++i) {
      const double center=std::stod(argv[i]);
      const double begin=center-exposure*.5,end=center+exposure*.5;
      if(!std::isfinite(center) || begin<0 || end>60.0)
        throw std::runtime_error("50 ms exposure lies outside the prepared 60 s trajectory");

      const std::size_t cells=static_cast<std::size_t>(std::ceil(exposure/sample_step-1e-10));
      std::vector<double> accumulated(pixels,0.0);
      std::vector<float> radiance;
      for(std::size_t cell=0;cell<cells;++cell) {
        const double left=begin+cell*sample_step;
        const double right=std::min(end,begin+(cell+1)*sample_step);
        if(right<=left)continue;
        const auto state=trajectory.evaluate((left+right)*.5);
        camera.render_main(dvs::vec3_to_float3(state.position),state.orientation.data(),
                           dvs::vec3_to_float3(state.light_direction),state.light_intensity,radiance);
        if(radiance.size()!=pixels)throw std::runtime_error("camera returned an invalid radiance frame");
        const double weight=(right-left)/exposure;
        for(std::size_t p=0;p<pixels;++p)accumulated[p]+=weight*radiance[p];
        if((cell+1)%10==0 || cell+1==cells)
          std::cout<<"SAMPLE center_s="<<center<<" sample="<<(cell+1)<<"/"<<cells<<std::endl;
      }
      std::vector<float> averaged(pixels);
      for(std::size_t p=0;p<pixels;++p)averaged[p]=static_cast<float>(accumulated[p]);
      std::vector<uint8_t> bytes;
      dvs::render_float_to_png_bytes(averaged.data(),width,height,0,bytes);
      std::ostringstream name;name<<std::fixed<<std::setprecision(3)<<"motion_blur_"
                                  <<begin<<"_"<<end<<"s.png";
      const auto path=output/name.str();
      cv::Mat image(height,width,CV_8U,bytes.data());
      if(!cv::imwrite(path.string(),image))throw std::runtime_error("cannot write "+path.string());
      std::cout<<"WINDOW center_s="<<center<<" begin_s="<<begin<<" end_s="<<end
               <<" temporal_samples="<<cells<<" step_s="<<sample_step
               <<" output="<<path<<'\n';
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr<<"Error: "<<e.what()<<'\n';
    return 1;
  }
}
