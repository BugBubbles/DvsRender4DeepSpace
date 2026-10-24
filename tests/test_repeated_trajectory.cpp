#include "dvs/repeated_trajectory.hpp"
#include "dvs/scene_data.hpp"
#include <filesystem>
#include <iostream>
using namespace dvs;
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
int main(){try{
  std::string error;Config config(std::string(DVS_SOURCE_DIR)+"/config/davis346.yaml",error);
  require(error.empty(),error.c_str());
  int tested=0;
  auto assets=std::filesystem::path(DVS_SOURCE_DIR).parent_path()/"assets";
  for(auto &e:std::filesystem::recursive_directory_iterator(assets)){
    if(e.path().extension()!=".txt" || e.path().filename().string().find("trajectory")!=0)continue;
    std::vector<FrameData<Vec3,Mat3>> frames;
    require(load_frames_from_text(e.path(),frames,&error),error.c_str());
    CameraTrajectory source(frames,config.scene);RepeatedTrajectory repeat(source);
    double d=repeat.source_duration();
    for(double t:{1.0,d/2,d-1}){
      auto a=source.evaluate(source.start_time()+t),b=repeat.evaluate(t),r=repeat.evaluate(2*d-t);
      require((a.position-b.position).norm()<.05 && (a.position-r.position).norm()<.05,"original path changed");
      require((a.linear_velocity-b.linear_velocity).norm()<.01,"motion slowed outside reversal");
      require((a.linear_velocity+r.linear_velocity).norm()<.01,"reverse velocity incorrect");
      require((a.orientation-b.orientation).norm()<1e-5,"original pose changed");
    }
    for(double t:{0.0,d,2*d}){
      auto at=repeat.evaluate(t),left=repeat.evaluate(t-1e-4),right=repeat.evaluate(t+1e-4);
      auto endpoint=source.evaluate(t==d?source.end_time():source.start_time());
      require((at.position-endpoint.position).norm()<.01,"source endpoint not reached");
      require(at.linear_velocity.norm()<1e-3 && at.angular_velocity.norm()<1e-3,"nonzero reversal velocity");
      require((left.position-right.position).norm()<.05,"teleport at reversal");
      require((left.orientation-right.orientation).norm()<1e-4,"pose jump at reversal");
    }
    for(int i=0;i<=12000;++i){auto s=repeat.evaluate(i/200.0);
      require(s.position.allFinite() && s.orientation.allFinite() && s.linear_velocity.allFinite(),"invalid 60 s pose");
      require(std::abs(s.orientation.determinant()-1)<1e-4,"pose is not a proper rotation");}
    ++tested;
  }
  require(tested==5,"not all five supplied trajectories were checked");
  std::cout<<"PASS five original-speed 60 s trajectories, endpoint and velocity continuity\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
