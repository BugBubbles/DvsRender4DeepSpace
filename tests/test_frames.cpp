#include "dvs/frame_writer.hpp"
#include "dvs/scene_types.hpp"
#include <iostream>
using namespace dvs;
void require(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
int main(){try{
  auto root=std::filesystem::temp_directory_path()/("dvs-frames-test-"+std::to_string(getpid()));
  std::filesystem::create_directories(root);DatasetBudget budget(root);
  for(auto device:{"davis346","prophesee"}){
    std::string error;Config c(std::string(DVS_SOURCE_DIR)+"/config/"+device+".yaml",error);require(error.empty(),error.c_str());
    auto path=root/(std::string(device)+".h5");auto pixels=std::size_t(c.camera.width)*c.camera.height;
    std::vector<float> light(pixels,5),depth(pixels,25);depth[0]=-1;
    {Hdf5FrameWriter writer(path,c.camera.width,c.camera.height,budget);writer.append(0,light,depth);writer.append(60,light,depth);
      bool rejected=false;try{writer.append(59,light,depth);}catch(const std::invalid_argument&){rejected=true;}
      require(rejected && writer.count()==2,"unordered frame append was accepted");writer.close();}
    hid_t f=H5Fopen(path.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),d=H5Dopen2(f,"depth",H5P_DEFAULT),t=H5Dopen2(f,"t",H5P_DEFAULT),im=H5Dopen2(f,"images",H5P_DEFAULT);
    hid_t space=H5Dget_space(d);hsize_t size[3];H5Sget_simple_extent_dims(space,size,nullptr);
    require(size[0]==2 && size[1]==hsize_t(c.camera.height) && size[2]==hsize_t(c.camera.width),"native image axes or dimensions changed");
    int64_t times[2];H5Dread(t,H5T_NATIVE_INT64,H5S_ALL,H5S_ALL,H5P_DEFAULT,times);require(times[0]==0 && times[1]==60000000,"frame timestamp precision/units wrong");
    std::vector<float> read_depth(2*pixels);H5Dread(d,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT,read_depth.data());
    require(read_depth[0]==-1 && read_depth[pixels]==-1 && read_depth[1]==25,"lossless depth storage failed");
    std::vector<uint8_t> read_image(2*pixels);H5Dread(im,H5T_NATIVE_UINT8,H5S_ALL,H5S_ALL,H5P_DEFAULT,read_image.data());
    auto expected=to_byte(linear_to_srgb(display_exposure(5)));require(read_image[0]==expected,"fixed image encoding changed");
    H5Sclose(space);H5Dclose(d);H5Dclose(t);H5Dclose(im);H5Fclose(f);
  }
  std::filesystem::remove_all(root);std::cout<<"PASS both native camera frame dimensions, times, image encoding and lossless depth\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
