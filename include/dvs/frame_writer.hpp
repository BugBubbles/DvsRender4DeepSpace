#pragma once
#include "dvs/dataset_budget.hpp"
#include "dvs/blenderize.hpp"
#include <hdf5.h>
#include <cstdint>
#include <limits>
#include <vector>

namespace dvs {
// Lossless storage. The image is an instantaneous fixed-exposure sRGB preview;
// depth is mean hit-ray range (not optical-axis Z), -1 where no ray hits.
class Hdf5FrameWriter {
public:
  Hdf5FrameWriter(const std::filesystem::path &path,int width,int height,const DatasetBudget &budget)
    :width_(width),height_(height),budget_(budget) {
    if(width<=0 || height<=0)throw std::invalid_argument("invalid frame dimensions");
    budget.require_path(path);
    budget.write(4*1024*1024,[&]{
      file_=H5Fcreate(path.c_str(),H5F_ACC_EXCL,H5P_DEFAULT,H5P_DEFAULT);check(file_,"create frames file");
      times_=create("t",H5T_STD_I64LE,1);
      images_=create("images",H5T_STD_U8LE,3);
      depths_=create("depth",H5T_IEEE_F32LE,3);
      check(H5Fflush(file_,H5F_SCOPE_GLOBAL),"flush frames header");
    });
  }
  ~Hdf5FrameWriter(){release();}
  void append(double time,const std::vector<float> &radiance,const std::vector<float>& depth) {
    auto pixels=std::size_t(width_)*height_;
    if(closed_ || !std::isfinite(time) || time<=last_time_ || radiance.size()!=pixels || depth.size()!=pixels)
      throw std::invalid_argument("invalid frame timestamp or array size");
    for(std::size_t i=0;i<pixels;++i)
      if(!std::isfinite(radiance[i]) || !std::isfinite(depth[i]) || (depth[i]<0 && depth[i]!=-1))
        throw std::invalid_argument("invalid radiance or depth value");
    std::vector<uint8_t> image;render_float_to_png_bytes(radiance.data(),width_,height_,0,image);
    int64_t timestamp=std::llround(time*1e6);
    budget_.write(pixels*5+4*1024*1024,[&]{
      put(times_,H5T_NATIVE_INT64,&timestamp,1);
      put(images_,H5T_NATIVE_UINT8,image.data(),3);
      put(depths_,H5T_NATIVE_FLOAT,depth.data(),3);
      check(H5Fflush(file_,H5F_SCOPE_GLOBAL),"flush frame");
    });
    ++count_;last_time_=time;
  }
  void close(){if(closed_)return;budget_.write(4*1024*1024,[&]{
      check(H5Fflush(file_,H5F_SCOPE_GLOBAL),"flush final frame");release();});closed_=true;}
  std::size_t count()const{return count_;}
private:
  static void check(int64_t code,const char*message){if(code<0)throw std::runtime_error(message);}
  hid_t create(const char *name,hid_t type,int rank) {
    hsize_t dimensions[3]={0,hsize_t(height_),hsize_t(width_)};
    hsize_t maximum[3]={H5S_UNLIMITED,hsize_t(height_),hsize_t(width_)};
    hsize_t chunks[3]={rank==1?4096ULL:1ULL,hsize_t(height_),hsize_t(width_)};
    hid_t space=H5Screate_simple(rank,dimensions,maximum),plist=H5Pcreate(H5P_DATASET_CREATE);
    check(space,"create frame dataspace");check(plist,"create frame properties");
    check(H5Pset_chunk(plist,rank,chunks),"set frame chunks");
    if(rank==3 && H5Zfilter_avail(H5Z_FILTER_DEFLATE)>0){
      check(H5Pset_shuffle(plist),"set lossless shuffle");check(H5Pset_deflate(plist,1),"set lossless deflate");
    }
    hid_t data=H5Dcreate2(file_,name,type,space,H5P_DEFAULT,plist,H5P_DEFAULT);
    H5Pclose(plist);H5Sclose(space);check(data,"create frame dataset");return data;
  }
  void put(hid_t dataset,hid_t type,const void *data,int rank) {
    hsize_t size[3]={count_+1,hsize_t(height_),hsize_t(width_)};
    hsize_t start[3]={count_,0,0},block[3]={1,hsize_t(height_),hsize_t(width_)};
    check(H5Dset_extent(dataset,size),"extend frame dataset");
    hid_t space=H5Dget_space(dataset);check(space,"get frame dataspace");
    check(H5Sselect_hyperslab(space,H5S_SELECT_SET,start,nullptr,block,nullptr),"select frame hyperslab");
    hid_t memory=H5Screate_simple(rank,block,nullptr);check(memory,"create frame memory space");
    auto status=H5Dwrite(dataset,type,memory,space,H5P_DEFAULT,data);
    H5Sclose(memory);H5Sclose(space);check(status,"write frame dataset");
  }
  void release(){for(auto p:{&times_,&images_,&depths_})if(*p>=0){H5Dclose(*p);*p=-1;}
    if(file_>=0){H5Fclose(file_);file_=-1;}}
  int width_,height_;const DatasetBudget &budget_;
  hid_t file_=-1,times_=-1,images_=-1,depths_=-1;
  std::size_t count_=0;double last_time_=-std::numeric_limits<double>::infinity();bool closed_=false;
};
} // namespace dvs
