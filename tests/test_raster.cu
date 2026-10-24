#include "tools/loader.hpp"
#include "tools/build_from_raster.hpp"
#include "dvs/scene_data.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace tools;
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
template<class T>T *upload(const std::vector<T>&v){T *p=nullptr;CUDA_CHECK(cudaMalloc(&p,v.size()*sizeof(T)),"test allocation");CUDA_CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),"test upload");return p;}
int main(int argc,char**argv){try {
  require(argc==2,"usage: test_raster output");std::filesystem::path output=argv[1];std::filesystem::create_directories(output);
  GDALAllRegister();auto *driver=GetGDALDriverManager()->GetDriverByName("GTiff");require(driver,"GTiff unavailable");
  auto raster=output/"dem.tif";
  auto *ds=driver->Create(raster.c_str(),4,4,1,GDT_Float32,nullptr);require(ds,"create DEM");
  double gt[6]={100,1,0,200,0,-1};ds->SetGeoTransform(gt);
  std::vector<float> data(16);for(int row=0;row<4;++row)for(int col=0;col<4;++col)data[row*4+col]=row+col;
  data[0]=-9999;auto *band=ds->GetRasterBand(1);band->SetNoDataValue(-9999);band->SetScale(2);band->SetOffset(10);
  require(band->RasterIO(GF_Write,0,0,4,4,data.data(),4,4,GDT_Float32,0,0)==CE_None,"write DEM");GDALClose(ds);
  Imagery dem;require(loadImageryData(raster.string(),dem),"load DEM");
  require(std::isnan(dem.data[0])&&dem.data[6]==16&&dem.x_coords[0]==100.5&&dem.y_coords[0]==199.5,"NoData/scale/offset/pixel-center loading");
  std::vector<float3> vertices,normals;std::vector<uint3> triangles;
  require(load_from_raster(raster.string(),vertices,normals,triangles),"raster mesh conversion");
  require(vertices.size()==15&&triangles.size()==16,"NoData created fake geometry");
  for(auto p:vertices)require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"nonfinite centered vertex");
  for(auto p:normals)require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"nonfinite normal");
  for(auto t:triangles)require(t.x<vertices.size()&&t.y<vertices.size()&&t.z<vertices.size(),"bad compact index");
  dem.data[10]=NAN;bool rejected=false;try{reference_height(dem);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"NoData reference origin accepted");
  std::vector<float> x={2,1,0},y={0,1,2},rx={0,.5f,1.5f,2,-1},ry={0,.5f,1.5f,2,3},source(9);
  for(int row=0;row<3;++row)for(int col=0;col<3;++col)source[row*3+col]=2*x[col]+3*y[row];
  auto dx=upload(x),dy=upload(y),drx=upload(rx),dry=upload(ry),dsource=upload(source),result=upload(std::vector<float>(25));
  std::vector<float> aligned(25);
  batch_align_kernel<<<dim3(1,1),dim3(8,8)>>>(nullptr,dry,drx,5,5,dsource,dy,dx,3,3,result);
  CUDA_CHECK(cudaMemcpy(aligned.data(),result,25*sizeof(float),cudaMemcpyDeviceToHost),"alignment copy");
  for(int row=0;row<5;++row)for(int col=0;col<5;++col){
    if(row==4||col==4)require(std::isnan(aligned[row*5+col]),"outside raster became black material");
    else require(std::abs(aligned[row*5+col]-2*rx[col]-3*ry[row])<1e-6,"bilinear interpolation/endpoints/axis direction");
  }
  source[4]=NAN;CUDA_CHECK(cudaMemcpy(dsource,source.data(),9*sizeof(float),cudaMemcpyHostToDevice),"missing material upload");
  batch_align_kernel<<<dim3(1,1),dim3(8,8)>>>(nullptr,dry,drx,5,5,dsource,dy,dx,3,3,result);
  CUDA_CHECK(cudaMemcpy(aligned.data(),result,25*sizeof(float),cudaMemcpyDeviceToHost),"NoData interpolation copy");
  require(std::isnan(aligned[6])&&aligned[0]==0,"interpolation crossed NoData or corrupted valid zero");
  for(auto pointer:{dx,dy,drx,dry,dsource,result})cudaFree(pointer);
  // Binary frame layout must keep fractional seconds and private per-thread matrices.
  std::filesystem::path bundle=output/"bundle";std::filesystem::create_directories(bundle);
  auto write=[&](const char *name,const auto &values){std::ofstream file(bundle/name,std::ios::binary);file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));};
  write("mesh_vertices.bin",vertices);write("mesh_normals.bin",normals);write("mesh_indices.bin",triangles);
  std::vector<float> frames;
  for(int i=0;i<32;++i){dvs::Mat4 transform=dvs::Mat4::Identity();transform.block<3,3>(0,0)=Eigen::AngleAxisf(i*.05f,dvs::Vec3::UnitZ()).toRotationMatrix();
    frames.push_back(.125f+i*.125f);frames.insert(frames.end(),{0,0,10});
    for(int row=0;row<4;++row)for(int col=0;col<4;++col)frames.push_back(transform(row,col));
    frames.insert(frames.end(),{0,0,1,1000});}
  write("frames.bin",frames);
  dvs::SceneBundle<float3,uint3> loaded;std::vector<dvs::FrameData<dvs::Vec3,dvs::Mat3>> poses;std::string error;
  require(dvs::load_scene_bundle(bundle,loaded,poses,&error),error.c_str());require(poses.size()==32,"frame count");
  for(int i=0;i<32;++i){auto expected=Eigen::AngleAxisf(i*.05f,dvs::Vec3::UnitZ()).toRotationMatrix();
    require(poses[i].timestamp_seconds==.125f+i*.125f&&(poses[i].camera_world-expected).norm()<1e-6,"binary seconds or parallel matrix race");}
  std::cout<<"PASS raster NoData/scaling, compact mesh, center validation, CUDA alignment and binary frame seconds/matrices\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
