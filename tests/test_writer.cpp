#include "dvs/dvs_writer.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <cstring>
using namespace dvs;
void check(bool yes,const char *msg){if(!yes)throw std::runtime_error(msg);}
int main(){try{
 auto dir=std::filesystem::temp_directory_path()/"dvs_writer_test";std::filesystem::create_directories(dir);
 auto bin=(dir/"events.bin").string(),h5=(dir/"events.h5").string();
 {BinaryEventStreamWriter w(bin);w.append({{.001234,2,3,1,10},{.0021,4,5,0,20}});w.close(.003);}
 check(std::filesystem::file_size(bin)==26,"binary compatibility");
 {Hdf5EventStreamWriter w(h5);w.append({{.001234,2,3,1,10},{.0021,4,5,0,20}});w.append({{50.000001,2,3,1,10},{50.000002,2,3,1,10}});w.close(50.001);}
 hid_t f=H5Fopen(h5.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT), t=H5Dopen2(f,"t",H5P_DEFAULT),idx=H5Dopen2(f,"ms_to_idx",H5P_DEFAULT);
 int64_t ts[4];H5Dread(t,H5T_NATIVE_INT64,H5S_ALL,H5S_ALL,H5P_DEFAULT,ts);
 check(ts[0]==1234&&ts[1]==2100&&ts[2]==50000001&&ts[3]==50000002,"microsecond precision/units");
 hid_t space=H5Dget_space(idx);hsize_t dims[1];H5Sget_simple_extent_dims(space,dims,nullptr);std::vector<int64_t> indices(dims[0]);H5Dread(idx,H5T_NATIVE_INT64,H5S_ALL,H5S_ALL,H5P_DEFAULT,indices.data());
 check(indices[0]==0&&indices[1]==0&&indices[2]==1&&indices[3]==2&&indices[50000]==2&&indices[50001]==4,"ms index boundaries");
 H5Sclose(space);H5Dclose(t);H5Dclose(idx);H5Fclose(f);std::filesystem::remove_all(dir);
 std::cout<<"PASS: binary compatibility, HDF5 microseconds and millisecond indexing\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
