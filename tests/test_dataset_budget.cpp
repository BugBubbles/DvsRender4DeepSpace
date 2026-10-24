#include "dvs/dataset_budget.hpp"
#include "dvs/dvs_writer.hpp"
#include <iostream>
#include <sys/wait.h>
using namespace dvs;
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
int main(){try{
  auto root=std::filesystem::temp_directory_path()/("dvs-budget-test-"+std::to_string(getpid()));
  std::filesystem::create_directories(root);
  DatasetBudget budget(root,10000,9000);
  // Three concurrent producers need 12000 bytes. Exactly two may commit.
  pid_t children[3];
  for(int i=0;i<3;++i){
    children[i]=fork();require(children[i]>=0,"fork failed");
    if(children[i]==0){
      try{budget.write_text(root/(std::to_string(i)+".txt"),std::string(4000,'x'));_exit(0);}
      catch(const DatasetSizeLimit&){_exit(17);}
      catch(...){_exit(1);}
    }
  }
  int accepted=0,stopped=0;
  for(auto child:children){int status;waitpid(child,&status,0);require(WIFEXITED(status),"child interrupted");
    accepted+=WEXITSTATUS(status)==0;stopped+=WEXITSTATUS(status)==17;}
  require(accepted==2 && stopped==1,"concurrent writers spent the same storage allowance");
  require(budget.logical_bytes()==8000,"incorrect aggregate accounting");
  bool rejected=false;
  try{budget.require_path(root/"../outside.txt");}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"path outside dataset accepted");
  auto sparse=root/"sparse.bin";
  {std::ofstream f(sparse);f.seekp(4000);f.put('x');}
  require(budget.logical_bytes()==12001,"sparse logical bytes were not counted");
  rejected=false;
  try{budget.write_text(root/"blocked.txt","x");}catch(const DatasetSizeLimit&){rejected=true;}
  require(rejected && !std::filesystem::exists(root/"blocked.txt"),"over-limit directory accepted new data");
  std::filesystem::remove_all(root);std::filesystem::create_directories(root);
  // Real HDF5 writes use the same capacity guard. Rejection precedes append.
  DatasetBudget hdf_budget(root,32000000,30000000);
  {Hdf5EventStreamWriter writer((root/"events.h5").string(),&hdf_budget);
   writer.append({{.001234,2,3,1,10}});
   auto before=hdf_budget.logical_bytes();
   auto filler=root/"filler.bin";
   {std::ofstream f(filler);f.seekp(29500000-before-1);f.put('x');}
   auto size_before=std::filesystem::file_size(root/"events.h5");
   rejected=false;
   try{writer.append({{.002,2,3,0,10},{.020,2,3,0,10},{.030,2,3,0,10}});}catch(const DatasetSizeLimit&){rejected=true;}
   require(rejected && writer.event_count()==1,"HDF5 append continued past stop budget");
   require(std::filesystem::file_size(root/"events.h5")==size_before,"rejected HDF5 write changed file");
   std::filesystem::remove(filler);writer.append({{.0021,2,3,0,10}});writer.close(.003);
  }
  hid_t file=H5Fopen((root/"events.h5").c_str(),H5F_ACC_RDONLY,H5P_DEFAULT),idx=H5Dopen2(file,"ms_to_idx",H5P_DEFAULT);
  hid_t space=H5Dget_space(idx);hsize_t shape[1];H5Sget_simple_extent_dims(space,shape,nullptr);
  require(shape[0]==4,"rejected write altered the millisecond index");
  int64_t index[4];H5Dread(idx,H5T_NATIVE_INT64,H5S_ALL,H5S_ALL,H5P_DEFAULT,index);
  require(index[0]==0 && index[1]==0 && index[2]==1 && index[3]==2,"capacity rejection corrupted event index state");
  H5Sclose(space);H5Dclose(idx);H5Fclose(file);
  std::filesystem::remove_all(root);
  std::cout<<"PASS aggregate capacity, concurrent producers, sparse files, path bounds and HDF5 pre-write stop\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
