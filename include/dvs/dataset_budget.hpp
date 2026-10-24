#pragma once
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dvs {
class DatasetSizeLimit : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// All producers of a batch share the same cross-process lock. The lock is kept
// outside the data directory. Count apparent bytes, which conservatively count
// sparse and compressed files, and leave 20 GB below the user's hard ceiling.
class DatasetBudget {
public:
  explicit DatasetBudget(const std::filesystem::path &root,
                         uint64_t hard=500000000000ULL,
                         uint64_t stop=480000000000ULL)
    :root_(std::filesystem::canonical(root)),hard_(hard),stop_(stop) {
    if(stop_==0 || stop_>hard_ || hard_>500000000000ULL)
      throw std::invalid_argument("invalid dataset capacity limits");
    uint64_t key=14695981039346656037ULL;
    for(unsigned char c:root_.string()){key^=c;key*=1099511628211ULL;}
    std::ostringstream name;name<<"dvs-dataset-"<<std::hex<<key<<".lock";
    lock_path_=std::filesystem::temp_directory_path()/name.str();
  }
  void require_path(const std::filesystem::path &p) const {
    auto relative=std::filesystem::weakly_canonical(p).lexically_relative(root_);
    if(relative.empty() || relative.is_absolute() || *relative.begin()=="..")
      throw std::runtime_error("output is outside the guarded dataset: "+p.string());
  }
  uint64_t logical_bytes() const {
    uint64_t total=0;
    for(const auto &e:std::filesystem::recursive_directory_iterator(root_)) {
      if(e.is_symlink())throw std::runtime_error("dataset symlinks are not supported by capacity accounting");
      if(e.is_regular_file()) {
        auto n=e.file_size();
        if(n>std::numeric_limits<uint64_t>::max()-total)throw DatasetSizeLimit("dataset size overflow");
        total+=n;
      }
    }
    return total;
  }
  const std::filesystem::path &root()const{return root_;}
  uint64_t hard_limit()const{return hard_;}
  uint64_t stop_limit()const{return stop_;}

  // The caller supplies a conservative bound on file growth and must flush its
  // buffers within this operation. An exclusive lock covers check + write +
  // flush, so eight render processes cannot each spend the same free capacity.
  template<class F> void write(uint64_t max_growth,F operation) const {
    Lock lock(lock_path_);
    auto before=logical_bytes();
    if(before>hard_ || before>=stop_ || max_growth>stop_-before) {
      throw DatasetSizeLimit("dataset capacity stop: current="+std::to_string(before)+
        ", reserved_write="+std::to_string(max_growth)+", stop="+std::to_string(stop_)+
        ", hard="+std::to_string(hard_)+" bytes");
    }
    auto available=std::filesystem::space(root_).available;
    if(max_growth>available || available-max_growth<128ULL*1024*1024)
      throw DatasetSizeLimit("insufficient filesystem space for reserved dataset write");
    operation();
    auto after=logical_bytes();
    if(after>before+max_growth)
      throw DatasetSizeLimit("writer exceeded its declared storage reservation; batch must stop");
    if(after>hard_)throw DatasetSizeLimit("dataset hard size limit reached; batch must stop");
  }
  void write_text(const std::filesystem::path &p,const std::string &text)const {
    require_path(p);
    // A temporary sibling preserves the previous valid metadata on interruption.
    write(text.size(),[&]{
      auto partial=p;partial+=".partial";
      std::ofstream out(partial,std::ios::binary|std::ios::trunc);
      if(!out)throw std::runtime_error("cannot open dataset metadata: "+partial.string());
      out.write(text.data(),text.size());out.flush();
      if(!out)throw std::runtime_error("cannot write dataset metadata: "+partial.string());
      out.close();std::filesystem::rename(partial,p);
    });
  }
private:
  class Lock {
  public:
    explicit Lock(const std::filesystem::path &path) {
      fd_=::open(path.c_str(),O_CREAT|O_RDWR|O_CLOEXEC,0600);
      if(fd_<0)throw std::runtime_error("cannot open dataset capacity lock: "+std::string(std::strerror(errno)));
      int result;
      do{result=flock(fd_,LOCK_EX);}while(result<0 && errno==EINTR);
      if(result<0){::close(fd_);fd_=-1;throw std::runtime_error("cannot lock dataset capacity budget");}
    }
    ~Lock(){if(fd_>=0){flock(fd_,LOCK_UN);::close(fd_);}}
    Lock(const Lock&)=delete;Lock&operator=(const Lock&)=delete;
  private:int fd_=-1;
  };
  std::filesystem::path root_,lock_path_;uint64_t hard_,stop_;
};
} // namespace dvs
