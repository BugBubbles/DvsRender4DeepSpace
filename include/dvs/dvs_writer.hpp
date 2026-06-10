#include "dvs/dvs_base.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <hdf5.h>
#include <limits>
#include <utility>
#include <vector>

namespace dvs {

class EventStreamWriter {
public:
  virtual ~EventStreamWriter() = default;
  virtual void append(std::vector<Event> events) = 0;
  virtual void close(double end_time_seconds) = 0;
  virtual const std::string &path() const = 0;
  std::size_t event_count() const { return event_count_; }

protected:
  std::size_t event_count_ = 0;
};

class BinaryEventStreamWriter final : public EventStreamWriter {
public:
  explicit BinaryEventStreamWriter(std::string output_path)
      : output_path_(std::move(output_path)),
        out_(output_path_, std::ios::binary) {
    if (!out_) {
      throw std::runtime_error("failed to open output file: " + output_path_);
    }
  }

  void append(std::vector<dvs::Event> events) override {
    if (events.empty()) {
      return;
    }
    out_.write(reinterpret_cast<const char *>(events.data()),
               static_cast<std::streamsize>(events.size() * sizeof(Event)));
    if (!out_) {
      throw std::runtime_error("failed to write output file: " + output_path_);
    }
    event_count_ += events.size();
  }

  void close(double) override { out_.close(); }
  const std::string &path() const override { return output_path_; }

private:
  std::string output_path_;
  std::ofstream out_;
};

class Hdf5EventStreamWriter final : public EventStreamWriter {
public:
  explicit Hdf5EventStreamWriter(std::string output_path)
      : output_path_(std::move(output_path)) {
    file_ = H5Fcreate(output_path_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                      H5P_DEFAULT);
    check(file_, "failed to create HDF5 file: " + output_path_);
    x_dataset_ = create_extendable_dataset("x", H5T_NATIVE_INT16);
    y_dataset_ = create_extendable_dataset("y", H5T_NATIVE_INT16);
    t_dataset_ = create_extendable_dataset("t", H5T_NATIVE_INT64);
    p_dataset_ = create_extendable_dataset("p", H5T_NATIVE_INT8);
  }

  ~Hdf5EventStreamWriter() override {
    close_dataset(x_dataset_);
    close_dataset(y_dataset_);
    close_dataset(t_dataset_);
    close_dataset(p_dataset_);
    if (file_ >= 0) {
      H5Fclose(file_);
      file_ = -1;
    }
  }

  void append(std::vector<Event> events) override {
    if (events.empty()) {
      return;
    }
    std::stable_sort(events.begin(), events.end(),
                     [](const Event &lhs, const Event &rhs) {
                       return lhs.timestamp < rhs.timestamp;
                     });

    std::vector<int16_t> xs(events.size());
    std::vector<int16_t> ys(events.size());
    std::vector<int64_t> ts(events.size());
    std::vector<int8_t> ps(events.size());
    for (std::size_t i = 0; i < events.size(); ++i) {
      xs[i] = events[i].x;
      ys[i] = events[i].y;
      ts[i] = std::llround(events[i].timestamp * 1e9);
      ps[i] = events[i].polarity;

      fill_ms_to_idx(ts[i] / 1000000, event_count_ + i);
    }

    append_dataset(x_dataset_, xs.data(), H5T_NATIVE_INT16, xs.size());
    append_dataset(y_dataset_, ys.data(), H5T_NATIVE_INT16, ys.size());
    append_dataset(t_dataset_, ts.data(), H5T_NATIVE_INT64, ts.size());
    append_dataset(p_dataset_, ps.data(), H5T_NATIVE_INT8, ps.size());
    event_count_ += events.size();
  }

  void close(double end_ts_s) override {
    if (closed_) {
      return;
    }

    if (end_ts_s >= 0) {
      fill_ms_to_idx(std::llround(end_ts_s * 1000), event_count_);
    }
    write_ms_to_idx_dataset();
    H5Fflush(file_, H5F_SCOPE_GLOBAL);
    closed_ = true;
  }

  const std::string &path() const override { return output_path_; }

private:
  static void check(herr_t status, const std::string &message) {
    if (status < 0) {
      throw std::runtime_error(message);
    }
  }

  static void check(hid_t id, const std::string &message) {
    if (id < 0) {
      throw std::runtime_error(message);
    }
  }

  hid_t create_extendable_dataset(const char *name, hid_t type) {
    const hsize_t dims[1] = {0};
    const hsize_t max_dims[1] = {H5S_UNLIMITED};
    hid_t space = H5Screate_simple(1, dims, max_dims);
    check(space, std::string("failed to create HDF5 dataspace for ") + name);
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    check(plist,
          std::string("failed to create HDF5 property list for ") + name);
    const hsize_t chunk_dims[1] = {chunk_size_};
    check(H5Pset_chunk(plist, 1, chunk_dims),
          std::string("failed to set HDF5 chunking for ") + name);
    hid_t dataset =
        H5Dcreate2(file_, name, type, space, H5P_DEFAULT, plist, H5P_DEFAULT);
    H5Pclose(plist);
    H5Sclose(space);
    check(dataset, std::string("failed to create HDF5 dataset ") + name);
    return dataset;
  }

  void append_dataset(hid_t dataset, const void *data, hid_t type,
                      std::size_t count) {
    const hsize_t old_dims[1] = {static_cast<hsize_t>(event_count_)};
    const hsize_t new_dims[1] = {static_cast<hsize_t>(event_count_ + count)};
    check(H5Dset_extent(dataset, new_dims),
          "failed to extend HDF5 event dataset");

    hid_t file_space = H5Dget_space(dataset);
    check(file_space, "failed to open HDF5 event dataspace");
    const hsize_t block[1] = {static_cast<hsize_t>(count)};
    check(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, old_dims, nullptr,
                              block, nullptr),
          "failed to select HDF5 event hyperslab");

    hid_t mem_space = H5Screate_simple(1, block, nullptr);
    check(mem_space, "failed to create HDF5 memory dataspace");
    check(H5Dwrite(dataset, type, mem_space, file_space, H5P_DEFAULT, data),
          "failed to write HDF5 event dataset");
    H5Sclose(mem_space);
    H5Sclose(file_space);
  }

  void fill_ms_to_idx(int64_t target_ms, std::size_t event_index) {
    if (target_ms < 0) {
      return;
    }
    if (target_ms > static_cast<int64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("ms_to_idx would be unreasonably large");
    }
    while (last_ms_ < target_ms) {
      ms_to_idx_.push_back(static_cast<int64_t>(event_index));
      ++last_ms_;
    }
  }

  void write_ms_to_idx_dataset() {
    const hsize_t dims[1] = {static_cast<hsize_t>(ms_to_idx_.size())};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    check(space, "failed to create HDF5 ms_to_idx dataspace");
    hid_t dataset = H5Dcreate2(file_, "ms_to_idx", H5T_NATIVE_INT64, space,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check(dataset, "failed to create HDF5 dataset ms_to_idx");
    if (!ms_to_idx_.empty()) {
      check(H5Dwrite(dataset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     ms_to_idx_.data()),
            "failed to write HDF5 dataset ms_to_idx");
    }
    H5Dclose(dataset);
    H5Sclose(space);
  }

  static void close_dataset(hid_t &dataset) {
    if (dataset >= 0) {
      H5Dclose(dataset);
      dataset = -1;
    }
  }

  static constexpr hsize_t chunk_size_ = 1 << 20;
  std::string output_path_;
  hid_t file_ = -1;
  hid_t x_dataset_ = -1;
  hid_t y_dataset_ = -1;
  hid_t t_dataset_ = -1;
  hid_t p_dataset_ = -1;
  std::vector<int64_t> ms_to_idx_;
  int64_t last_ms_ = -1;
  bool closed_ = false;
};
} // namespace dvs