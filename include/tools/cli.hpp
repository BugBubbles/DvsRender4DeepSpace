#pragma once
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace tools {

class Options {
public:
  std::string scene_dir, dem_path, albedo_path, traj_path;
  std::string output_dir = "./output";
  std::string config_path = "./config/davis346.yaml";
  std::string event_format = "bin";
  std::string event_solver = "auto";

  static void print_usage(const char *prog) {
    std::cout
        << "Usage: " << prog << "[options]\n"
        << "Options:\n"
        << "  --scene_dir <path>         path to scene bundle directory\n"
        << "  --dem <path>               path to DEM file (optional)\n"
        << "  --albedo <path>            path to albedo file (optional)\n"
        << "  --traj <path>              path to camera trajectory file "
           "(optional)\n"
        << "  --output <path>            output images file (default: ./)\n"
        << "  --config <path>            path to YAML config file "
        << "(default: ./config/davis346.yaml)\n"
        << "  --event_format <bin|hdf5>  event stream output format "
        << "(default: bin)\n"
        << "  --solver <auto|native|adaptive-optics>  auto selects adaptive optics for physical HDF5 output"
        << std::endl;
  }

  float parse_float_arg(const char *key, const std::string &value) {
    try {
      return std::stof(value);
    } catch (const std::exception &) {
      throw std::runtime_error(std::string("invalid float for ") + key + ": " +
                               value);
    }
  }
  Options() = default;

  Options(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto require_value = [&](const char *key) -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error(std::string("missing value for ") + key);
        }
        return std::string(argv[++i]);
      };

      if (arg == "--scene_dir" || arg == "--scene-dir") {
        scene_dir = require_value(arg.c_str());
      } else if (arg == "--dem") {
        dem_path = require_value(arg.c_str());
      } else if (arg == "--albedo") {
        albedo_path = require_value(arg.c_str());
      } else if (arg == "--traj") {
        traj_path = require_value(arg.c_str());
      } else if (arg == "--output") {
        output_dir = require_value(arg.c_str());
      } else if (arg == "--config") {
        config_path = require_value(arg.c_str());
      } else if (arg == "--event_format" || arg == "--event-format" ||
                 arg == "--events_format" || arg == "--events-format") {
        event_format = require_value(arg.c_str());
        if (event_format == "h5") {
          event_format = "hdf5";
        } else if (event_format == "binary") {
          event_format = "bin";
        }
        if (event_format != "bin" && event_format != "hdf5") {
          throw std::runtime_error(
              "invalid event format for --event_format: " + event_format +
              " (expected bin or hdf5)");
        }
      } else if (arg == "--solver") {
        event_solver=require_value(arg.c_str());
        if(event_solver!="auto" && event_solver!="native" && event_solver!="adaptive-optics")
          throw std::runtime_error("solver must be auto, native, or adaptive-optics");
      } else if (arg == "--help" || arg == "-h") {
        print_usage(argv[0]);
        std::exit(0);
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }

    if (scene_dir.empty()) {
      if (dem_path.empty() || traj_path.empty()) {
        throw std::runtime_error("either --scene_dir or --dem and --traj are required");
      }
    }
    std::ifstream config_in(config_path);
    if (!config_in.good()) {
      throw std::runtime_error("failed to open config file: " + config_path);
    }
  }
};

} // namespace dvs
