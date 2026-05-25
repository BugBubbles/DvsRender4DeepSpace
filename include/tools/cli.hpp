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
  std::string config_path = "./config/default.yaml";

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

      if (arg == "--scene_dir") {
        scene_dir = require_value("--scene_dir");
      } else if (arg == "--dem") {
        dem_path = require_value("--dem");
      } else if (arg == "--albedo") {
        albedo_path = require_value("--albedo");
      } else if (arg == "--traj") {
        traj_path = require_value("--traj");
      } else if (arg == "--output") {
        output_dir = require_value("--output");
      } else if (arg == "--config") {
        config_path = require_value("--config");
      } else if (arg == "--help" || arg == "-h") {
        print_usage(argv[0]);
        std::exit(0);
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }

    if (scene_dir.empty()) {
      if (dem_path.empty() || traj_path.empty()) {
        throw std::runtime_error("either --scene_dir or all --dem, --albedo "
                                 "and --traj are required");
      }
    }
    std::ifstream config_in(config_path);
    if (!config_in.good()) {
      throw std::runtime_error("failed to open config file: " + config_path);
    }
  }
};

} // namespace dvs