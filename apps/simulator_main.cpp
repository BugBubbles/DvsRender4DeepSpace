#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "bvh/build.h"
#include "dvs/dvs_base.hpp"
#include "dvs/hapke.hpp"
#include "dvs/progress.hpp"
#include "dvs/raycast.hpp"
#include "dvs/scene_data.hpp"
#include "dvs/scene_types.hpp"

namespace {

struct Options {
  std::string scene_dir;
  std::string output_path = "events.txt";
  float threshold_pos = 0.3f;
  float threshold_neg = 0.3f;
  float end_time = std::numeric_limits<float>::max();
  bool threshold_neg_set = false;
  dvs::HapkeParams hapke;
};

void print_usage(const char *prog) {
  std::cout << "Usage: " << prog << " --scene_dir <bundle_dir> [options]\n"
            << "Options:\n"
            << "  --output <path>            output events file (default: "
               "events.txt)\n"
            << "  --threshold_pos <float>    positive event threshold\n"
            << "  --threshold_neg <float>    negative event threshold "
               "(default: threshold_pos)\n"
            << "  --w <float> --b <float> --c <float>\n"
            << "  --hS <float> --B_S0 <float> --hC <float> --B_C0 <float>\n"
            << "  --phi <float> --thetap_deg <float>\n";
}

float parse_float_arg(const char *key, const std::string &value) {
  try {
    return std::stof(value);
  } catch (const std::exception &) {
    throw std::runtime_error(std::string("invalid float for ") + key + ": " +
                             value);
  }
}

Options parse_args(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char *key) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + key);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--scene_dir") {
      opts.scene_dir = require_value("--scene_dir");
    } else if (arg == "--end-time") {
      opts.end_time =
          parse_float_arg("--end-time", require_value("--end-time"));
    } else if (arg == "--output") {
      opts.output_path = require_value("--output");
    } else if (arg == "--threshold_pos") {
      opts.threshold_pos =
          parse_float_arg("--threshold_pos", require_value("--threshold_pos"));
    } else if (arg == "--threshold_neg") {
      opts.threshold_neg =
          parse_float_arg("--threshold_neg", require_value("--threshold_neg"));
      opts.threshold_neg_set = true;
    } else if (arg == "--w") {
      opts.hapke.w = parse_float_arg("--w", require_value("--w"));
    } else if (arg == "--b") {
      opts.hapke.b = parse_float_arg("--b", require_value("--b"));
    } else if (arg == "--c") {
      opts.hapke.c = parse_float_arg("--c", require_value("--c"));
    } else if (arg == "--hS") {
      opts.hapke.hS = parse_float_arg("--hS", require_value("--hS"));
    } else if (arg == "--B_S0") {
      opts.hapke.B_S0 = parse_float_arg("--B_S0", require_value("--B_S0"));
    } else if (arg == "--hC") {
      opts.hapke.hC = parse_float_arg("--hC", require_value("--hC"));
    } else if (arg == "--B_C0") {
      opts.hapke.B_C0 = parse_float_arg("--B_C0", require_value("--B_C0"));
    } else if (arg == "--phi") {
      opts.hapke.phi = parse_float_arg("--phi", require_value("--phi"));
    } else if (arg == "--thetap_deg") {
      const float deg =
          parse_float_arg("--thetap_deg", require_value("--thetap_deg"));
      opts.hapke.thetap = deg * dvs::kPi / 180.0f;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (opts.scene_dir.empty()) {
    throw std::runtime_error("--scene_dir is required");
  }
  if (opts.threshold_pos <= 0.0f) {
    throw std::runtime_error("--threshold_pos must be > 0");
  }
  if (!opts.threshold_neg_set) {
    opts.threshold_neg = opts.threshold_pos;
  }
  if (opts.threshold_neg <= 0.0f) {
    throw std::runtime_error("--threshold_neg must be > 0");
  }
  return opts;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc <= 1) {
      print_usage(argv[0]);
      return 1;
    }

    const Options opts = parse_args(argc, argv);
    dvs::SceneBundle<dvs::Vec3, dvs::Uint3, dvs::Mat3> scene;
    std::string error_message;
    if (!dvs::load_scene_bundle(opts.scene_dir, &scene, &error_message)) {
      throw std::runtime_error("failed to load scene bundle: " + error_message);
    }
    std::cout << "Loaded scene bundle from " << opts.scene_dir << "\n";
    dvs::BVHBuildParams params; // 可使用默认参数
    int root_index =
        dvs::BVHBuilder::build(params, scene.bvh_nodes, scene.bvh_triangles);

    dvs::GpuRaycaster raycaster(scene, root_index);
    dvs::DVSBase dvs_sensor(scene.height, scene.width, opts.threshold_pos,
                            opts.threshold_neg);

    std::ofstream events_out(opts.output_path);
    if (!events_out) {
      throw std::runtime_error("failed to open output file: " +
                               opts.output_path);
    }

    std::size_t total_events = 0;
    dvs::ProgressBar bar(scene.frames.size());
    std::vector<float> light_frame(
        static_cast<std::size_t>(scene.width * scene.height), 0.0f);
    for (std::size_t frame_idx = 0; frame_idx < scene.frames.size();
         ++frame_idx) {
      const auto &frame = scene.frames[frame_idx];

      const dvs::Vec3 light_dir = frame.light_dir.normalized();
      raycaster.render_main(frame, light_frame);

      const std::vector<dvs::Event> events =
          dvs_sensor.process_frame(frame.timestamp, light_frame);
      for (const dvs::Event &e : events) {
        events_out << e.timestamp << " " << e.x << " " << e.y << " "
                   << e.polarity << "\n";
      }
      total_events += events.size();
      bar.update(frame_idx + 1);
    }

    std::cout << "Finished. Wrote " << total_events << " events to "
              << opts.output_path << "\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
