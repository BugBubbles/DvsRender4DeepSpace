#include "dvs/scene_types.hpp"
#include "dvs/scene_data.hpp"
#include <charconv>
#include <fstream>
#include <iostream>

namespace dvs {
// 将 YAML 解析为两层字符串映射（内部实现）
bool parse_yaml_to_map(
    const std::string &path,
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::string>> &kv,
    std::string &err) {
  std::ifstream file(path);
  if (!file.is_open()) {
    err = "Cannot open file: " + path;
    return false;
  }

  std::string line;
  std::string current_section;
  int line_number = 0;

  while (std::getline(file, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back(); // 处理 Windows 换行
    }

    // 跳过前导空白
    size_t start = 0;
    while (start < line.size() &&
           std::isspace(static_cast<unsigned char>(line[start]))) {
      ++start;
    }

    // 空行或注释行
    if (start == line.size() || line[start] == '#') {
      continue;
    }

    // 查找冒号
    size_t colon = line.find(':', start);
    if (colon == std::string::npos) {
      err = "Line " + std::to_string(line_number) + ": missing colon";
      return false;
    }

    // 提取键（去除尾部空格）
    std::string key = line.substr(start, colon - start);
    while (!key.empty() &&
           std::isspace(static_cast<unsigned char>(key.back()))) {
      key.pop_back();
    }
    if (key.empty()) {
      err = "Line " + std::to_string(line_number) + ": empty key";
      return false;
    }

    // 提取值部分（冒号之后）
    std::string value_part =
        (colon + 1 < line.size()) ? line.substr(colon + 1) : std::string();

    // 去掉行尾注释（考虑引号内 # 不视为注释）
    bool in_single = false, in_double = false;
    size_t comment_pos = std::string::npos;
    for (size_t i = 0; i < value_part.size(); ++i) {
      char c = value_part[i];
      if (in_single) {
        if (c == '\'')
          in_single = false;
      } else if (in_double) {
        if (c == '"')
          in_double = false;
      } else {
        if (c == '\'')
          in_single = true;
        else if (c == '"')
          in_double = true;
        else if (c == '#') {
          comment_pos = i;
          break;
        }
      }
    }
    if (comment_pos != std::string::npos) {
      value_part = value_part.substr(0, comment_pos);
    }

    // 去除值部分的首尾空格
    size_t val_start = 0;
    while (val_start < value_part.size() &&
           std::isspace(static_cast<unsigned char>(value_part[val_start]))) {
      ++val_start;
    }
    size_t val_end = value_part.size();
    while (val_end > val_start &&
           std::isspace(static_cast<unsigned char>(value_part[val_end - 1]))) {
      --val_end;
    }
    std::string value = value_part.substr(val_start, val_end - val_start);

    // 根据缩进判断层级
    if (start == 0) { // 顶层键
      if (!value.empty()) {
        err = "Line " + std::to_string(line_number) + ": top-level key '" +
              key + "' has a value, expected a mapping";
        return false;
      }
      current_section = key;
      kv[current_section]; // 创建空子映射
    } else {               // 子键（有缩进）
      if (current_section.empty()) {
        err = "Line " + std::to_string(line_number) + ": key '" + key +
              "' without a parent section";
        return false;
      }
      kv[current_section][key] = value;
    }
  }

  return true;
}

// 字符串转 int，失败返回 false
bool safe_stoi(const std::string &s, int &out) {
  const char *first = s.data();
  const char *last = s.data() + s.size();
  auto [ptr, ec] = std::from_chars(first, last, out);
  return ec == std::errc{} && ptr == last;
}

// 字符串转 float，失败返回 false
bool safe_stof(const std::string &s, float &out) {
  const char *first = s.data();
  const char *last = s.data() + s.size();
  auto [ptr, ec] = std::from_chars(first, last, out);
  return ec == std::errc{} && ptr == last;
}
// 字符串列表转 float 列表'[a, b, c]'，失败返回 false
bool safe_stof_list(const std::string &s, std::vector<float> &out) {
  out.clear();
  if (s.size() < 2 || s.front() != '[' || s.back() != ']') {
    return false;
  }
  std::string inner = s.substr(1, s.size() - 2);
  std::istringstream ss(inner);
  std::string item;
  while (std::getline(ss, item, ',')) {
    // 去除首尾空格
    size_t start = 0;
    while (start < item.size() &&
           std::isspace(static_cast<unsigned char>(item[start]))) {
      ++start;
    }
    size_t end = item.size();
    while (end > start &&
           std::isspace(static_cast<unsigned char>(item[end - 1]))) {
      --end;
    }
    std::string trimmed = item.substr(start, end - start);
    float value;
    if (!safe_stof(trimmed, value)) {
      return false;
    }
    out.push_back(value);
  }
  return true;
}

Config::Config(const std::string &config_file, std::string &err) {

  // 2. 解析 YAML 到 map
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      kv;
  if (!parse_yaml_to_map(config_file, kv, err)) {
    return;
  }

  // 3. 检查必须有 intrinsic, physic, scene 三个顶层键
  for (const auto &section : {"intrinsic", "physic", "scene"}) {
    if (kv.find(section) == kv.end()) {
      err = "Missing top-level section '" + std::string(section) + "'";
      return;
    }
  }

  // 4. 填充 CameraConfig
  const auto &cam_map = kv["intrinsic"];
  if (!extract_required(cam_map, "intrinsic", "width", camera.width, err,
                        safe_stoi))
    return;
  if (!extract_required(cam_map, "intrinsic", "height", camera.height, err,
                        safe_stoi))
    return;
  if (!extract_required(cam_map, "intrinsic", "fx", camera.fx, err, safe_stof))
    return;
  if (!extract_required(cam_map, "intrinsic", "fy", camera.fy, err, safe_stof))
    return;
  if (!extract_required(cam_map, "intrinsic", "cx", camera.cx, err, safe_stof))
    return;
  if (!extract_required(cam_map, "intrinsic", "cy", camera.cy, err, safe_stof))
    return;

  // 5. 填充 PhysicConfig
  const auto &noise_map = kv["physic"];
  if (!extract_required(noise_map, "physic", "threshold_pos",
                        physic.threshold_pos, err, safe_stof))
    return;
  if (!extract_required(noise_map, "physic", "threshold_neg",
                        physic.threshold_neg, err, safe_stof))
    return;
  if (!extract_required(noise_map, "physic", "refractory_time",
                        physic.refractory_time, err, safe_stof))
    return;
  if (!extract_required(noise_map, "physic", "readout_latency",
                        physic.readout_latency, err, safe_stof))
    return;
  if (!extract_required(noise_map, "physic", "readout_bandwidth",
                        physic.readout_bandwidth, err, safe_stof))
    return;

  for (const char *key : {"threshold_noise","dark_current","hpf_time_constant","shot_intensity",
       "photoreceptor_tau","intensity_reference","min_bandwidth_fraction","leak_rate_hz",
       "shot_noise_rate_hz","shot_noise_bright_factor"}) {
    if(noise_map.count(key)) {err=std::string("removed physic key '")+key+"'; migrate to dimensioned parameters in docs/physical_sensor.md";return;}
  }
  auto optional_float = [&](const char *key, float &value) {
    auto it = noise_map.find(key);
    if (it != noise_map.end() && !safe_stof(it->second, value))
      throw std::invalid_argument(std::string("invalid physic parameter: ") + key);
  };
  auto optional_int = [&](const char *key, int &value) {
    auto it = noise_map.find(key);
    if (it != noise_map.end() && !safe_stoi(it->second, value))
      throw std::invalid_argument(std::string("invalid physic parameter: ") + key);
  };
  optional_float("threshold_std", physic.threshold_std);
  optional_float("threshold_min", physic.threshold_min);
  optional_float("wavelength_nm", physic.wavelength_nm);
  optional_float("quantum_efficiency", physic.quantum_efficiency);
  optional_float("pixel_pitch_um", physic.pixel_pitch_um);
  optional_float("fill_factor", physic.fill_factor);
  optional_float("f_number", physic.f_number);
  optional_float("lens_transmission", physic.lens_transmission);
  optional_float("radiance_scale", physic.radiance_scale);
  optional_float("temperature_k", physic.temperature_k);
  optional_float("subthreshold_kappa", physic.subthreshold_kappa);
  optional_float("receptor_capacitance_f", physic.receptor_capacitance_f);
  optional_float("reset_capacitance_f", physic.reset_capacitance_f);
  optional_float("dark_current_a", physic.dark_current_a);
  optional_float("reset_leakage_a", physic.reset_leakage_a);
  optional_float("circuit_step", physic.circuit_step);
  optional_int("photon_shot_noise", physic.photon_shot_noise);
  optional_int("reset_thermal_noise", physic.reset_thermal_noise);
  optional_int("random_seed", physic.random_seed);
  optional_int("max_events_per_step", physic.max_events_per_step);
  optional_int("readout_queue_capacity", physic.readout_queue_capacity);
  for (float value : {physic.threshold_pos,physic.threshold_neg,physic.threshold_std,
       physic.threshold_min,physic.refractory_time,physic.readout_bandwidth,physic.readout_latency,
physic.wavelength_nm,physic.quantum_efficiency,physic.pixel_pitch_um,physic.fill_factor,physic.f_number,physic.lens_transmission,physic.radiance_scale,physic.temperature_k,physic.subthreshold_kappa,physic.receptor_capacitance_f,physic.reset_capacitance_f,physic.dark_current_a,physic.reset_leakage_a,physic.circuit_step}) {
    if (!std::isfinite(value) || value<0) {err="physic values must be finite and nonnegative";return;}
  }
  if(physic.threshold_pos<=0 || physic.threshold_neg<=0 || physic.threshold_min<=0 ||
     physic.wavelength_nm<=0 || physic.pixel_pitch_um<=0 || physic.f_number<=0 ||
     physic.temperature_k<=0 || physic.subthreshold_kappa<=0 || physic.subthreshold_kappa>1 ||
     physic.receptor_capacitance_f<=0 || physic.reset_capacitance_f<=0 || physic.circuit_step<=0 ||
     physic.quantum_efficiency>1 || physic.fill_factor>1 || physic.lens_transmission>1 ||
     physic.photon_shot_noise<0 || physic.photon_shot_noise>1 || physic.reset_thermal_noise<0 || physic.reset_thermal_noise>1 ||
     physic.random_seed<0 || physic.max_events_per_step<1 || physic.max_events_per_step>64 || physic.readout_queue_capacity<1) {
    err="invalid optical/circuit model parameter range";return;
  }

  // 6. 填充 SceneConfig
  const auto &scene_map = kv["scene"];
  if (!extract_required(scene_map, "scene", "frame_count", scene.frame_count,
                        err, safe_stoi))
    return;
  if (!extract_required(scene_map, "scene", "fps", scene.fps, err, safe_stof))
    return;
  if (!extract_required(scene_map, "scene", "imu_fps", scene.imu_fps, err,
                        safe_stof))
    return;
  if (!extract_required(scene_map, "scene", "cam_fps", scene.cam_fps, err,
                        safe_stof))
    return;
  if (!extract_required(scene_map, "scene", "time_resolution",
                        scene.time_resolution, err, safe_stof))
    return;
  if (!extract_required(scene_map, "scene", "supersample_factor",
                        scene.supersample_factor, err, safe_stoi))
    return;
  if (!extract_required(scene_map, "scene", "blur_resolution",
                        scene.blur_resolution, err, safe_stof))
    return;
  scene.translation = {0, 0, 0}; scene.rotation = {0, 0, 0};
  for (auto pair : {std::make_pair("translation", &scene.translation), std::make_pair("rotation", &scene.rotation)}) {
    auto it = scene_map.find(pair.first);
    if (it != scene_map.end() && (!safe_stof_list(it->second, *pair.second) || pair.second->size() != 3)) {
      err = std::string("scene.") + pair.first + " must have three numbers"; return;
    }
    for (float value : *pair.second) if (!std::isfinite(value)) { err = "nonfinite extrinsics"; return; }
  }
  for (float value : {camera.fx, camera.fy, camera.cx, camera.cy, scene.fps,
                      scene.time_resolution, scene.cam_fps, scene.imu_fps, scene.blur_resolution})
    if (!std::isfinite(value)) { err = "nonfinite camera or scene parameter"; return; }
  if (camera.width <= 0 || camera.height <= 0 || camera.width > 32767 || camera.height > 32767 ||
      camera.fx <= 0 || camera.fy <= 0 || scene.frame_count < 2 || scene.fps <= 0 ||
      scene.time_resolution <= 0 || scene.cam_fps <= 0 || scene.imu_fps <= 0 ||
      scene.blur_resolution <= 0 || scene.supersample_factor < 1) { err = "invalid scene or camera range"; return; }

  double sample_count = static_cast<double>(camera.width) * camera.height * scene.supersample_factor * scene.supersample_factor;
  if (sample_count > std::numeric_limits<int>::max()) { err = "too many aperture samples"; return; }

  const auto &hapke_map = kv["hapke"];
  if (!extract_required(hapke_map, "hapke", "b", hapke.b, err, safe_stof))
    return;
  if (!extract_required(hapke_map, "hapke", "c", hapke.c, err, safe_stof))
    return;
  if (!extract_required(hapke_map, "hapke", "hS", hapke.hS, err, safe_stof))
    return;
  if (!extract_required(hapke_map, "hapke", "hC", hapke.hC, err, safe_stof))
    return;
  if (!extract_required(hapke_map, "hapke", "thetap", hapke.thetap, err,
                        safe_stof))
    return;
  if (!extract_required(hapke_map, "hapke", "w", hapke.w, err, safe_stof))
    return;
  if (!extract_required(hapke_map, "hapke", "B_S0", hapke.B_S0, err, safe_stof))
    return;
  if (!extract_required(hapke_map, "hapke", "B_C0", hapke.B_C0, err, safe_stof))
    return;
  if (!extract_required(hapke_map, "hapke", "phi", hapke.phi, err, safe_stof))
    return;
  for (float value : {hapke.b, hapke.c, hapke.hS, hapke.hC, hapke.thetap,
                      hapke.w, hapke.B_S0, hapke.B_C0, hapke.phi})
    if (!std::isfinite(value)) { err = "nonfinite Hapke parameter"; return; }
  if (hapke.b < 0 || hapke.b >= 1 || hapke.c < -1 || hapke.c > 1 ||
      hapke.hS < 0 || hapke.hC < 0 || hapke.thetap < 0 || hapke.thetap >= kPi/2 ||
      hapke.w < 0 || hapke.w > 1 || hapke.B_S0 < 0 || hapke.B_C0 < 0 ||
      hapke.phi < 0 || 1.209f * std::pow(hapke.phi, 2.0f/3.0f) >= 1) {
    err = "invalid Hapke parameter range"; return;
  }

}
} // namespace dvs
