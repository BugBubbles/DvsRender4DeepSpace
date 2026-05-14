#include "dvs/scene_types.hpp"
#include "dvs/scene_data.hpp"
#include <charconv>
#include <fstream>

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
  if (!extract_required(noise_map, "physic", "threshold_noise",
                        physic.threshold_noise, err, safe_stof))
    return;
  if (!extract_required(noise_map, "physic", "dark_current",
                        physic.dark_current, err, safe_stof))
    return;
  if (!extract_required(noise_map, "physic", "refractory_time",
                        physic.refractory_time, err, safe_stof))
    return;
  if (!extract_required(noise_map, "physic", "readout_bandwidth",
                        physic.readout_bandwidth, err, safe_stof))
    return;

  // 6. 填充 SceneConfig
  const auto &scene_map = kv["scene"];
  if (!extract_required(scene_map, "scene", "frame_count", scene.frame_count,
                        err, safe_stoi))
    return;
  if (!extract_required(scene_map, "scene", "fps", scene.fps, err, safe_stof))
    return;
}
} // namespace dvs
