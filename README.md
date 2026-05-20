# DVS Render (CUDA) Usage

## Overview
This project renders scene lighting with a GPU BVH + Hapke BRDF and can output:
- 8-bit PNG camera frames
- Depth maps
- DVS event stream (binary)
- Ground-truth pose and motion logs

## Build Requirements
- Linux
- CUDA Toolkit
- CMake >= 3.22
- C++17 compiler
- OpenMP
- Eigen3
- OpenCV

## Build (Conda + CMake)
This uses conda for Eigen/OpenCV/CMake, while CUDA and the compiler stay in the system.

```bash
conda env create -f environment.yml
conda activate dvs4deepspace

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If you want to build with make instead of ninja:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
cmake --build build -j
# or
make -C build -j
```

Optional feature flags:
- `-DENABLE_PHYSIC_EFFECTS=ON|OFF`
- `-DENABLE_HIGH_PASS_FILTER=ON|OFF`
- `-DENABLE_AER_DELAY=ON|OFF`

## Build (System Packages)
Example on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libeigen3-dev libopencv-dev
# Install CUDA Toolkit separately (nvcc must be in PATH)

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Export Blender Bundle
Use the provided Blender scripts. Example:

```bash
blender -b your_scene.blend \
  -P tools/export_blender_bundle.py \
  -- --output_dir /path/to/bundle --start_frame 1 --end_frame 250
```

Optional continuous trajectory export:

```bash
blender -b your_scene.blend \
  -P tools/export_camera_fcurves.py \
  -- --output_csv /path/to/bundle/camera_fcurves.csv
```

## Run
Executables are generated from `apps/main_*.cpp` and written to `bin/`.

### Camera frames
```bash
./bin/cam --scene_dir /path/to/bundle --output ./output --config ./config/default.yaml
```
Outputs:
- `output/images/*.png` (8-bit, grayscale)
- `output/depth/*.tif` (32-bit float depth)

### Motion blur camera frames
```bash
./bin/cam_blur --scene_dir /path/to/bundle --output ./output --config ./config/default.yaml
```
Outputs:
- `output/blur_images/*.png`

### DVS events
```bash
./bin/dvs --scene_dir /path/to/bundle --output ./output --config ./config/default.yaml
```
Outputs:
- `output/events.bin`

### Ground-truth pose and motion
```bash
./bin/gt_pose --scene_dir /path/to/bundle --output ./output --config ./config/default.yaml
```
Outputs:
- `output/gt_motion.txt`
- `output/gt_pose.txt`

## Bundle Format (Current Code)
The current C++ loader expects `frames.bin` to use a 24-float stride written by
`tools/export_blender_bundle.py`:

```
[timestamp,
 cam_origin.x, cam_origin.y, cam_origin.z,
 cam_world(4x4 row-major, 16 floats),
 light_dir.x, light_dir.y, light_dir.z,
 light_intensity]
```

Notes:
- The loader uses the 3x3 rotation from `cam_world`; projection is not read.
- `metadata.txt` is written by the Blender script but is not consumed by the
  C++ loader; camera intrinsics come from `config/default.yaml`.
- The older `docs/blender_export_data.md` mentions a 40-float stride. The
  current implementation uses 24 floats as shown above.

## Event Stream Format
`events.bin` stores packed `dvs::Event` records (13 bytes each):

```
float  timestamp
int16  x
int16  y
int8   polarity   # 1 for positive, 0 for negative
float  depth
```

## Image Conversion
Camera rendering converts scene-linear intensity to 8-bit PNG using
`render_float_to_png_bytes`:
- `tone_map_aces(x) = clamp(x / 100, 0..1)`
- linear to sRGB
- optional dither
- `to_byte` quantization

## Configuration File
`config/default.yaml` defines:
- `intrinsic`: width, height, fx, fy, cx, cy
- `physic`: event thresholds, dark current, refractory time, HPF time constant,
  shot noise, AER delay parameters
- `scene`: frame_count, fps, imu_fps, cam_fps, time_resolution,
  supersample_factor, blur_resolution
- `hapke`: BRDF parameters

## Tips
- If `events.bin` is empty, check thresholds and light intensity.
- For stable DVS initialization, the first frame mean intensity is used.
- If CUDA is not found, ensure `nvcc` is in PATH and `CUDAToolkit` can be
  located by CMake.

## Acknowledge
We thank the contributors to this project and the open-source software [Blender](https://www.blender.org/), [Eigen](https://eigen.tuxfamily.org/), and [OpenCV](https://opencv.org/).
