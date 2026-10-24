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
./bin/cam --scene_dir /path/to/bundle --output ./output --config ./config/davis346.yaml
```
Outputs:
- `output/images/*.png` (8-bit, grayscale)
- `output/depth/*.tif` (32-bit float depth)

### Motion blur camera frames
```bash
./bin/cam_blur --scene_dir /path/to/bundle --output ./output --config ./config/davis346.yaml
```
Outputs:
- `output/blur_images/*.png`

### DVS events
```bash
./bin/dvs --scene_dir /path/to/bundle --output ./output --config ./config/davis346.yaml
```
Outputs:
- `output/events.bin`

### Ground-truth pose and motion
```bash
./bin/gt_pose --scene_dir /path/to/bundle --output ./output --config ./config/davis346.yaml
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
  C++ loader; camera intrinsics come from `config/davis346.yaml`.
- Binary frame timestamps are seconds; text trajectory timestamps are frame indices divided by scene FPS.
- Pixel-center coordinates are integer-valued; intrinsics use x-right/y-down, and poses use Blender x-right/y-up/z-backward.
- Every application saves `source_config.yaml` and `camera_used.yaml`. Default CLI camera is DAVIS346; select Prophesee explicitly with `config/prophesee.yaml`.


## Event Stream Format
The DVS executable writes `events.bin` by default. Use
`--event_format hdf5` (or `--event-format h5`) to write `events.h5` in the
same output directory instead.

`events.bin` stores legacy packed records (13 bytes each); internal events use double timestamps:

```
float  timestamp
int16  x
int16  y
int8   polarity   # 1 for positive, 0 for negative
float  depth
```

`events.h5` stores root-level one-dimensional datasets:

```
x          int16 pixel x coordinate
y          int16 pixel y coordinate
t          int64 timestamp in microseconds
p          int8  polarity, 1 for positive and 0 for negative
ms_to_idx  int64 event index for each integer millisecond timestamp
```

`ms_to_idx[ms]` is the first event index whose timestamp is greater than or
equal to `ms * 1000` microseconds. The final entries map to the total event
count so `[ms0, ms1)` slices can use `ms_to_idx[ms0]:ms_to_idx[ms1]`.

## Image Conversion
Camera rendering converts scene-linear intensity to 8-bit PNG using
`render_float_to_png_bytes`:
- `display_exposure(x) = clamp(x / 100, 0..1)`
- linear to sRGB
- optional dither
- `to_byte` quantization

## Configuration File
`config/davis346.yaml` defines:
- `intrinsic`: width, height, fx, fy, cx, cy
- `physic`: event thresholds, dimensioned optical/circuit parameters, reset hold,
  electron shot noise and readout queue parameters
- `scene`: frame_count, fps, imu_fps, cam_fps, time_resolution,
  supersample_factor, blur_resolution
- `hapke`: BRDF parameters

## Tips
- If `events.bin` is empty, check thresholds and light intensity.
- DVS initializes each continuous pixel circuit from its own scene-ray photocurrent, including dark current.
- If CUDA is not found, ensure `nvcc` is in PATH and `CUDAToolkit` can be
  located by CMake.

## Acknowledge
We thank the contributors to this project and the open-source software [Blender](https://www.blender.org/), [Eigen](https://eigen.tuxfamily.org/), and [OpenCV](https://opencv.org/).

## Physical event sensor

The DVS path traces rays within trajectory intervals, converts Hapke radiance to photoelectron current, integrates nonlinear pixel circuits, and solves comparator crossing times in CUDA. It uses no image-pair differencing or ESIM/v2e runtime.

See [physical_sensor.md](docs/physical_sensor.md) for the local-Zotero literature, model equations, parameter units, calibration limits, and CUDA regression tests. For subsequent experiments use the user-defined `config/davis346.yaml`, `config/prophesee.yaml`, or the selected dataset camera calibration, preserving its resolution and intrinsic parameters. `generic_physical.yaml` documents the reduced circuit defaults; earlier 64×48 runs are reduced-resolution regression checks, not matched-device experiments. Each run records its configuration and readout statistics.

## CUDA 性能优化

完整相机配置的优化模式、原生对照、数值检查和多 GPU 长序列运行见 [docs/optimization.md](docs/optimization.md)。HDF5 事件入口支持 `--solver adaptive-optics`；60 s 数据集使用 `tools/run_sequences.py --mode render-optimized`。保持原分辨率、内参、16×16 子光线和原电路步长，改变光学查询的时间组织方式。RTX 4090 本地构建需指定 `-DCMAKE_CUDA_ARCHITECTURES=89`。
