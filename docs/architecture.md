# DVS Render (CUDA) Architecture

## High-Level Flow
1. Load mesh and frames from a Blender bundle.
2. Build a BVH over triangles for fast ray intersection.
3. For each time step, generate rays, intersect BVH, shade with Hapke BRDF.
4. Produce camera images, depth maps, or event streams depending on the app.

## Core Modules

### Scene Data
- `scene_data.cpp` loads `mesh_vertices.bin`, `mesh_normals.bin`,
  `mesh_indices.bin`, and `frames.bin`.
- Mesh data keeps indexed triangles; invalid indices are rejected before GPU upload.
- Frames provide camera pose and light parameters per time step.

### BVH Build
- `bvh/build.cpp` builds a binary BVH using SAH binning (`bvh/binning.cpp`).
- Triangle order is rearranged to match BVH leaf ranges.

### Shared GPU Scene and Optics
- `DeviceScene` owns indexed BVH/mesh/material buffers and releases partial allocations on failure.
- `device_ray.hpp` contains nearest-hit and any-hit shadow traversal.
- `device_optics.cuh::sample_pixel` supplies the same fixed stratified rays, normal/albedo interpolation, hard shadows, Hapke directional reflectance and cosine-fourth optical falloff to image and event paths.
- Integer pixel centers and y-down image coordinates are converted to Blender y-up/-Z camera rays.

### Camera Rendering
- `CamRayCaster` in `cam_render.cu` overwrites per-pixel image and range buffers each call; missed rays have range -1.
- Image radiance averages every spatial sample; depth averages only geometrically hit rays, including shadowed surfaces.
- Image-only and image/depth APIs use one kernel. Motion blur is time-weighted midpoint exposure integration in `main_cam_blur.cpp`, including a final partial exposure.

### DVS Event Generation
- `DvsRayCaster` uploads the BVH/mesh once and supplies scene pointers and trajectory horizons to `EventSensor`; pixel ray sampling is inside the fused circuit kernel.
- `pixel_model.hpp` defines nonlinear subthreshold current-balance receptor, voltage thresholds, self-timed reset hold and capacitor leakage.
- `event_sensor.cu` fuses time-local BVH/Hapke ray queries, electron charge sampling, circuit integration and nonlinear comparator roots on CUDA. It takes scene/trajectory intervals directly; no intensity image pair is consumed. Internal times are double precision.
- `device_ray.hpp` shares intersection/shadow routines with image rendering; `device_optics.cuh` evaluates pose/light and rays within each integration cell.
- `event_readout.cpp` serializes a bounded FIFO across scene steps, reports drops and flushes the final backlog.
- `dvs_writer.hpp` preserves the 13-byte binary format or writes int64 microseconds in HDF5.
- Detailed equations, literature and limitations are in [physical_sensor.md](physical_sensor.md).

### Trajectory
- `trajectory.cpp` builds cubic Bezier curves (auto-clamped).
- `CameraTrajectory::evaluate(t)` returns pose, velocity, and light values.
- Current apps use frame samples; exported fcurves are not wired into apps.

### Image Conversion
- `render_float_to_png_bytes` (in `blenderize.hpp`) converts scene-linear
  intensity to 8-bit PNG:
  - simple tone mapping (x / 100 clamp)
  - linear to sRGB
  - optional dither
  - byte quantization

## App Layer
- `apps/main_cam.cpp` renders PNG + depth at `cam_fps`.
- `apps/main_cam_blur.cpp` renders motion blur frames.
- `apps/main_dvs.cpp` outputs `events.bin` at `time_resolution`.
- `apps/main_gt_pose.cpp` outputs ground-truth motion and pose logs.

## Data Dependencies
- `config/davis346.yaml` or `config/prophesee.yaml` configures intrinsics, DVS physics, sampling rates, and
  Hapke BRDF parameters.
- Blender export scripts in `tools/` produce the bundle inputs.
