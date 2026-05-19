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
- Mesh data is converted into triangles; invalid indices are skipped.
- Frames provide camera pose and light parameters per time step.

### BVH Build
- `bvh/build.cpp` builds a binary BVH using SAH binning (`bvh/binning.cpp`).
- Triangle order is rearranged to match BVH leaf ranges.

### Ray Casting (GPU)
- `raycast.cu` generates per-pixel ray directions.
- `batch_intersect_kernel` traverses the BVH with a small stack.
- `batch_render_kernel` computes intensity using Hapke BRDF.

### Camera Rendering
- `CamRayCaster` (in `cam_render.cu`) renders a frame + depth map.
- Supersampling uses random offsets per pixel (`supersample_factor`).
- `CamBlurRayCaster` accumulates multiple sub-frames over time for motion blur.

### DVS Event Generation
- `DvsRayCaster` (in `dvs_render.cu`) renders intensity and converts to events.
- Base model in `dvs_base.cu`:
  - log intensity thresholding
  - positive and negative polarity events
- Optional physics:
  - threshold noise
  - dark current
  - refractory time
  - high-pass filter (photoreceptor response)
  - AER row delay

Pseudocode (idealized):

```text
logI = log(I + dark_current)
if logI - last_logI >= threshold_pos:
  emit event(+)
  last_logI = logI
elif last_logI - logI >= threshold_neg:
  emit event(-)
  last_logI = logI
```

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
- `config/default.yaml` configures intrinsics, DVS physics, sampling rates, and
  Hapke BRDF parameters.
- Blender export scripts in `tools/` produce the bundle inputs.
