# Blender 导出数据规范（给 C++/CUDA DVS 仿真器）

本工程的 CUDA 仿真器不再直接依赖 `bpy`，而是读取 Blender 离线导出的 bundle 目录。  
bundle 目录必须包含以下文件：

| 文件名 | 类型 | 说明 |
|---|---|---|
| `metadata.txt` | 文本 | 分辨率、FPS、帧数等元信息 |
| `mesh_vertices.bin` | `float32` 二进制 | 顶点坐标，按 `Nx3` 存储，世界坐标系 |
| `mesh_normals.bin` | `float32` 二进制 | 顶点法线，按 `Nx3` 存储，世界坐标系、单位向量 |
| `mesh_indices.bin` | `uint32` 二进制 | 三角形索引，按 `Mx3` 存储 |
| `frames.bin` | `float32` 二进制 | 每帧相机/光照参数，固定 stride=24 float |

可选（用于连续位姿轨迹）：

| 文件名 | 类型 | 说明 |
|---|---|---|
| `camera_fcurves.csv` | CSV 文本 | 相机位置/四元数通道的 Bezier F-Curve 关键帧与左右控制柄 |

## 1. 坐标与单位约定

1. 使用 Blender 世界坐标（右手系，Z-up）。
2. 网格必须先三角化后导出（仿真器按三角形求交）。
3. 相机和光源按动画每帧导出，单位沿用 Blender 默认（米/秒系即可）。
4. `light_dir` 约定为光源局部 +Z 轴变换到世界坐标后的单位方向。

## 2. `metadata.txt` 格式

简单 `key=value`，至少包含：

```text
width=1024
height=768
fps=30.0
frame_count=250
```

## 3. `frames.bin` 内存布局（每帧 24 个 float32）

按顺序拼接：

1. `timestamp`：1，单位秒；不会再次除以 FPS
2. `camera_origin.xyz`：3
3. `camera_world`（4x4，row-major）：16
4. `light_dir.xyz`：3
5. `light_intensity`：1

即每帧：

```text
[t,
 ox, oy, oz,
 cam_world[16],
 lx, ly, lz,
 L]
```

## 4. Blender 导出命令

工程已提供导出脚本：`tools/export_blender_bundle.py`

```bash
blender -b your_scene.blend \
  -P /home/chenbofei/Blender/dvs_render/tools/export_blender_bundle.py \
  -- --output_dir /path/to/bundle --start_frame 1 --end_frame 250
```

连续位姿（Bezier/F-Curve）导出：

```bash
blender -b your_scene.blend \
  -P /home/chenbofei/Blender/dvs_render/tools/export_camera_fcurves.py \
  -- --output_csv /path/to/bundle/camera_fcurves.csv
```

CSV 列定义：

```text
channel,frame,value,hl_frame,hl_value,hr_frame,hr_value
```

其中 `channel` 取值：
`loc_x, loc_y, loc_z, quat_w, quat_x, quat_y, quat_z`。

> 应用通过 `CameraTrajectory` 对导出的帧样本作连续插值；CSV 的原始控制柄尚未接入。二进制时间为秒，文本轨迹首列为帧索引。

## 5. 渲染与相机参数

相机射线来自配置文件内参，采用整数像素中心及图像 x 向右、y 向下的坐标；位姿沿用 Blender x 向右、y 向上、-Z 向前。导出的 `metadata.txt` 不覆盖用户配置。运行时显式选择 `config/davis346.yaml` 或 `config/prophesee.yaml`，所有应用保存来源及实际相机参数。

图像与事件通道共用 `device_optics.cuh` 和 `hapke.hpp`。DVS 内核在轨迹区间内部追踪光线、采样电子数并积分像素电路。深度是命中射线距离的均值，空像素为 -1。此实现尚未包含镜头畸变及间接照明。
