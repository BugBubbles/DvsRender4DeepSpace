# 原始资源与原始相机参数的长序列

输出根目录是 `/home/temp/cbf_datasets/simulate`。5 条输入轨迹分别使用
`config/davis346.yaml` 和 `config/prophesee.yaml`，共 10 段，每段观测时间为
60 秒。`dataset.json` 与各段 `sequence.json` 记录资源和完成状态。
`poses_prepared_not_rendered` 表示位姿已生成，**事件、图像、深度尚未渲染**。

原轨迹不足一分钟时，以原来的时间尺度往返延续。端点附近 0.5 秒使用 C2
相位转向，位姿没有跳变；区间外速度完全保留，转向区间的峰值相位速度为
原来的 1.25 倍。该扩展是运动学轨迹，不是经过航天器动力学验证的飞行轨迹。
星空按用户授权省略。Prinzvent 的 DOM 用作地图匹配参考，SSA 文件用于材质；
DOM 不作为单次散射反照率输入。

## 数据组成

共享的 `maps` 保存原始 DEM、SSA、DOM 和地理参考信息。每段保存以下数据：

- `source_config.yaml`、`source_trajectory.txt`：原始参数与轨迹。
- `calibration.yaml`、`camera_used.yaml`：标定与实际读取的相机参数。
  源配置使用理想针孔模型，零畸变明确属于仿真模型，不能替代硬件实测标定。
- `poses.csv`：200 Hz 的相机到局部世界位姿，0 到 60 秒，包含首尾。
  相机坐标为 OpenCV 的右、下、前；内部 Blender 旋转乘以 `diag(1,-1,-1)`。
- `motion.csv`：世界坐标下的线速度、角速度与光照；不冒充带噪声 IMU 测量。
- `pose_geographic.csv`：原始投影坐标、源 DEM 高程和月面经纬度。
  原始地图 CRS 由 GDAL/OGR 转换；局部世界不是另行定义的 ENU。
  高程基准沿用资源，尚未独立核验。
- `events.h5`：等长的 `x,y,t,p` 列和 `ms_to_idx`。`t` 为 int64 微秒，
  `p=1` 是 ON，`p=0` 是 OFF，时间表示 FIFO 传输完成。观测在 60 秒停止，
  已排队事件可以在之后输出，结束时间另行记录。
- `frames.h5`：`t[N]`、`images[N,H,W]`、`depth[N,H,W]`。
  两种相机均为原始分辨率，每段 30 Hz 共 1801 帧。图像是固定曝光映射的
  瞬时灰度 sRGB 预览，不是标定过的 APS 电路仿真。深度为命中子光线的
  平均射线距离，未命中为 -1，不是光轴方向 Z。深度使用无损 float32 存储。
- `sequence.json`、`validation.json`：来源与完成检查。日志在工作区 audit 中。

事件由 CUDA 光线查询、Hapke 辐射、电子计数、非线性受光电路与比较器产生。
帧文件不参与事件生成。光学/电路参数仍是未标定的通用物理参数，不能据此
宣称已经匹配真实 DAVIS 或 Prophesee 的背景活动和噪声分布。

## 容量限制

500 GB 指十进制 `500000000000` 字节，覆盖整个输出根目录，包括共享地图、
所有相机、所有序列和未完成文件。按逻辑文件长度计数，稀疏、压缩文件也
保守计入。480 GB 为提前停止线，余下 20 GB 保留给写入边界和停止元数据。
Python 与 C++ 使用相同的跨进程锁，检查、预留、写入、刷新在锁内执行。
事件、帧、CSV 和元数据都经过容量检查。容量不足时进程返回 42，监督器
终止其它生成进程，保留已有数据，并在 audit 中记录分类体积分析。
普通输出不得使用停止后的元数据应急预算。

按当前启用的 5 百万事件/秒 FIFO 上界、100 万事件队列尾部、每事件 13 字节、
两款原始相机每帧 5 字节/像素（灰度 + 深度）、各 1801 帧估算，10 段加地图和
保守元数据余量约 59.45 GB。该估算不依赖压缩率；取消带宽上界或额外保存
光子/光线/中间缓冲区会改变数据量，需要重新估算。

## 运行与验证

代码和可执行文件留在项目内。准备资源：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/chenbofei/miniconda3/bin/python tools/prepare_sequences.py
PYTHONDONTWRITEBYTECODE=1 /home/chenbofei/miniconda3/bin/python tools/run_sequences.py --mode prepare-poses
```

查看体积计划：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/chenbofei/miniconda3/bin/python tools/run_sequences.py --mode plan
```

原采样渲染入口：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/chenbofei/miniconda3/bin/python tools/run_sequences.py --mode render-native --gpus 0,1,2,3,4,5,6,7
```

该入口保留配置的 16×16 空间采样、10 µs 轨迹区间和 1 µs 电路单元，只合并
CUDA 启动，不改变每个单元的光线或随机电子计数。事件与图像共享一份 GPU
地形，完整南极地形的 CPU 构建会串行，避免多个大网格同时耗尽内存。
不覆盖已有事件/帧文件，也尚不支持从中断的电路/RNG 状态恢复。

**性能限制：**完整 Prinzvent DEM 的单次原采样基准：10 µs 仿真耗时约
2.639 秒；线性外推单段 DAVIS 60 秒约 183 天。这不是已完成的长序列测量。
当前未启动这批耗时数月的渲染，也没有默认启用自适应近似。自适应光学与
电路求解需取得用户选择后另行实现和验证，不能擅自缩小相机分辨率、内参、
阈值或放慢轨迹来掩盖计算成本。

生成完成后逐段检查原始参数、60 秒帧/位姿覆盖、事件顺序、坐标边界、极性、
毫秒索引与有效深度；全部通过才标记为 `complete`。

## 优化渲染模式

使用 `--mode render-optimized --gpus 0,1,2,3,4,5,6,7` 启动十段序列。实际数值参数保存在各段 `numerical_model.json`；原相机参数与原速度轨迹不变。见 [optimization.md](optimization.md)。日志与性能报告位于项目 audit 目录，数据集目录只保存数据、标定、地图和来源/完成状态信息。
