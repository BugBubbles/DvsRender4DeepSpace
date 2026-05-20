# -*- coding: utf-8 -*-
import numpy as np
import rasterio
import os
import tqdm
import argparse

argsparser = argparse.ArgumentParser(description="分块导出 DEM 为 OBJ")
argsparser.add_argument("--input_dem", type=str, required=True, help="输入极区投影 DEM")
argsparser.add_argument("--output_dir", type=str, required=True, help="输出 OBJ 文件夹")
argsparser.add_argument("--tile_size", type=int, default=2000, help="tile 大小")
args = argsparser.parse_args()

def idx(i, j, nc):
    return i * nc + j + 1

# =========================
# 用户配置
# =========================

input_dem = args.input_dem     # 输入极区投影 DEM
output_dir = args.output_dir  # 输出 OBJ 文件夹
tile_size = args.tile_size                                    # tile 大小
os.makedirs(output_dir, exist_ok=True)

# =========================
# 2. 分块导出 OBJ
# =========================
tile_id = 0

with rasterio.open(input_dem) as src:
    transform = src.transform
    nodata = src.nodata
    rows, cols = src.height, src.width
    
    total_tiles = (rows // tile_size) * (cols // tile_size)
    progbar = tqdm.tqdm(total=total_tiles, unit="tiles", desc="Processing tiles", dynamic_ncols=True)

    for r0 in range(0, rows - 1, tile_size):
        for c0 in range(0, cols - 1, tile_size):

            r1 = min(r0 + tile_size, rows - 1)
            c1 = min(c0 + tile_size, cols - 1)

            window = rasterio.windows.Window(
                col_off=c0,
                row_off=r0,
                width=c1 - c0 + 1,
                height=r1 - r0 + 1,
            )

            tile_dem = src.read(1, window=window).astype(np.float32)

            # 处理 NoData
            if nodata is not None:
                tile_dem = np.where(tile_dem == nodata, np.nan, tile_dem)

            nr, nc = tile_dem.shape

            # 计算世界坐标
            row_grid, col_grid = np.meshgrid(
                np.arange(r0, r1 + 1),
                np.arange(c0, c1 + 1),
                indexing="ij"
            )

            XS = transform.a * (col_grid + 0.5) + transform.b * (row_grid + 0.5) + transform.c
            YS = transform.d * (col_grid + 0.5) + transform.e * (row_grid + 0.5) + transform.f

            # =======================================
            # 构建索引：仅对有效像素（非 NaN）写入顶点
            # =======================================

            vertex_index = -np.ones((nr, nc), dtype=np.int32)
            valid_mask = ~np.isnan(tile_dem)

            current_v = 1
            for i in range(nr):
                for j in range(nc):
                    if valid_mask[i, j]:
                        vertex_index[i, j] = current_v
                        current_v += 1

            # =======================================
            # 输出 OBJ
            # =======================================
            out_path = os.path.join(output_dir, f"LDEM_875S_5M_{tile_id:04d}.obj")
            with open(out_path, "w", encoding="utf-8") as f:

                # ---- 顶点（只对有效像素写入） ----
                for i in range(nr):
                    for j in range(nc):
                        if vertex_index[i, j] > 0:
                            f.write(
                                f"v {XS[i, j]:.3f} {YS[i, j]:.3f} {tile_dem[i, j]:.3f}\n"
                            )

                # ---- 面（只在四点都有效时写入） ----
                for i in range(nr - 1):
                    for j in range(nc - 1):

                        v1 = vertex_index[i, j]
                        v2 = vertex_index[i, j + 1]
                        v3 = vertex_index[i + 1, j]
                        v4 = vertex_index[i + 1, j + 1]

                        # 只要任意 v 是 -1，就跳过
                        if v1 < 0 or v2 < 0 or v3 < 0 or v4 < 0:
                            continue

                        f.write(f"f {v1} {v3} {v2}\n")
                        f.write(f"f {v2} {v3} {v4}\n")

            tile_id += 1
            progbar.update(1)
            
print("全部完成！OBJ 已生成。")