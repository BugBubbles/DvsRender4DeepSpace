# How to DIY a custom environment Render result?
首先，你需要安装一些必要的前置软件，包括：
- [Blender](https://www.blender.org/)：一个强大的开源3D建模和渲染软件，可以用来创建和编辑3D模型，推荐安装4.2.3 LTS以上版本 （必选）。
- [NASA ISIS](https://isis.astrogeology.usgs.gov/9.0.0/index.html)：一个专门用于处理观测数据的软件，可以用于光度校正与地形生成，推荐安装9.0.0版本（Optional for photometric correction）。
- [GDAL](https://gdal.org/)：一个用于处理地理空间数据的库，可以用来处理和转换地形数据，推荐安装3.7.0版本（Optional for photometric correction）（Optional for photometric correction）。

# 建立地形三维环境
## 准备地形数据
选取感兴趣的研究区域，从NASA的[PDS RDR](https://data.lroc.im-ldi.com/lroc/rdr_product_select)检索网站或其他公开数据源下载相关的地形数据和影像数据。你下载的数据应当至少包含DEM（数字高程模型）和对应的影像数据。
## 构建三维地形
利用[build_obj.py脚本](../tools/build_obj.py)，将下载的DEM数据转为多个`.obj`文件，这些文件每个对应一小块三维地形。然后使用Blender图形化界面打开这些`.obj`文件，进行必要的调整和优化，将它们组合成一个完整的三维地形模型，并设置必要的Blender相机参数、相机运动轨迹、光照方向与强度、仿真帧率等参数。

然后，使用[导出工具](../tools/export_blender_bundle.py)，通过命令行导出当前Blender的工程为几个包含所有必要资源的bin文件，命令示例如下：
```bash
blender -b your_blender_file.blend --python export_blender_bundle.py -- --output_dir output_directory --start_frame 1 --end_frame 100
```
导出所得bin文件组织形式为：
```output_directory/
├── frame.bin            # 每个仿真帧的相机位姿、光照参数等信息
├── mesh_indices.bin     # 三维地形模型的面片索引数据
├── mesh_normals.bin     # 三维地形模型的法线数据
├── mesh_vertices.bin    # 三维地形模型的顶点数据
└── metadata.txt         # 相机内参、仿真参数等元数据信息
```
## (Optional) 进行光度校正
以上所得的三维地形模型可以直接用于渲染，但是其缺失了当地的纹理，因此渲染所得的图像与真实地形存在差异，且随地形范围增大而增大。如果为了得到更真实的仿真图像，需要获取当地纹理，必须通过ISIS光度校正工具进行处理。

使用以下脚本在ISIS中进行光度校正：
```bash
lronac2isis from=your_nac_imagery.img to=edr.cub
spiceinit from=edr.cub
lronaccal from=edr.cub to=cdr
cam2map from=cdr.cub to=map.cub map=dem.map pixelres=map
phocube from=map.cub to=your_nac_imagery_pho.cub dn=true phase=true emission=true incidence=true latitude=false longitude=false
```

校正得到的your_nac_imagery_pho.cub文件包含了四个波段，分别存储DN值（I/F值）、相位角、发射角和入射角。利用这四个值以及已知的散射模型（Hapke模型）可计算当地的纹理信息`albedo`，可利用CUDA C++工具[albedo_map](../tools/cpp_tools/main_albedo_map.cu)快速计算，注意单个tiff不超出显存限制（示例数据：8207x52175像素 ~ 7.98 GB，可通过ISIS切片或下采样进一步压缩内存）。只需要执行命令：
```bash
albedo_map --input_cub your_nac_imagery_pho.cub --output albedo.tiff
```

得到的`albedo.tiff`文件即为当地纹理信息，且具有投影坐标系信息，利用Gdal库可将多块tiff合并为整体tiff，使用命令，注意在使用前将各个子块重采样至相同分辨率大小再进行合并操作：
```bash
gdalwarp albedo_tile_0000.tiff albedo_tile_0001.tiff -r bilinear -of GTiff merged_albedo.tiff
```

最后，将得到的纹理贴图`merged_albedo.tiff`与之前构建的三维地形模型进行结合，重新导入Blender的材质节点图中，以albedo代替Hapke的`w`变量输入进行渲染，即可得到更真实的仿真图像。