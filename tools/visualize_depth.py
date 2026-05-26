import cv2
import numpy as np
import argparse

argsparser = argparse.ArgumentParser(description='Visualize depth from EXR files')
argsparser.add_argument('--input', type=str, default='../bin/output/depth/2.083333.tif', help='Path to the input EXR file')
args = argsparser.parse_args()

file_path = args.input
# 加载EXR图像
exr_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
# 提取单通道数据（假设我们使用第一个通道）
channel_data = exr_image[:, :]
# 归一化数据到0-255范围，排除为负的深度值
mask = channel_data > 0  # 只处理正值
channel_data[mask] = (channel_data[mask] - channel_data[mask].min()) / (channel_data[mask].max() - channel_data[mask].min()) * 255
channel_data[~mask] = 0  # 将非正值设为0
channel_data = channel_data.astype(np.uint8)
# 显示图像
cv2.imwrite('DepthImage.png', channel_data)