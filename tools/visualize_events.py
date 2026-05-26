import numpy as np
import cv2
import argparse

argsparser = argparse.ArgumentParser(description='Visualize DVS events from a binary file.')
argsparser.add_argument('--file', type=str, default='../bin/output/events.bin', help='Path to the binary file containing DVS events.')
argsparser.add_argument('--batch_size', type=int, default=30000, help='Number of events to read in each batch.')
args = argsparser.parse_args()


file_path = args.file
batch_size = args.batch_size
events_file = open(file_path, 'rb')

event_dtype = np.dtype([
    ('timestamp', np.float32),
    ('x', np.int16),
    ('y', np.int16),
    ('polarity', np.int8),
    ('depth', np.float32),
])

while True:
    # 读取一批事件数据
    batch_events = np.fromfile(events_file, dtype=event_dtype, count=batch_size)
    if batch_events.size == 0:
        break  # 读取完毕，退出循环
    # 将事件数据转换为numpy float32格式
    # ts, x, y, p = batch_events['timestamp'], batch_events['x'], batch_events['y'], batch_events['polarity']
    # 绘制事件图像
    height, width = 512, 512
    event_image = np.zeros((height, width, 3), dtype=np.uint8)
    for event in batch_events:
        t, x, y, p, d = event
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            if p > 0:
                event_image[y, x, 2] = 255  # 正事件用红色表示
            else:
                event_image[y, x, 0] = 255  # 负事件用蓝色表示

    cv2.imwrite('EventImage.png', event_image)