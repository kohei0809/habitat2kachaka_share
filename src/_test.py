import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image

# ストリームの設定
config = rs.config()
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)

# ストリーミング開始
pipeline = rs.pipeline()
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    for i in range(100):
        # フレーム待ち
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        # RGB
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = Image.fromarray(color_image)
        color_image.save("test_" + str(i) + ".png")

        # 深度
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image / 1000
        print(depth_image)

        # 2次元データをカラーマップに変換
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.02), cv2.COLORMAP_JET)

        depth_colormap = Image.fromarray(depth_colormap)
        depth_colormap.save("test_" + str(i) + "_.png")
        
        # イメージの結合
        #images = np.vstack(np.hstack((color_image, depth_colormap)) )
        #images = np.vstack(color_image)

finally:
    # ストリーミング停止
    pipeline.stop()
