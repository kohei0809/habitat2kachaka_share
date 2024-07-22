# リアルタイムでrealsenseの画像をwindowsに表示するプログラム

import pyrealsense2 as rs
import numpy as np
import cv2

# ストリームの設定
config = rs.config()
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)

# ストリーミング開始
pipeline = rs.pipeline()
pipeline.start(config)

# AlignオブジェクトでRGBとDepthをキャリブレーション
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # フレーム待ち
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)   

        # RGB
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # 深度
        #depth_frame = frames.get_depth_frame()
        #depth_image = np.asanyarray(depth_frame.get_data())

        # 2次元データをカラーマップに変換
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.02), cv2.COLORMAP_JET)

        # イメージの結合
        #images = np.vstack(np.hstack((color_image, depth_colormap)) )
        images = np.vstack(color_image)

        # 表示
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # q キー入力で終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

finally:
    # ストリーミング停止
    pipeline.stop()
