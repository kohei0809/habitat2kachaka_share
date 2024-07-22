# OpenCVをインポート
import cv2

# サンプル動画ファイル
videoPath = "./video_dir/24-04-23 16-39-21/episode=aaa-ckpt=362-0.0-1.mp4"

cap = cv2.VideoCapture(videoPath)

# 横幅
print(f"width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")