import cv2
import open3d as o3d

def get_rgb_image_from_sensor():
    # Open3Dを使用してXtion ProからRGB画像を取得する
    # センサーの初期化
    pipeline = o3d.io.AzureKinectSensorConfig()
    sensor = o3d.io.AzureKinectSensor(pipeline)
    sensor.connect()
    
    # RGB画像を取得する
    rgbd = sensor.capture_video_frame()
    rgb_image = rgbd.color
    
    # センサーとの接続を切断する
    sensor.disconnect()
    
    return rgb_image

# RGB画像を取得する
rgb_image = get_rgb_image_from_sensor()

# OpenCVを使ってRGB画像を表示する
rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imshow('RGB Image', rgb_image_cv2)
cv2.imwrite('photo_rgb.jpg', frame)
print('写真が保存されました。')
cv2.waitKey(0)
cv2.destroyAllWindows()
