import cv2
import time

cap = cv2.VideoCapture(1)
size = (640, 480)

for i in range(10):
    ret, frame = cap.read()

    cv2.imshow('webカメラ', frame)

    print("####################")
    print(frame.shape)
    cv2.imwrite('photo_' + str(i) + '.jpg', frame)
    print('写真が保存されました。')
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
