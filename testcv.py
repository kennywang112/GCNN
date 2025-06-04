import cv2

# 嘗試從 0 到 4 找可用的攝影機 ID
for cam_id in range(5):
    cap = cv2.VideoCapture(cam_id)
    if cap.isOpened():
        print(f"攝影機 ID {cam_id} 可用")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f'Camera {cam_id}', frame)
            cv2.waitKey(1000)  # 顯示 1 秒
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"攝影機 ID {cam_id} 無法使用")