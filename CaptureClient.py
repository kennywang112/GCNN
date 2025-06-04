import os, cv2, time, requests

# ─── 環境參數 ───
SERVER   = os.getenv("SERVER_URL",  "http://172.20.10.11:8080")
TOKEN    = os.getenv("INGEST_TOKEN", "changeme")
CAM_ID   = int(os.getenv("CAM_ID", 0))        # 通常第一隻鏡頭就是 0
INTERVAL = float(os.getenv("INTERVAL", 0.04))  # 25 fps

# ─── 攝像頭嘗試用 DirectShow (CAP_DSHOW) ───
# 可改成 CAP_MSMF, CAP_DSHOW, CAP_VFW 等
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError(f"無法開啟攝像頭 (index={CAM_ID})，請確認：\n"
                       "  1. 沒有其他程式在用相機\n"
                       "  2. CAMERA ID 是否正確（可試 0,1,2 ...）\n"
                       "  3. 驅動程式是否安裝正常")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type":  "image/jpeg"
}

print(f"[INFO] 開始擷取攝像頭 (index={CAM_ID})，目標伺服器：{SERVER}/upload_frame")
while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARNING] 無法抓到 frame，重試中...")
        time.sleep(0.1)
        continue

    # 80% JPEG 品質
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    try:
        # 把 JPEG bytes 直接 POST 到 /upload_frame
        resp = requests.post(f"{SERVER}/upload_frame",
                             data=buf.tobytes(),
                             headers=headers,
                             timeout=5)
        if resp.status_code != 200:
            print(f"[ERROR] 上傳失敗 (HTTP {resp.status_code}): {resp.text}")
    except requests.exceptions.RequestException as e:
        # 網路連線有問題，就印訊息，但不讓程式停掉
        print(f"[WARNING] 網路傳送失敗：{e}")

    time.sleep(INTERVAL)