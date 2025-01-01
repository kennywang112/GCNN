from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import numpy as np
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from models import Net_Alex
from utils_app import get_adjacency_matrix, process_for_model, visualize_adjacency_matrix
import torch.nn.functional as F

app = Flask(__name__)

# 初始化 MediaPipe 和 PyTorch 模型
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)

label_map = {
    0: "Surprised",
    1: "Fearful",
    2: "Disgusted",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral"
}

model = Net_Alex(hidden_channels=64, num_node_features=21)
model.load_state_dict(torch.load('./model/trained_model.pth'))
model.eval()
model.to(device)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./model/face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE
)

camera = cv2.VideoCapture(0)
camera_active = True
display_visualization = False  # 用於控制是否顯示視覺化
latest_probabilities = None

def generate_frames():
    global camera_active, display_visualization, latest_probabilities, latest_detection
    print(f"Display Visualization: {display_visualization}") 
    last_detection_time = 0  # 初始化上一次檢測的時間
    detection_interval = 0.2   # 設定檢測間隔為 2 秒

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            if camera_active:
                success, frame = camera.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape

                # 獲取當前時間
                current_time = time.time()

                # 判斷是否到達檢測間隔
                if current_time - last_detection_time >= detection_interval:
                    last_detection_time = current_time  # 更新上一次檢測時間

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    detection_result = landmarker.detect(mp_image)

                    if detection_result.face_landmarks:
                        for face_landmarks in detection_result.face_landmarks:
                            face_points = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks])
                            x_min, y_min = np.min(face_points, axis=0)
                            x_max, y_max = np.max(face_points, axis=0)
                            padding = 20
                            x_min, y_min = max(0, int(x_min) - padding), max(0, int(y_min) - padding)
                            x_max, y_max = min(w, int(x_max) + padding), min(h, int(y_max) + padding)

                            # 更新全局檢測結果
                            latest_detection = {
                                'face_landmarks': face_landmarks,
                                'x_min': x_min, 'y_min': y_min,
                                'x_max': x_max, 'y_max': y_max,
                            }

                            face_roi = frame[y_min:y_max, x_min:x_max]
                            adjacency_matrix, node_features = get_adjacency_matrix(face_landmarks)
                            x, edge_index, edge_weight, batch, image_tensor = process_for_model(
                                face_roi, adjacency_matrix, node_features
                            )
                            with torch.no_grad():
                                out = model(x, edge_index, edge_weight, batch, image_tensor)
                                probabilities = F.softmax(out, dim=1)  # dim=1 表示按類別進行操作
                                latest_probabilities = probabilities.squeeze().tolist()  # 更新全局變量
                                pred = out.max(dim=1)[1].item()

                                for idx, prob in enumerate(latest_probabilities):
                                    class_name = label_map[idx]

                # 使用最新檢測結果繪製
                if latest_detection is not None:
                    face_landmarks = latest_detection['face_landmarks']
                    x_min, y_min = latest_detection['x_min'], latest_detection['y_min']
                    x_max, y_max = latest_detection['x_max'], latest_detection['y_max']

                    predicted_label = label_map[pred]
                    cv2.putText(frame, f"Emotion: {predicted_label}",
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    if display_visualization:
                        matrix_vis = visualize_adjacency_matrix(frame, adjacency_matrix, face_landmarks, w, h)
                        matrix_h, matrix_w = matrix_vis.shape[:2]
                        frame[0:matrix_h, 0:matrix_w] = matrix_vis
                        cv2.putText(frame, "Visualization ON", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 將幀編碼並返回
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return jsonify({'camera_active': camera_active})

@app.route('/toggle_visualization', methods=['POST'])
def toggle_visualization():
    global display_visualization
    display_visualization = not display_visualization
    return jsonify({'display_visualization': display_visualization})

@app.route('/get_probabilities', methods=['GET'])
def get_probabilities():
    global latest_probabilities
    if latest_probabilities is None:
        return jsonify({'error': 'No probabilities available yet'}), 400

    probabilities_with_labels = [
        {"label": label_map[idx], "probability": prob * 100}
        for idx, prob in enumerate(latest_probabilities)
    ]
    return jsonify({'probabilities': probabilities_with_labels})

if __name__ == '__main__':
    app.run(debug=True)
