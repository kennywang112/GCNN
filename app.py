from flask import Flask, Response, render_template, jsonify, send_file, make_response, request

import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from models import Net_Alex, Net_ResNet18, NetWrapper
from utils.utils_app import get_adjacency_matrix, process_for_model, visualize_adjacency_matrix

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

app = Flask(__name__)

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

model_alexnet_gnn = Net_Alex(hidden_channels=64, num_node_features=21)
model_alexnet_gnn.load_state_dict(torch.load('./model/model_Net_Alex.pth', map_location=torch.device(device)))
model_alexnet_gnn.eval()
model_alexnet_gnn.to(device)

model_resnet_gnn = Net_ResNet18(hidden_channels=64, num_node_features=21)
model_resnet_gnn.load_state_dict(torch.load('./model/model_Net_Resnet.pth', map_location=torch.device(device)))
model_resnet_gnn.eval()
model_resnet_gnn.to(device)

# Available models
target_layers_map = {
    "alexnet_gnn": model_alexnet_gnn,
    "resnet18_gnn": model_resnet_gnn
}
current_model_name = "alexnet_gnn"
current_model = model_alexnet_gnn

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./model/face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE
)

camera = cv2.VideoCapture(0)
print(f"Camera Opened: {camera.isOpened()}")

if not camera.isOpened():
    print("Warning: Camera not available. The application will still run.")
    camera_active = False
else:
    camera_active = True

display_visualization = False  # 用於控制是否顯示視覺化
latest_probabilities = None
visualization_bgr = None
latest_detection = None

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_frames():
    global camera_active, display_visualization, latest_probabilities, latest_detection, visualization_bgr
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
                                out = current_model(x, edge_index, edge_weight, batch, image_tensor)
                                logits = out[0] if isinstance(out, tuple) else out 
                                probabilities = F.softmax(logits, dim=1)  # dim=1 表示按類別進行操作
                                latest_probabilities = probabilities.squeeze().tolist()  # 更新全局變量
                                pred = logits.max(dim=1)[1].item()
                                
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

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global current_model, current_model_name
    data = request.get_json()
    selected_model = data.get('model')

    if selected_model in target_layers_map:
        current_model = target_layers_map[selected_model]
        current_model_name = selected_model
        print(f"Model switched to: {current_model_name}")
        return jsonify({'success': True, 'message': f'Model switched to {selected_model}'})
    else:
        print(f"Invalid model selected: {selected_model}")
        return jsonify({'success': False, 'message': 'Invalid model selected'}), 400


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


@app.route('/generate_grad_cam', methods=['POST'])
def generate_grad_cam():
    global current_model, latest_detection, visualization_bgr

    if latest_detection is None:
        return jsonify({'error': 'No face detected'}), 400

    try:
        face_landmarks = latest_detection['face_landmarks']
        x_min, y_min = latest_detection['x_min'], latest_detection['y_min']
        x_max, y_max = latest_detection['x_max'], latest_detection['y_max']

        _, frame = camera.read()
        face_roi = frame[y_min:y_max, x_min:x_max]

        if face_roi.size == 0:
            return jsonify({'error': 'Invalid face region'}), 400

        adjacency_matrix, node_features = get_adjacency_matrix(face_landmarks)
        x, edge_index, edge_weight, batch, image_tensor = process_for_model(
            face_roi, adjacency_matrix, node_features
        )

        rgb_img = cv2.resize(face_roi, (224, 224)) / 255.0
        rgb_img = rgb_img.astype(np.float32)
        rgb_img_pil = Image.fromarray(cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

        input_tensor = transform(rgb_img_pil).unsqueeze(0).to(device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        input_tensor = input_tensor * std + mean

        wrapped_model = NetWrapper(current_model, edge_index, edge_weight, batch)
        target_layers = [wrapped_model.model.alexnet.features[-1]]
        cam = GradCAM(model=wrapped_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor)

        visualization = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)
        visualization_bgr = (visualization * 255).astype(np.uint8)

        return jsonify({'message': 'Grad-CAM generated successfully'})
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return jsonify({'error': f'Failed to generate Grad-CAM: {str(e)}'}), 500

@app.route('/grad_cam_feed')
def grad_cam_feed():
    global visualization_bgr
    if visualization_bgr is None:
        return "No Grad-CAM available", 404
    _, buffer = cv2.imencode('.jpg', visualization_bgr)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    return response

file_path_train_acc = './mlflow/train_accuracy.csv'
file_path_train_loss = './mlflow/train_loss.csv'
file_path_val_correct = './mlflow/val_correct.csv'
file_path_val_loss = './mlflow/val_loss.csv'
file_path_model_performance = './mlflow/train_accuracy-val_correct-lr-model-num_epochs.csv'

# 读取数据
train_data_acc = pd.read_csv(file_path_train_acc)
train_data_loss = pd.read_csv(file_path_train_loss)
val_data_correct = pd.read_csv(file_path_val_correct)
val_data_loss = pd.read_csv(file_path_val_loss)
model_performance_data = pd.read_csv(file_path_model_performance)

@app.route('/get_train_accuracy', methods=['GET'])
def get_train_accuracy():
    response = []
    for run, group in train_data_acc.groupby('Run'):
        response.append({
            "run": run,
            "steps": group['step'].tolist(),
            "values": group['value'].tolist()
        })
    return jsonify(response)

@app.route('/get_train_loss', methods=['GET'])
def get_train_loss():
    response = []
    for run, group in train_data_loss.groupby('Run'):
        response.append({
            "run": run,
            "steps": group['step'].tolist(),
            "values": group['value'].tolist()
        })
    return jsonify(response)

@app.route('/get_val_correct', methods=['GET'])
def get_val_correct():
    response = []
    for run, group in val_data_correct.groupby('Run'):
        response.append({
            "run": run,
            "steps": group['step'].tolist(),
            "values": group['value'].tolist()
        })
    return jsonify(response)

@app.route('/get_val_loss', methods=['GET'])
def get_val_loss():
    response = []
    for run, group in val_data_loss.groupby('Run'):
        response.append({
            "run": run,
            "steps": group['step'].tolist(),
            "values": group['value'].tolist()
        })
    return jsonify(response)

@app.route('/get_model_performance', methods=['GET'])
def get_model_performance():
    data = {
        "Run": ["Restnet18+GNN", "Resnet18", "Alexnet+GNN", "Alexnet"],
        "train_accuracy": [0.9906, 0.9917, 0.9209, 0.8776],
        "val_correct": [0.7874, 0.7796, 0.7377, 0.7324],
        "lr": [0.001, 0.001, 0.001, 0.001],
        "num_epochs": [25, 25, 25, 25],
    }
    df = pd.DataFrame(data)
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)