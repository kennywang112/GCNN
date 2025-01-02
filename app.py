from flask import Flask, Response, render_template, jsonify, make_response
import io
import cv2
import time
import numpy as np
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

# 設備檢測
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# 標籤映射
label_map = {
    0: "Surprised",
    1: "Fearful",
    2: "Disgusted",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral"
}

# 模型加載
model_alexnet_gnn = Net_Alex(hidden_channels=64, num_node_features=21)
model_alexnet_gnn.load_state_dict(torch.load('./model/model_Net_Alex.pth'))
model_alexnet_gnn.eval()
model_alexnet_gnn.to(device)

# model_resnet_gnn = Net_ResNet18(hidden_channels=64, num_node_features=21)
# model_resnet_gnn.load_state_dict(torch.load('./model/model_Net_Restnet.pth'))
# model_resnet_gnn.eval()
# model_resnet_gnn.to(device)

# 可用模型映射
target_layers_map = {
    "alexnet_gnn": model_alexnet_gnn,
    "resnet_gnn": model_alexnet_gnn
}
current_model_name = "alexnet_gnn"
current_model = model_alexnet_gnn

# Mediapipe 配置
BaseOptions = mp.tasks.BaseOptions
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./model/face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE
)

# 初始化攝像頭
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Warning: Camera not available. The application will still run.")
    camera_active = False
else:
    camera_active = True

# 視覺化和其他全局變量
display_visualization = False
latest_probabilities = None
visualization_bgr = None
latest_detection = None

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def generate_frames():
    """
    視頻流生成器。如果攝像頭不可用，返回空白畫面。
    """
    global camera_active, latest_probabilities, visualization_bgr
    if not camera_active:
        while True:
            # 返回一個空白圖片作為佔位
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    else:
        while True:
            success, frame = camera.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)  # 翻轉攝像頭畫面
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    if not camera.isOpened():
        return jsonify({'error': 'Camera not available'}), 400
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

import pandas as pd
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
