from flask import Flask, Response, render_template, jsonify, send_file, make_response, request
import traceback
import os
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from models import Net_Alex, Net_ResNet18, Net_VGG, NetWrapper
from utils.utils_app import get_adjacency_matrix, process_for_model, visualize_adjacency_matrix, upload_emotion_log_to_cosmos

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from dotenv import load_dotenv
import base64
import threading

load_dotenv()

endpoint = os.getenv('COSMOS_ENDPOINT')
key = os.getenv('COSMOS_KEY')
database_name = os.getenv('COSMOS_DATABASE')
container_name = os.getenv('COSMOS_CONTAINER')

state_lock = threading.Lock()

state = {
    "frame_vis": None,          # 給 /video_feed 串流
    "latest_prob": None,        # list[float] 0–1
    "predicted_label": None,    # str
    "latest_detection": None,   # {'face_landmarks': lm, 'bbox': (x0,y0,x1,y1)}
    "last_face_roi": None,      # numpy array (BGR) for Grad-CAM
    "grad_cam_vis": None        # numpy array (BGR) 最後一次產生的 Grad-CAM
}

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
hidden_channels=64
num_node_features=21

# Azure空間問題暫時不載入其他模型
# model_alexnet_gnn = Net_Alex(num_node_features, hidden_channels)
# model_alexnet_gnn.load_state_dict(torch.load('./model/model_Net_Alex.pth', map_location=torch.device(device)))
# model_alexnet_gnn.eval()
# model_alexnet_gnn.to(device)

# model_resnet_gnn = Net_ResNet18(num_node_features, hidden_channels)
# model_resnet_gnn.load_state_dict(torch.load('./model/model_Net_Resnet.pth', map_location=torch.device(device)))
# model_resnet_gnn.eval()z
# model_resnet_gnn.to(device)

model_vgg_gnn = Net_VGG(num_node_features, hidden_channels)
model_vgg_gnn.load_state_dict(torch.load('./model/model_Net_VGG.pth', map_location=torch.device(device)))
model_vgg_gnn.eval()
model_vgg_gnn.to(device)

model_alexnet_gnn = None
model_resnet_gnn = None

# Available models
target_layers_map = {
    "alexnet_gnn": model_alexnet_gnn,
    "resnet18_gnn": model_resnet_gnn,
    "vgg16_gnn": model_vgg_gnn
}
current_model_name = "vgg16_gnn"
current_model = model_vgg_gnn

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./model/face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE
)

camera_active = False
frame = None         # ← 最新影格會存這裡
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "changeme")   # ← 新增權杖
display_visualization = False  # 用於控制是否顯示視覺化
latest_probabilities = None
visualization_bgr = None
latest_detection = None
camera = None
frame = None

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

landmarker = FaceLandmarker.create_from_options(options)
import atexit, sys, traceback
atexit.register(landmarker.close)

@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    try:
        # 1. 讀取並解碼
        img_bytes = request.get_data()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "Bad image", 400

        h, w = img.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = landmarker.detect(mp_img)

        if not detection_result.face_landmarks:
            return jsonify({"error": "No face detected"}), 400

        # 2. 取第一張臉，計算 ROI
        lm  = detection_result.face_landmarks[0]
        pts = np.array([(p.x*w, p.y*h) for p in lm])
        x0, y0 = pts.min(axis=0).astype(int) - 20
        x1, y1 = pts.max(axis=0).astype(int) + 20
        x0, y0 = np.clip([x0, y0], 0, [w, h])
        x1, y1 = np.clip([x1, y1], 0, [w, h])
        roi = img[y0:y1, x0:x1]

        # 3. 前處理 + 推論
        A, X = get_adjacency_matrix(lm)
        x, ei, ew, batch, img_tensor = process_for_model(roi, A, X)

        with torch.no_grad():
            out    = current_model(x, ei, ew, batch, img_tensor)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs  = torch.softmax(logits, dim=1).squeeze()   # Tensor
            pred   = int(torch.argmax(probs).item())
            label  = label_map[pred]
            probs  = probs.cpu().tolist()                     # → list[float]

        # 4. 畫框＋文字
        vis = img.copy()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(vis, label, (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 5. 寫入全域 state
        with state_lock:
            state["frame_vis"]        = vis
            state["latest_prob"]      = probs
            state["predicted_label"]  = label
            state["latest_detection"] = {"face_landmarks": lm,
                                         "bbox": (x0, y0, x1, y1)}
            state["last_face_roi"]    = roi.copy()

        prob_out = [{"label": label_map[i], "probability": round(p*100, 2)}
                    for i, p in enumerate(probs)]
        return jsonify({"predicted_label": label,
                        "probabilities": prob_out})

    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500



def generate_frames():
    """僅串流 state['frame_vis']，不再重跑推論"""
    while True:
        with state_lock:
            frame_vis = state["frame_vis"]
        if frame_vis is None:
            time.sleep(0.05)
            continue

        _, buffer = cv2.imencode('.jpg', frame_vis)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')
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
    global camera_active
    if not camera_active:
        return jsonify({'error': 'Camera is not active'}), 400
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
    with state_lock:
        probs = state["latest_prob"]
    if probs is None:
        return jsonify({'error': 'No probabilities available yet'}), 400

    data = [{"label": label_map[i], "probability": p*100} for i, p in enumerate(probs)]
    return jsonify({'probabilities': data})
    
@app.route('/generate_grad_cam', methods=['POST'])
def generate_grad_cam():
    try:
        with state_lock:
            det  = state["latest_detection"]
            roi  = state["last_face_roi"]

        if det is None or roi is None:
            return jsonify({'error': 'No face detected yet'}), 400

        lm = det['face_landmarks']
        A, X = get_adjacency_matrix(lm)
        x, ei, ew, batch, img_tensor = process_for_model(roi, A, X)

        rgb = cv2.resize(roi, (224, 224)) / 255.0
        rgb = rgb.astype(np.float32)
        input_tensor = transform(Image.fromarray(
            cv2.cvtColor((rgb*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        )).unsqueeze(0).to(device)

        wrapped = NetWrapper(current_model, ei, ew, batch)
        if current_model_name == 'alexnet_gnn':
            layers = [wrapped.model.alexnet.features[-1]]
        elif current_model_name == 'resnet18_gnn':
            layers = [wrapped.model.resnet.layer4[-1]]
        else:  # vgg16_gnn
            layers = [wrapped.model.vgg16.features[-1]]

        cam = GradCAM(model=wrapped, target_layers=layers)
        grayscale_cam = cam(input_tensor=input_tensor)[0]

        vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
        vis_bgr = (vis * 255).astype(np.uint8)

        with state_lock:
            state["grad_cam_vis"] = vis_bgr

        return jsonify({'message': 'Grad-CAM generated'})
    except Exception as e:
        import sys, traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/grad_cam_feed')
def grad_cam_feed():
    with state_lock:
        vis = state["grad_cam_vis"]
    if vis is None:
        return "No Grad-CAM available", 404
    _, buf = cv2.imencode('.jpg', vis)
    return Response(buf.tobytes(), mimetype='image/jpeg')

file_path_train_acc = './mlflow/train_accuracy.csv'
file_path_train_loss = './mlflow/train_loss.csv'
file_path_val_correct = './mlflow/val_correct.csv'
file_path_val_loss = './mlflow/val_loss.csv'
# file_path_model_performance = './mlflow/train_accuracy-val_correct-lr-model-num_epochs.csv'

# 读取数据
train_data_acc = pd.read_csv(file_path_train_acc)
train_data_loss = pd.read_csv(file_path_train_loss)
val_data_correct = pd.read_csv(file_path_val_correct)
val_data_loss = pd.read_csv(file_path_val_loss)
# model_performance_data = pd.read_csv(file_path_model_performance)

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

last_upload_time = 0
upload_interval = 5  # 每5秒上傳一次，你可以改秒數
upload_thread_running = True

def background_upload_thread():
    global last_upload_time
    while upload_thread_running:
        time.sleep(1)  # 每1秒檢查一次
        with state_lock:
            predicted_label = state["predicted_label"]
        if predicted_label and (time.time() - last_upload_time) >= upload_interval:
            single_log = pd.DataFrame([{
                "id": str(int(time.time() * 1000)),  # 以時間戳記當ID
                "face": predicted_label
            }])
            try:
                upload_emotion_log_to_cosmos(single_log, endpoint, key, database_name, container_name)
                print(f"[UPLOAD] Uploaded to CosmosDB: {predicted_label}")
                last_upload_time = time.time()
            except Exception as e:
                print(f"[ERROR] Failed to upload to CosmosDB: {e}")

# 啟動 Background Thread
upload_thread = threading.Thread(target=background_upload_thread, daemon=True)
upload_thread.start()

# 程式結束時關掉 Thread
import atexit
def cleanup():
    global upload_thread_running
    upload_thread_running = False
    print("Shutting down background upload thread...")

atexit.register(cleanup)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)