import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import torch
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform

from models import Net_Alex

# 檢查是否支持 MPS
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)
print(f"使用設備: {device}")

# 載入預訓練模型
model = Net_Alex(hidden_channels=64, num_node_features=21)
model.load_state_dict(torch.load('./model/trained_model.pth'))
model.eval()
model.to(device)

# 初始化 MediaPipe Face Mesh
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./model/face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

face_mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION

# 初始化攝像頭
cap = cv2.VideoCapture(0)

# 定義圖像預處理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_adjacency_matrix(landmarks):
    # 提取特徵點座標（只用於計算距離）
    points = []
    for landmark in landmarks:
        points.append([landmark.x, landmark.y, landmark.z])
    points = np.array(points)
    
    # 初始化鄰接矩陣
    num_points = len(points)
    adjacency_matrix = np.zeros((num_points, num_points))
    
    # 根據FACEMESH_TESSELATION建立連接
    for connection in face_mesh_connections:
        point1, point2 = connection
        if point1 < num_points and point2 < num_points:
            distance = np.linalg.norm(points[point1] - points[point2])
            adjacency_matrix[point1][point2] = distance
            adjacency_matrix[point2][point1] = distance
    
    # 生成隨機特徵，與訓練時保持一致
    node_features = torch.rand((num_points, 21))  # 每個節點21維隨機特徵
    
    return adjacency_matrix, node_features

def process_for_model(face_roi, adjacency_matrix, node_features):
    # 處理圖像特徵
    image_tensor = transform(face_roi).unsqueeze(0).to(device)
    
    # 只使用有連接的邊
    edge_index = []
    edge_weight = []
    rows, cols = np.nonzero(adjacency_matrix)
    for i, j in zip(rows, cols):
        if i < j:  # 避免重複邊
            edge_index.append([i, j])
            edge_weight.append(adjacency_matrix[i][j])
    
    # 轉換為PyTorch張量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32).to(device)
    x = node_features.to(device)  # 直接使用生成的隨機特徵
    batch = torch.zeros(len(node_features), dtype=torch.long).to(device)
    
    return x, edge_index, edge_weight, batch, image_tensor

def visualize_adjacency_matrix(image, adjacency_matrix, face_landmarks, w, h):
    # 創建一個用於顯示鄰接矩陣的小視窗
    matrix_size = 200
    matrix_vis = np.zeros((matrix_size, matrix_size, 3), dtype=np.uint8)
    
    # 將鄰接矩陣歸一化並轉換為可視化圖像
    adj_normalized = (adjacency_matrix - adjacency_matrix.min()) / (adjacency_matrix.max() - adjacency_matrix.min())
    adj_resized = cv2.resize(adj_normalized, (matrix_size, matrix_size))
    matrix_vis = (adj_resized * 255).astype(np.uint8)
    matrix_vis = cv2.cvtColor(matrix_vis, cv2.COLOR_GRAY2BGR)
    
    # 在原始圖像上繪製特徵點之間的連接
    for connection in face_mesh_connections:
        point1, point2 = connection
        if point1 < len(face_landmarks) and point2 < len(face_landmarks):
            pt1 = (int(face_landmarks[point1].x * w), int(face_landmarks[point1].y * h))
            pt2 = (int(face_landmarks[point2].x * w), int(face_landmarks[point2].y * h))
            cv2.line(image, pt1, pt2, (0, 255, 0), 1)
    
    # 在原始圖像上繪製特徵點
    for landmark in face_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
    return matrix_vis

# 在主循環之前添加顯示控制變量
display_visualization = False

# 修改主循環部分
with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        success, image = cap.read()
        if not success:
            print("無法讀取攝像頭畫面")
            continue

        # 將圖像轉換為MediaPipe格式
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # 進行人臉檢測
        detection_result = landmarker.detect(mp_image)
        
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                # 獲取臉部邊界框
                h, w, c = image.shape
                face_points = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks])
                x_min, y_min = np.min(face_points, axis=0)
                x_max, y_max = np.max(face_points, axis=0)
                
                # 擴大邊界框範圍
                padding = 20
                x_min = max(0, int(x_min) - padding)
                y_min = max(0, int(y_min) - padding)
                x_max = min(w, int(x_max) + padding)
                y_max = min(h, int(y_max) + padding)
                
                # 提取臉部區域
                face_roi = image[y_min:y_max, x_min:x_max]
                
                # 獲取鄰接矩陣和特徵點
                adjacency_matrix, node_features = get_adjacency_matrix(face_landmarks)
                
                # 準備模型輸入
                x, edge_index, edge_weight, batch, image_tensor = process_for_model(
                    face_roi, adjacency_matrix, node_features
                )
                
                # 進行預測
                with torch.no_grad():
                    out = model(x, edge_index, edge_weight, batch, image_tensor)
                    pred = out.max(dim=1)[1].item()
                
                # 顯示預測結果
                label_map = {0: "Suprised", 1: "Fearful", 2: "Disgusted", 3: "Happy", 4: "Sad", 5: "Angry", 6: "Neutral"}
                predicted_label = label_map[pred]
                cv2.putText(image, f"Predicted: {predicted_label}", 
                           (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 255, 0), 2)
                
                # 繪製邊界框
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # 只在display_visualization為True時顯示視覺化
                if display_visualization:
                    matrix_vis = visualize_adjacency_matrix(image, adjacency_matrix, face_landmarks, w, h)
                    matrix_h, matrix_w = matrix_vis.shape[:2]
                    image[0:matrix_h, 0:matrix_w] = matrix_vis
        
        # 添加顯示狀態文字
        status_text = "Visualization: ON" if display_visualization else "Visualization: OFF"
        cv2.putText(image, status_text, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):  # 使用'v'鍵切換視覺化顯示
            display_visualization = not display_visualization

    cap.release()
    cv2.destroyAllWindows()