import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import pandas as pd
import torch
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform

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
model = torch.load('./1_25_70.pth')
model.eval()
model.to(device)

# 初始化 MediaPipe Face Mesh
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./face_landmarker.task'),
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

# 創建FaceLandmarker對象
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
                label_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", 6: "7"}
                predicted_label = label_map[pred]
                cv2.putText(image, f"Predicted: {predicted_label}", 
                           (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 255, 0), 2)
                
                # 繪製邊界框
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        cv2.imshow('Face Detection', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()