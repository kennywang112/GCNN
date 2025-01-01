import cv2
import torch
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION

def get_adjacency_matrix(landmarks):
    points = [[lm.x, lm.y, lm.z] for lm in landmarks]
    points = np.array(points)
    adjacency_matrix = np.zeros((len(points), len(points)))
    for connection in face_mesh_connections:
        point1, point2 = connection
        if point1 < len(points) and point2 < len(points):
            distance = np.linalg.norm(points[point1] - points[point2])
            adjacency_matrix[point1][point2] = distance
            adjacency_matrix[point2][point1] = distance
    node_features = torch.rand((len(points), 21))
    return adjacency_matrix, node_features

def process_for_model(face_roi, adjacency_matrix, node_features):
    image_tensor = transform(face_roi).unsqueeze(0).to(device)
    edge_index, edge_weight = [], []
    rows, cols = np.nonzero(adjacency_matrix)
    for i, j in zip(rows, cols):
        if i < j:
            edge_index.append([i, j])
            edge_weight.append(adjacency_matrix[i][j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32).to(device)
    x = node_features.to(device)
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