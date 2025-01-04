import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, Batch

def count_valid_files(directory):
    return sum(1 for filename in os.listdir(directory) if not filename.startswith('.') and os.path.isfile(os.path.join(directory, filename)))

def read_and_process_csv(file_path):
    # Placeholder function to read CSV and process into graph data
    # This should return edge_index, edge_weight, and node features (x)
    # Replace with your own implementation
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    x = torch.rand((2, 21))  # Assume each node has 21 features
    return edge_index, edge_weight, x

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize as per ImageNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')  # Ensure RGB mode
    return transform(image).unsqueeze(0)


def process_adj_matrix(file_path, image_path):
    edge_index, edge_weight, x = read_and_process_csv(file_path)
    image_features = process_image(image_path)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, image_features=image_features)
    return data


def custom_collate(data_list):
    batch = Batch.from_data_list(data_list)
    batch.image_features = torch.stack([data.image_features for data in data_list])  # Stack manually to keep shape
    return batch

def process_files(directory, label, image_dir, data_dict, key, num_node_features):
    
    adjacency_files = set(f.replace('adjacency_matrix_', '').replace('.jpg.csv', '') for f in os.listdir(directory) if f.endswith('.csv'))
    image_files = set(f.replace('.jpg', '') for f in os.listdir(os.path.join(image_dir, key)) if f.endswith('.jpg'))

    # Image with no adj unique_id
    missing_adj = image_files - adjacency_files
    
    for unique_id in image_files:
        image_path = os.path.join(image_dir, key, f'{unique_id}.jpg')
        file_path = os.path.join(directory, f'adjacency_matrix_{unique_id}.jpg.csv')

        if unique_id in missing_adj:
            print(unique_id)
            # No adj, use CNN only
            image_data = process_image(image_path)
            
            # for handling model error
            placeholder_x = torch.zeros(1, num_node_features)
            placeholder_index = torch.empty((2, 0), dtype=torch.long)
            placeholder_weight = torch.empty((0,), dtype=torch.float)
            
            data = Data(x=placeholder_x, edge_index=placeholder_index, edge_weight=placeholder_weight, image_features=image_data)
            data.y = torch.tensor([label])
            data_dict[key].append(data)
            
        elif os.path.exists(file_path) and os.path.exists(image_path):
            print(file_path)
            # Image with adjacency matrix
            data = process_adj_matrix(file_path, image_path)
            data.y = torch.tensor([label])
            data_dict[key].append(data)
            
        else:
            print(f"Unexpected missing file for {unique_id}")
            
    return data_dict

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