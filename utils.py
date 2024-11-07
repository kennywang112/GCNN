import torch
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, Batch
from torchvision import transforms
from PIL import Image

# def process_adj_matrix(file_path):
#     # 2. 確保 adj_matrix 是 NxN 的矩陣
#     adj_matrix = np.loadtxt(file_path, delimiter=',', skiprows=1)
#     N = adj_matrix.shape[0]
#     assert adj_matrix.shape == (N, N), "矩陣必須是 NxN 的形狀"

#     # 3. Self-loop
#     adj_matrix_with_self_loop = adj_matrix + np.eye(N)

#     # 4. D
#     degree_matrix = np.diag(adj_matrix_with_self_loop.sum(axis=1))
#     # 5. normalize adjacency matrix  A^
#     # D^(-1/2)
#     degree_matrix_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))

#     # A^ = D^(-1/2) * A * D^(-1/2)
#     adj_matrix_normalized = degree_matrix_inv_sqrt @ adj_matrix_with_self_loop @ degree_matrix_inv_sqrt

#     # 6. 轉換為edge_index和edge_weight
#     adj_sparse = coo_matrix(adj_matrix_normalized)
#     edge_index = np.vstack((adj_sparse.row, adj_sparse.col))
#     edge_weight = adj_sparse.data

#     edge_index = torch.tensor(edge_index, dtype=torch.long)
#     edge_weight = torch.tensor(edge_weight, dtype=torch.float)

#     X = torch.eye(N)
#     # 有特徵的話改成 X = torch.tensor(your_feature_matrix, dtype=torch.float)

#     return Data(x=X, edge_index=edge_index, edge_weight=edge_weight)


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