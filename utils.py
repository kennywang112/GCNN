import torch
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, Batch
from torchvision import transforms
from PIL import Image

def process_files(directory, label, image_dir, data_dict, key):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            unique_id = filename.replace('adjacency_matrix_', '').replace('.csv', '')
            image_path = os.path.join(image_dir, f'{key}/{unique_id}')
            data = process_adj_matrix(file_path, image_path)
            data.y = torch.tensor([label])
            data_dict[key].append(data)

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