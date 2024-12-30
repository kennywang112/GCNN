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


def process_files(directory, label, image_dir, data_dict, key):
    
    num_node_features = 21
    
    adjacency_files = set(f.replace('adjacency_matrix_', '').replace('.jpg.csv', '') for f in os.listdir(directory) if f.endswith('.csv'))
    image_files = set(f.replace('.jpg', '') for f in os.listdir(os.path.join(image_dir, key)) if f.endswith('.jpg'))

    # Image with no adj unique_id
    missing_adj = image_files - adjacency_files
    
    for unique_id in image_files:
        image_path = os.path.join(image_dir, key, f'{unique_id}.jpg')
        file_path = os.path.join(directory, f'adjacency_matrix_{unique_id}.jpg.csv')

        if unique_id in missing_adj:
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
            # Image with adjacency matrix
            data = process_adj_matrix(file_path, image_path)
            data.y = torch.tensor([label])
            data_dict[key].append(data)
            
        else:
            print(f"Unexpected missing file for {unique_id}")
            
    return data_dict