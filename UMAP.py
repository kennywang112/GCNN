import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeoDataLoader

from models import *
from utils import *

device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)
print(f'Using device: {device}')

# Load data
data_dict_path = './output_data/data_dict.pth'
data_dict = torch.load(data_dict_path)
data_list = sum([data_dict[str(i)] for i in range(1, 8)], [])
total_size = len(data_list)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_data, val_data = random_split(data_list, [train_size, val_size])

# Create DataLoader
batch_size = 64
train_loader = GeoDataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = GeoDataLoader(val_data, batch_size=batch_size, shuffle=False)

# Model setup
num_node_features = next(data.x.shape[1] for data in data_list if data.x is not None)
hidden_channels = 64
model = Net_Alex(num_node_features, hidden_channels).to(device)
model.load_state_dict(torch.load('./model/trained_model.pth'))  # Load trained model
model.eval()

# Extract embeddings
print("Extracting embeddings...")
features_list = []
labels_list = []

with torch.no_grad():
    for batch in train_loader:
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        image_features = batch.image_features
        batch_y = batch.y

        # Extract embeddings
        embeddings = model.extract_features(x, edge_index, edge_weight, batch.batch, image_features)
        features_list.append(embeddings.cpu().numpy())
        labels_list.extend(batch_y.cpu().numpy())

# Combine features and labels
features = np.vstack(features_list)
labels = np.array(labels_list)

# Dimensionality reduction with UMAP
print("Performing UMAP dimensionality reduction...")
reducer = umap.UMAP()
low_dim_embeddings = reducer.fit_transform(features)

# Visualization
print("Visualizing data...")
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    low_dim_embeddings[:, 0], 
    low_dim_embeddings[:, 1], 
    c=labels, 
    cmap='tab10', 
    s=10
)
plt.colorbar(scatter, label='Class')
plt.title('UMAP Visualization of Feature Space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
