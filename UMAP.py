import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import umap.umap_ as umap

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeoDataLoader

from models import *
from utils.utils_preprocess import *

label_map = {
    0: "Surprised",
    1: "Fearful",
    2: "Disgusted",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral"
}

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
model.load_state_dict(torch.load('./model/trained_model.pth'))
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

        # 將節點特徵、邊信息、圖像特徵輸入到模型中
        embeddings = model.extract_features(x, edge_index, edge_weight, batch.batch, image_features)
        # 每個批次提取的嵌入特徵
        features_list.append(embeddings.cpu().numpy())
        labels_list.extend(batch_y.cpu().numpy())

# Combine features and labels
features = np.vstack(features_list)
labels = np.array(labels_list)

print(features.shape)
# # Dimensionality reduction with UMAP
# print("Performing UMAP dimensionality reduction...")
# reducer = umap.UMAP()
# low_dim_embeddings = reducer.fit_transform(features)

# # Visualization
# print("Visualizing data...")
# plt.figure(figsize=(12, 10))

# # Create scatter plot
# scatter = plt.scatter(
#     low_dim_embeddings[:, 0], 
#     low_dim_embeddings[:, 1], 
#     c=labels, 
#     cmap='tab10', 
#     s=40, 
#     edgecolor='k', 
#     alpha=0.8
# )

# # Create a legend instead of a colorbar
# unique_labels = np.unique(labels)
# handles = [
#     mpatches.Patch(color=scatter.cmap(scatter.norm(label)), label=label_map[label]) 
#     for label in unique_labels
# ]
# plt.legend(
#     handles=handles, 
#     title="Emotion Categories", 
#     loc='upper right', 
#     fontsize=10, 
#     title_fontsize=12
# )

# # Add titles and labels
# plt.xlabel('Dimension 1', fontsize=14)
# plt.ylabel('Dimension 2', fontsize=14)

# # Customize grid and background
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.gca().set_facecolor('#f7f7f7')

# # Save the plot
# output_dir = './UMAPplots'
# os.makedirs(output_dir, exist_ok=True)
# plt.savefig(f'{output_dir}/umap_alex_gnn.png', dpi=300, bbox_inches='tight')