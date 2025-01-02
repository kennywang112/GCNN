import os
import torch
import random
import numpy as np
from scipy.sparse import coo_matrix
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

from models import *
from utils.utils_preprocess import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

# Set directory paths
dir_1 = './output_data/adjacency/adjacency_1'
dir_2 = './output_data/adjacency/adjacency_2'
dir_3 = './output_data/adjacency/adjacency_3'
dir_4 = './output_data/adjacency/adjacency_4'
dir_5 = './output_data/adjacency/adjacency_5'
dir_6 = './output_data/adjacency/adjacency_6'
dir_7 = './output_data/adjacency/adjacency_7'
image_dir = './Image_data/DATASET/train'
dir_label_map = {
    '1': (dir_1, 0),
    '2': (dir_2, 1),
    '3': (dir_3, 2),
    '4': (dir_4, 3),
    '5': (dir_5, 4),
    '6': (dir_6, 5),
    '7': (dir_7, 6),
}
data_dict = {str(i): [] for i in range(1, 8)}

for key, (directory, label) in dir_label_map.items():
    process_files(directory, label, image_dir, data_dict, key)
print('File processed')


# Data setting
data_list = data_dict['1'] + data_dict['2'] + data_dict['3'] + data_dict['4'] + data_dict['5'] + data_dict['6'] + data_dict['7']
total_size = len(data_list)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_data, val_data = random_split(data_list, [train_size, val_size])


# Step 2: Create DataLoader for training and validation
batch_size = 64
train_loader = GeoDataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = GeoDataLoader(val_data, batch_size=batch_size, shuffle=False)


# Step 3: Model setup
# Some of the datas don't have adj matrix
num_node_features = next(data.x.shape[1] for data in data_list if data.x is not None)
hidden_channels = 64
num_epochs = 20
model = Net_Alex(num_node_features, hidden_channels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(1, num_epochs + 1):
    # Initialize epoch tracking variables
    model.train()
    total_loss = 0
    correct = 0

    # Training phase
    for batch in train_loader:
        batch = batch.to(device)

        x = batch.x
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        image_features = batch.image_features
        batch_y = batch.y

        optimizer.zero_grad()
        
        out = model(x, edge_index, edge_weight, batch.batch, image_features)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, pred = out.max(dim=1)
        correct += (pred == batch_y).sum().item()

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / len(train_data)

    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    epoch_preds = []
    epoch_labels = []
    
    with torch.no_grad():
        
        for batch in val_loader:
            batch = batch.to(device)

            x = batch.x
            edge_index = batch.edge_index
            edge_weight = batch.edge_weight
            image_features = batch.image_features
            batch_y = batch.y

            out = model(x, edge_index, edge_weight, batch.batch, image_features)
            loss = criterion(out, batch_y)
            val_loss += loss.item()

            _, pred = out.max(dim=1)
            val_correct += (pred == batch_y).sum().item()

            # Collect predictions and labels for this epoch
            epoch_preds.extend(pred.cpu().numpy())
            epoch_labels.extend(batch_y.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / len(val_data)

    print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
            f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')