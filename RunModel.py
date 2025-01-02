import os
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeoDataLoader

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

import mlflow

from train_model_utils import *
from models import *
from utils import *

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)
print(f'Using device: {device}')

data_dict_path = './output_data/data_dict.pth'
data_dict = torch.load(data_dict_path)

print("Processing data")
print([len(data_dict[f'{i}']) for i in range(1, 8)])

data_list = sum([data_dict[str(i)] for i in range(1, 8)], [])
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
model_dict = get_model_list(device, num_node_features, hidden_channels)
# model = Net_Alex(num_node_features, hidden_channels).to(device)
criterion = nn.CrossEntropyLoss()
num_epochs = 25
print('Start training ...')
for i, (model_name, model) in enumerate(model_dict.items()):
    params = {
    "model": model_name,
    "num_epochs": 25,
    "lr":0.001,
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cumulative_preds = []
    cumulative_labels = []
    with mlflow.start_run():
        mlflow.log_params(params)
        for epoch in range(1, num_epochs+1):
            model.train()
            total_loss = 0
            correct = 0
            for batch in train_loader:
                batch  = batch.to(device)
                x = batch.x
                edge_index = batch.edge_index
                edge_weight = batch.edge_weight
                image_features = batch.image_features
                batch_y = batch.y

                optimizer.zero_grad()
                
                if i % 2 == 0:
                    out = model(image_features)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    _, pred = out.max(dim=1)
                    correct += (pred == batch_y).sum().item()
                else:
                    GNN_output, CNN_output = model(x, edge_index, edge_weight, batch.batch, image_features)
                    if GNN_output is not None:
                        loss_GNN = criterion(GNN_output, batch_y)
                        loss_AlexNet = criterion(CNN_output, batch_y)
                        loss = (loss_GNN + loss_AlexNet) / 2
                        loss.backward()
                        _, pred = GNN_output.max(dim=1)
                        correct += (pred == batch_y).sum().item()
                    else:
                        loss = criterion(CNN_output, batch_y)
                        loss.backward()
                        _, pred = CNN_output.max(dim=1)
                        correct += (pred == batch_y).sum().item()
                    optimizer.step()
                    total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            train_accuracy = correct / len(train_data)

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

                    if i % 2 == 0:
                        out = model(image_features)
                        loss = criterion(out, batch_y)
                        val_loss += loss.item()

                        _, pred = out.max(dim=1)
                        val_correct += (pred == batch_y).sum().item()
                    else:
                        GNN_output, AlexNet_output = model(x, edge_index, edge_weight, batch.batch, image_features)
                        if GNN_output is not None:
                            loss_GNN = criterion(GNN_output, batch_y)
                            loss_AlexNet = criterion(AlexNet_output, batch_y)
                            loss = (loss_GNN + loss_AlexNet) / 2
                            _, pred = GNN_output.max(dim=1)
                            val_correct += (pred == batch_y).sum().item()
                        else:
                            loss = criterion(AlexNet_output, batch_y)
                            _, pred = AlexNet_output.max(dim=1)
                            val_correct += (pred == batch_y).sum().item()
                        val_loss += loss.item()

                    epoch_preds.extend(pred.cpu().numpy())
                    epoch_labels.extend(batch_y.cpu().numpy())
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / len(val_data)
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('train_accuracy', train_accuracy, step=epoch)
            mlflow.log_metric('val_loss', avg_val_loss, step=epoch)
            mlflow.log_metric('val_correct', val_accuracy, step=epoch)
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    # Accumulate all predictions and labels across epochs
            cumulative_preds.extend(epoch_preds)
            cumulative_labels.extend(epoch_labels)
    
    save_path = f'./model/{model_name}.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    del model
    torch.cuda.empty_cache()