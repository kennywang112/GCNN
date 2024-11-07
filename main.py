import numpy as np
import os
import torch
from model import *
from torch_geometric.data import Data, DataLoader
from scipy.sparse import coo_matrix
from utils import *

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 設定資料夾路徑
fear_dir = '/output_data/adjacency/adjacency_1'
anger_dir = '/output_data/adjacency/adjacency_2'
happy_dir = '/output_data/adjacency/adjacency_2'

data_list = []

for filename in os.listdir(fear_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(fear_dir, filename)
        
        unique_id = filename.replace('adjacency_matrix_', '').replace('.csv', '')
        
        data = process_adj_matrix(file_path)
        data.y = torch.tensor([0])
        data_list.append(data)

for filename in os.listdir(anger_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(anger_dir, filename)
        
        unique_id = filename.replace('adjacency_matrix_', '').replace('.csv', '')
        
        data = process_adj_matrix(file_path)
        data.y = torch.tensor([1])
        data_list.append(data)

for filename in os.listdir(happy_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(happy_dir, filename)
        
        unique_id = filename.replace('adjacency_matrix_', '').replace('.csv', '')
        
        data = process_adj_matrix(file_path)
        data.y = torch.tensor([2])
        data_list.append(data)

print(f"Total data samples: {len(data_list)}")

print(data_list[0])

# 使用 DataLoader 進行批量處理
batch_size = 64
loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# 訓練模型
num_node_features = data_list[0].x.shape[1]  # 節點特徵數
hidden_channels = 64

model = Net_KAN(num_node_features, hidden_channels).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    for batch in loader:

        batch.x = batch.x.to(device)
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_weight = batch.edge_weight.to(device)
        batch.batch = batch.batch.to(device)
        batch.y = batch.y.to(device)
    
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 計算當前批次的準確率
        _, pred = out.max(dim=1)  # 取得每個樣本的預測類別
        correct += (pred == batch.y).sum().item()  # 計算正確預測的數量

    # 計算並打印當前 epoch 的平均損失和準確率
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)  # 整體準確率
    print(f'Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')