import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class Net_Alex(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(Net_Alex, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Pretrained AlexNet model
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(
            *list(self.alexnet.classifier.children())[:-1]  # Remove last classification layer
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels + 4096, 64),  # Combine GCN and AlexNet embeddings
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 7)  # 3 classes
        )

        self.fc_only_alex = nn.Sequential(
            nn.Linear(4096, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 7)
        )
        
    def forward(self, x, edge_index, edge_weight, batch, image_features):

        if x is not None and edge_index is not None and edge_index.numel() > 0:
            # GCN component
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            x = global_mean_pool(x, batch)
            # Combine GCN and AlexNet features
            x = torch.cat([x, self.alexnet(image_features)], dim=1)
            return self.fc_layers(x)
        
        else:
            # Only use AlexNet features
            x = self.alexnet(image_features)
            return self.fc_only_alex(x)
        
class KAN(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim=3):
        super(KAN, self).__init__()
        self.hidden_channels = hidden_channels
        
        # Define individual layers inspired by Kolmogorov-Arnold
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_channels * 2, output_dim)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(x)
        
        # Summing parts inspired by the Kolmogorov-Arnold structure
        x = torch.cat([h1, h2], dim=1)
        x = self.output_layer(x)
        return x


class Net_KAN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(Net_KAN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
    
        self.kan = KAN(hidden_channels, output_dim=3)
    
    def forward(self, x, edge_index, edge_weight, batch):
        # GCN 部分
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = global_mean_pool(x, batch)
        
        x = self.kan(x)
        return x


class Net(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
    
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )
    
    def forward(self, x, edge_index, edge_weight, batch):
        # GCN 部分
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        # 使用全局池化将节点嵌入转换为图嵌入
        x = global_mean_pool(x, batch)
        
        # 移除 CNN 部分（AlexNet）
        # image_features = self.alexnet(image_features)
        # x = torch.cat([x, image_features], dim=1)

        x = self.fc_layers(x)
        return x
