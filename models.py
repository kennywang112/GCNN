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
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # Pretrained AlexNet model
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(
            *list(self.alexnet.classifier.children())[:-1]  # Remove last classification layer
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels + 4096, 256),  # Combine GCN and AlexNet embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 7)  # 7 classes
        )

        self.fc_only_alex = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 7)
        )
        
    def forward(self, x, edge_index, edge_weight, batch, image_features):

        if x is not None and edge_index is not None and edge_index.numel() > 0:
            # GCN component
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.conv3(x, edge_index, edge_weight=edge_weight)
            x = global_mean_pool(x, batch)
            # Combine GCN and AlexNet features
            x = torch.cat([x, self.alexnet(image_features)], dim=1)
            GNN_output = self.fc_layers(x)
            # Only use AlexNet features
            x = self.alexnet(image_features)
            return GNN_output, self.fc_only_alex(x)
        else:
            x = self.alexnet(image_features)
            return None, self.fc_only_alex(x)
        
class Net_VGG(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(Net_VGG, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19.classifier = nn.Sequential(
            *list(self.vgg19.classifier.children())[:-1]
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels + 4096, 256),  # GCN + VGG19 特徵
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 7)  # 7 classes
        )

        self.fc_only_vgg = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 7)
        )

    def forward(self, x, edge_index, edge_weight, batch, image_features):

        # 如果有 GCN 的輸入資料，先做三層 GCN + pool
        if x is not None and edge_index is not None and edge_index.numel() > 0:
            # GCN component
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.conv3(x, edge_index, edge_weight=edge_weight)
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

            # 取得 VGG19 的特徵 (4096 維)
            cnn_features = self.vgg19(image_features)  # [batch_size, 4096]

            # 結合 GCN 與 VGG19 的特徵
            combined_features = torch.cat([x, cnn_features], dim=1)  # [batch_size, hidden_channels+4096]
            GNN_output = self.fc_layers(combined_features)

            # 僅使用 VGG19 特徵做分類
            only_vgg_output = self.fc_only_vgg(cnn_features)

            return GNN_output, only_vgg_output

        else:
            # 沒有 GCN 的輸入，僅使用 VGG19
            cnn_features = self.vgg19(image_features)
            return None, self.fc_only_vgg(cnn_features)
        
class Net_ResNet18(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(Net_ResNet18, self).__init__()
        
        # --------------------
        # GCN Layers
        # --------------------
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # --------------------
        # Pretrained ResNet18 (移除最後一層全連接 fc)
        # --------------------
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # 直接用 Identity 取代全連接層

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_channels + 512, 256),  # GCN + ResNet18 特徵
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 7)  # 7 classes
        )

        self.fc_only_resnet = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 7)
        )

    def forward(self, x, edge_index, edge_weight, batch, image_features):

        if x is not None and edge_index is not None and edge_index.numel() > 0:
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.conv3(x, edge_index, edge_weight=edge_weight)
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

            resnet_features = self.resnet(image_features)  # [batch_size, 512]

            combined_features = torch.cat([x, resnet_features], dim=1)  
            GNN_output = self.fc_layers(combined_features)

            only_resnet_output = self.fc_only_resnet(resnet_features)

            return GNN_output, only_resnet_output

        else:
            resnet_features = self.resnet(image_features)
            return None, self.fc_only_resnet(resnet_features)
        
class AlexNet_Only(nn.Module):
    def __init__(self, num_classes=7):
        super(AlexNet_Only, self).__init__()

        # 載入預訓練的 AlexNet
        self.alexnet = models.alexnet(pretrained=True)

        # 移除最後一層，保留 4096 維的全連接輸出
        # (原本 AlexNet 最後的 classifier[-1] 是輸出到 1000 維)
        self.alexnet.classifier = nn.Sequential(
            *list(self.alexnet.classifier.children())[:-1]  # 最後一層去掉
        )
        
        # 自定義全連接層，用於最終的分類 (這裡假設 7 類)
        self.fc = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: (N, 3, 224, 224) - 輸入圖像
        """
        # 利用 AlexNet 抽取 4096 維特徵
        features = self.alexnet(x)  # [N, 4096]
        # 經過自定義的分類層
        out = self.fc(features)     # [N, num_classes]
        return out
    
class VGG19_Only(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG19_Only, self).__init__()

        # 載入預訓練的 VGG19
        self.vgg19 = models.vgg19(pretrained=True)
        
        # 原本 VGG19 的 classifier 結構最後是輸出到 1000 維
        # 移除最後一層，保留 4096 維的輸出
        self.vgg19.classifier = nn.Sequential(
            *list(self.vgg19.classifier.children())[:-1]
        )

        # 自定義全連接層，用於最終的分類 (這裡假設 7 類)
        self.fc = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: (N, 3, 224, 224) - 輸入圖像
        """
        # 利用 VGG19 抽取 4096 維特徵
        features = self.vgg19(x)  # [N, 4096]
        # 經過自定義的分類層
        out = self.fc(features)   # [N, num_classes]
        return out
    
class ResNet18_Only(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet18_Only, self).__init__()

        # 載入預訓練的 ResNet18
        self.resnet = models.resnet18(pretrained=True)

        # 將原本最後的全連接層 (fc) 改成 Identity
        # 讓輸出維度維持在 512，而不是原本的 1000
        self.resnet.fc = nn.Identity()

        # 自定義全連接層 (輸入維度 512, 輸出 7 類)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        x: (N, 3, 224, 224) - 輸入圖像
        """
        # 取出 ResNet18 的 512 維特徵向量
        features = self.resnet(x)  # [N, 512]
        # 經過自定義的線性層做最終分類
        out = self.fc(features)    # [N, num_classes]
        return out

class NetWrapper(nn.Module):
    def __init__(self, model, edge_index, edge_weight, batch):
        super(NetWrapper, self).__init__()
        self.model = model
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.batch = batch

    def forward(self, image_features):
        return self.model(None, self.edge_index, self.edge_weight, self.batch, image_features)