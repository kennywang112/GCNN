import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from models import Net_Alex

device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)
num_node_features = 21
hidden_channels = 64 

model_name = 'model_Net_Alex'
model = Net_Alex(num_node_features, hidden_channels)
model.load_state_dict(torch.load(f'./model/{model_name}.pth', map_location=torch.device(device)))
model.eval()
model.to(device)

test_data_dict_path = './output_data/data_dict.pth'
test_data_dict = torch.load(test_data_dict_path)
test_list = sum([test_data_dict[str(i)] for i in range(1, 8)], [])
test_loader = GeoDataLoader(test_list, batch_size=64, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        edge_weight = batch.edge_weight
        image_features = batch.image_features
        batch_y = batch.y
        
        if model_name in ['model_alex_only', 'model_Restnet18_only', 'model_EfficientNet_Only', 
                          'model_VGG16_only', 'model_GoogleNet_Only']:
            # CNN only
            out = model(image_features)
        elif model_name in ['model_Net_Alex', 'model_Net_VGG', 'model_Net_Restnet', 
                            'model_Net_EfficientNet', 'model_Net_Google']:
            # GNN + CNN
            GNN_output, AlexNet_output = model(x, edge_index, edge_weight, batch.batch, image_features)
            out = GNN_output if GNN_output is not None else AlexNet_output
        
        _, preds = out.max(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(cm.shape[0])])
print("Classification Report:")
print(report)
