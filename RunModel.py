import os
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeoDataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

image_data_dir = './Image_data/DATASET/train'
image_lengths = {subdir: count_valid_files(os.path.join(image_data_dir, subdir)) for subdir in os.listdir(image_data_dir) if not subdir.startswith('.') and os.path.isdir(os.path.join(image_data_dir, subdir))}

adjacency_dir = './output_data/adjacency'
adjacency_lengths = {subdir: count_valid_files(os.path.join(adjacency_dir, subdir)) for subdir in os.listdir(adjacency_dir) if not subdir.startswith('.') and os.path.isdir(os.path.join(adjacency_dir, subdir))}

print("Image data lengths:")
print(image_lengths)

print("Adjacency data lengths:")
print(adjacency_lengths)

print("Process data...")
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
num_node_features = 21

for key, (directory, label) in dir_label_map.items():
    data_dict = process_files(directory, label, image_dir, data_dict, key)

print("Processed data")
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
model = Net_Alex(num_node_features, hidden_channels).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# Initialize variables for cumulative predictions and labels
cumulative_preds = []
cumulative_labels = []

print('Start training ...')
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

    # Logging results for each epoch
    print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    # Accumulate all predictions and labels across epochs
    cumulative_preds.extend(epoch_preds)
    cumulative_labels.extend(epoch_labels)
    
print("Getting GradCAM...")

test_pth = [
    'Image_data/DATASET/test/1/test_0002_aligned.jpg',
    'Image_data/DATASET/test/2/test_0274_aligned.jpg',
    'Image_data/DATASET/test/3/test_0007_aligned.jpg',
    'Image_data/DATASET/test/4/test_0003_aligned.jpg',
    'Image_data/DATASET/test/5/test_0001_aligned.jpg',
    'Image_data/DATASET/test/6/test_0017_aligned.jpg',
    'Image_data/DATASET/test/7/test_2389_aligned.jpg'
]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
for index, pth in enumerate(test_pth):

    input_image = Image.open(pth)
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
    rgb_img = np.array(input_image.resize((224, 224))) / 255.0

    # model = Net_Alex(num_node_features, hidden_channels).to(device)

    edge_weight = data_dict['1'][0].edge_weight
    edge_index = data_dict['1'][0].edge_index
    batch = val_loader.dataset[0]

    wrapped_model = NetWrapper(model, edge_index, edge_weight, batch)

    target_layers = [wrapped_model.model.alexnet.features[-1]]

    cam = GradCAM(model=wrapped_model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)

    visualization = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)

    try:
        # save the plot
        plt.imsave(f'GradCam/netalex_{index + 1}.png', visualization)
    except FileNotFoundError:
        os.mkdir('GradCam')
        plt.imsave(f'GradCam/netalex_{index + 1}.png', visualization)
