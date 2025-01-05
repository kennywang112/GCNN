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

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

num_class = 7

# Model setup
num_node_features = next(data.x.shape[1] for data in data_list if data.x is not None)
hidden_channels = 64

print("Load model...")
model_alexnet_gnn = Net_Alex(num_node_features, hidden_channels)
model_alexnet_gnn.load_state_dict(torch.load('./model/model_Net_Alex.pth', map_location=torch.device(device)))
model_alexnet_gnn.eval()
model_alexnet_gnn.to(device)
print("Load model...")
model_resnet_gnn = Net_ResNet18(num_node_features, hidden_channels)
model_resnet_gnn.load_state_dict(torch.load('./model/model_Net_Resnet.pth', map_location=torch.device(device)))
model_resnet_gnn.eval()
model_resnet_gnn.to(device)
print("Load model...")
model_alexnet = AlexNet_Only(num_class)
model_alexnet.load_state_dict(torch.load('./model/model_alex_only.pth', map_location=torch.device(device)))
model_alexnet.eval()
model_alexnet.to(device)
print("Load model...")
model_resnet = ResNet18_Only(num_class)
model_resnet.load_state_dict(torch.load('./model/model_Resnet18_only.pth', map_location=torch.device(device)))
model_resnet.eval()
model_resnet.to(device)
print("Load model...")
model_vgg = VGG16_Only(num_class)
model_vgg.load_state_dict(torch.load('./model/model_VGG16_only.pth', map_location=torch.device(device)))
model_vgg.eval()
model_vgg.to(device)
print("Load model...")
model_vgg = Net_VGG(num_node_features, hidden_channels)
model_vgg.load_state_dict(torch.load('./model/model_Net_VGG.pth', map_location=torch.device(device)))
model_vgg.eval()
model_vgg.to(device)

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

for model in [model_alexnet_gnn, model_resnet_gnn, model_alexnet, model_resnet, model_vgg]:

    print(model.__class__.__name__)

    for index, pth in enumerate(test_pth):

        input_image = Image.open(pth)
        input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
        rgb_img = np.array(input_image.resize((224, 224))) / 255.0

        edge_weight = data_dict['1'][0].edge_weight
        edge_index = data_dict['1'][0].edge_index
        batch = val_loader.dataset[0]

        if model.__class__.__name__ == 'Net_Alex':
            wrapped_model = NetWrapper(model, edge_index, edge_weight, batch)
            target_layers = [wrapped_model.model.alexnet.features[-1]]
        elif model.__class__.__name__ == 'Net_ResNet18':
            wrapped_model = NetWrapper(model, edge_index, edge_weight, batch)
            target_layers = [wrapped_model.model.resnet.layer4[-1]]
        elif model.__class__.__name__ == 'AlexNet_Only':
           target_layers = [model.alexnet.features[-1]]
           wrapped_model = model
        elif model.__class__.__name__ == 'ResNet18_Only':
            target_layers = [model.resnet.layer4[-1]]
            wrapped_model = model
        elif model.__class__.__name__ == 'VGG16_Only':
            target_layers = [model.vgg16.features[-1]]
            wrapped_model = model
        elif model.__class__.__name__ == 'Net_VGG':
            wrapped_model = NetWrapper(model, edge_index, edge_weight, batch)
            target_layers = [wrapped_model.model.vgg16.features[-1]]


        cam = GradCAM(model=wrapped_model, target_layers=target_layers)

        grayscale_cam = cam(input_tensor=input_tensor)

        visualization = show_cam_on_image(rgb_img, grayscale_cam[0, :], use_rgb=True)

        try:
            # save the plot
            plt.imsave(f'static/GradCam/{model.__class__.__name__}_{index + 1}.png', visualization)
        except FileNotFoundError:
            os.mkdir('static/GradCam')
            plt.imsave(f'static/GradCam/{model.__class__.__name__}_{index + 1}.png', visualization)
