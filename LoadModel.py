from flask import Flask, Response, render_template, jsonify, send_file, make_response, request

import time
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from models import Net_Alex, Net_ResNet18, Net_VGG, NetWrapper
from utils.utils_app import get_adjacency_matrix, process_for_model, visualize_adjacency_matrix

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

device = "cpu"

label_map = {
    0: "Surprised",
    1: "Fearful",
    2: "Disgusted",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Neutral"
}

hidden_channels=64
num_node_features=21

# model_alexnet_gnn = Net_Alex(num_node_features, hidden_channels)
# model_alexnet_gnn.load_state_dict(torch.load('model/model_Net_Alex.pth', map_location=torch.device(device)))
# model_alexnet_gnn.eval()
# model_alexnet_gnn.to(device)

# model_resnet_gnn = Net_ResNet18(num_node_features, hidden_channels)
# model_resnet_gnn.load_state_dict(torch.load('model/model_Net_Resnet.pth', map_location=torch.device(device)))
# model_resnet_gnn.eval()
# model_resnet_gnn.to(device)

model_vgg_gnn = Net_VGG(num_node_features, hidden_channels)
model_vgg_gnn.load_state_dict(torch.load('model/model_Net_VGG.pth', map_location=torch.device(device)))
model_vgg_gnn.eval()
model_vgg_gnn.to(device)