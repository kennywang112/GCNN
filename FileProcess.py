import os
import torch

from models import *
from utils.utils_preprocess import *

device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cuda" 
    if torch.cuda.is_available() 
    else "cpu"
)
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
# print('Processing dir...')

data_dict = {str(i): [] for i in range(1, 8)}
num_node_features = 21

for key, (directory, label) in dir_label_map.items():
    print(key)
    data_dict = process_files(directory, label, image_dir, data_dict, key, num_node_features)

# save
torch.save(data_dict, './output_data/data_dict.pth')

print("Processed data")
print([len(data_dict[f'{i}']) for i in range(1, 8)])