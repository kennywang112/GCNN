import os
import cv2
import csv
import numpy as np
import pandas as pd
import mediapipe as mp

from utils.getfacefeature import paths, verify_file_pairing

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./model/face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

DIRECTORY = r"./Image_data/DATASET/train"
CATEGORIES = []

folders = os.listdir(DIRECTORY)
print(f"Directories in '{DIRECTORY}':")

for folder in folders:
    if os.path.isdir(os.path.join(DIRECTORY, folder)):
        CATEGORIES.append(folder)

print("Categories:", CATEGORIES)

print("Creating connections.csv")
face_mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
folder_path = './output_data'
file_path = os.path.join(folder_path, 'connection.csv')
os.makedirs(folder_path, exist_ok=True)
with open(file_path, 'w', newline='') as connections_csv:
    connections_writer = csv.writer(connections_csv)
    connections_writer.writerow(["index","point1", "point2"])
    index = -1
    for connection in face_mesh_connections:
        index += 1
        point1 = connection[0]
        point2 = connection[1]
        connections_writer.writerow([index, point1, point2])

print("Creating landmarks.csv")

for desired_category in CATEGORIES:
    
    print(f"Processing category: {desired_category}")

    landmarks_file, category_path = paths(desired_category, DIRECTORY=DIRECTORY)

    # create CSV
    with open(landmarks_file, 'w', newline='') as landmarks_csv:
        
        landmarks_writer = csv.writer(landmarks_csv)
        
        landmarks_writer.writerow(["image_name", "category", "landmark_index", "x", "y", "z"])  # Landmarks
        # create FaceLandmarker
        with FaceLandmarker.create_from_options(options) as landmarker:
            print(f"Processing category: {desired_category}")
            
            # Iterate through each image in the specified category
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                frame = cv2.imread(image_path)
                
                if frame is None:
                    print(f"Cannot read image: {image_name}")
                    continue

                h, w = frame.shape[:2]
                # to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # create Mediapipe image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                # FaceLandmarker identify face
                face_landmarker_result = landmarker.detect(mp_image)
                
                # Get face feature
                face_landmarks_list = face_landmarker_result.face_landmarks

                if not face_landmarks_list:
                    continue

                # process face feature
                for face_landmarks in face_landmarks_list:
                    # save landmarks
                    for idx, landmark in enumerate(face_landmarks):
                        x = landmark.x * w
                        y = landmark.y * h
                        z = landmark.z * w
                        landmarks_writer.writerow([image_name, desired_category, idx, x, y, z])

    landmarks_df = pd.read_csv(f'./output_data/landmarks/face_landmarks_{desired_category}.csv')
    connections_df = pd.read_csv('./output_data/connection.csv')
    output_folder = f'./output_data/adjacency/adjacency_{desired_category}'
    os.makedirs(output_folder, exist_ok=True)

    for image_name in landmarks_df['image_name'].unique():
        # Filter landmarks for the current image
        image_landmarks_df = landmarks_df[landmarks_df['image_name'] == image_name].reset_index(drop=True)
        
        # Extract coordinates and the number of landmarks for the image
        points_coordinates = image_landmarks_df[['x', 'y', 'z']].values
        num_points = len(points_coordinates)
        
        # Initialize the adjacency matrix
        adjacency_matrix = np.zeros((num_points, num_points))

        point_indices_1 = connections_df['point1'].astype(int).values
        point_indices_2 = connections_df['point2'].astype(int).values
        valid_indices = (point_indices_1 < num_points) & (point_indices_2 < num_points)
        point_indices_1 = point_indices_1[valid_indices]
        point_indices_2 = point_indices_2[valid_indices]

        coords1 = points_coordinates[point_indices_1]
        coords2 = points_coordinates[point_indices_2]
        
        # Compute distances for all connection pairs
        distances = np.linalg.norm(coords1 - coords2, axis=1)
        
        # Fill the adjacency matrix with distances
        adjacency_matrix[point_indices_1, point_indices_2] = distances
        adjacency_matrix[point_indices_2, point_indices_1] = distances  # Symmetric
        
        adjacency_df = pd.DataFrame(adjacency_matrix)
        output_path = os.path.join(output_folder, f'adjacency_matrix_{image_name}.csv')
        adjacency_df.to_csv(output_path, index=False)
        # print(f"Saved adjacency matrix for {image_name} to {output_path}")
        
    print(f"Verifying for adjacency matrix for category {desired_category}")
    # Define the directories for images and adjacency matrices
    IMAGE_DIRECTORY = r"./Image_data/DATASET/train/" + desired_category
    ADJACENCY_DIRECTORY = r"output_data/adjacency/adjacency_" + desired_category

    # Run the verification
    verify_file_pairing(IMAGE_DIRECTORY, ADJACENCY_DIRECTORY)
    print(f"Verified adjacency matrix for category {desired_category}")