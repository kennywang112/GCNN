import os

def paths(desired_category, DIRECTORY):
    # Check if the category directory exists
    category_path = os.path.join(DIRECTORY, desired_category)

    if not os.path.isdir(category_path):
        print(f"Category '{desired_category}' does not exist in '{DIRECTORY}'.")
        # List available categories if the desired one is missing
        available_categories = [folder for folder in os.listdir(DIRECTORY) if os.path.isdir(os.path.join(DIRECTORY, folder))]
        print("Available categories:", available_categories)
        # Raise an error if the category does not exist
        raise FileNotFoundError(f"Category '{category_path}' not found.")

    # Create the output directory structure if it does not exist
    output_dir = './output_data'
    landmarks_dir = os.path.join(output_dir, 'landmarks')

    # Ensure output directories exist
    for path in [landmarks_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

    # Specify the output file paths for landmarks and connections
    landmarks = os.path.join(landmarks_dir, f'face_landmarks_{desired_category}.csv')
    
    # Return the paths for further use
    return landmarks, category_path

def verify_file_pairing(image_dir, adjacency_dir):
    """
    Verify if each adjacency matrix file has a corresponding image file.
    """
    # List all adjacency matrix files
    adjacency_files = [f for f in os.listdir(adjacency_dir) if f.endswith('.csv')]

    unmatched_files = []

    for adjacency_file in adjacency_files:
        # Extract the base name without extensions for matching
        base_name = os.path.splitext(adjacency_file.replace('adjacency_matrix_', '').replace('.csv', ''))[0]

        # Construct the corresponding image file name
        image_file = f"{base_name}.jpg"
        image_path = os.path.join(image_dir, image_file)

        if not os.path.exists(image_path):
            unmatched_files.append((adjacency_file, image_file))

    if unmatched_files:
        print("The following adjacency matrices do not have matching image files:")
        for adj, img in unmatched_files:
            print(f"Adjacency Matrix: {adj} | Expected Image: {img}")
    else:
        print("All adjacency matrices have corresponding image files.")
