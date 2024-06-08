import pandas as pd
import numpy as np
import os

def calculate_features_from_point_cloud(file_path):

    # load points
    points = pd.read_csv(file_path, header=None).values

    # Columns are in the order x, y, z
    z_coords = points[:, 2]
    
    # Calculate penetration
    total_points = len(z_coords)
    below_2m = np.sum(z_coords < 2)
    penetration = below_2m / total_points if total_points > 0 else 0

    # Calculate densities (ChatGPT)
    valid_points = z_coords[z_coords >= 2]
    if len(valid_points) > 0:
        # Shifting range by 2 to start at 0
        interval_height = (max(valid_points) - 2) / 10
        densities = [np.sum((valid_points >= (2+i*interval_height)) & (valid_points < (2+(i+1)*interval_height))) for i in range(10)]
    else:
        densities = [0] * 10
    
    densities = [d / total_points if total_points > 0 else 0 for d in densities]

    return penetration, densities

# This step is to handle the specific folder structure i have the point clouds in. 
def process_species_folder(train_or_val_folder):
    if train_or_val_folder=="train":
        train_or_val_folder = 'Training_Data_Ref_BINEXP/split_data/mrg_train'

    else:
        train_or_val_folder = 'Training_Data_Ref_BINEXP/split_data/mrg_val'

    # Load feature table
    features_df = pd.read_csv(os.path.join(train_or_val_folder, 'features_h_intensity.csv'))

    # Initialize new features
    for i in range(1, 11):
        features_df[f'D{i}'] = 0
    features_df['penetration'] = 0

    # Process each species folder
    for i in range(1, 10):
        species_folder = os.path.join(train_or_val_folder, str(i))
        for file_name in os.listdir(species_folder):
            if file_name.endswith(".txt"):
                # Extract area code and label from file name
                parts = file_name.split("_")
                area_code = parts[1]
                label = parts[2].split(".")[0]
                
                file_path = os.path.join(species_folder, file_name)
                penetration, densities = calculate_features_from_point_cloud(file_path)
                
                # Find row index in feature table that matches area code and label
                row_index = features_df[(features_df['area_code'] == area_code) & (features_df['label'].astype(str) == label)].index
                if len(row_index) == 1:
                    # Update the feature table
                    features_df.at[row_index[0], 'penetration'] = penetration
                    for i, density in enumerate(densities, start=1):
                        features_df.at[row_index[0], f'D{i}'] = density

    # Save the updated feature table
    features_df.to_csv(os.path.join(train_or_val_folder, 'features_h_intensity.csv'), index=False)

process_species_folder('train')
process_species_folder('val')

# process_species_folder('path/to/your/mrg_val')