import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

def calculate_features_from_point_cloud(file_path):
    # load points
    points = pd.read_csv(file_path, header=None).values

    # Columns are in the order x, y, z
    z_coords = points[:, 2]

    # Filter points above 2m 
    filtered_points = points[z_coords >= 2]

    # Heights to use for calculating features
    heights = filtered_points[:, 2]

    # Height distribution features from Yu et al. 2017
    hmax = heights.max() if len(heights) > 0 else 0
    hmin = heights.min() if len(heights) > 0 else 0
    hrange = hmax - hmin
    hmean = heights.mean() if len(heights) > 0 else 0
    hstd = heights.std() if len(heights) > 0 else 0
    HP10_to_HP90 = np.percentile(heights, np.arange(10, 100, 10)) if len(heights) > 0 else np.zeros(9)


    # CA, CV, and CD implementations were suggested by ChatGPT
    # Crown Area (CA)
    if len(filtered_points) > 2:
        hull2d = ConvexHull(filtered_points[:, :2])
        CA = hull2d.volume
    else:
        CA = 0

    # Crown Volume (CV)
    if len(filtered_points) > 3:
        hull3d = ConvexHull(filtered_points)
        CV = hull3d.volume
    else:
        CV = 0

    # Crown Diameter (CD)
    if len(filtered_points) > 2:
        CD = np.max([np.linalg.norm(filtered_points[i] - filtered_points[j])
                     for i in range(len(hull2d.vertices))
                     for j in range(i + 1, len(hull2d.vertices))])
    else:
        CD = 0

    return hmax,hrange,hmean, hstd, HP10_to_HP90, CA, CV, CD



# This step is to handle the specific folder structure i have the point clouds in. 
def process_species_folder(train_or_val_folder):
    if train_or_val_folder=="train":
        train_or_val_folder = 'Training_Data_Ref_BINEXP/split_data/mrg_train'

    else:
        train_or_val_folder = 'Training_Data_Ref_BINEXP/split_data/mrg_val'

    # Load feature table
    features_df = pd.read_csv(os.path.join(train_or_val_folder, 'features_h_intensity.csv'))

    # Initialize new features
    features_df['hmax'] = 0
    features_df['hrange'] = 0
    features_df['hmean'] = 0
    features_df['hstd'] = 0
    features_df['CA'] = 0
    features_df['CV'] = 0
    features_df['CD'] = 0

    for i in range(1, 10):
        features_df[f'HP{i*10}'] = 0

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
                hmax, hrange, hmean, hstd, HP10_to_HP90, CA, CV, CD = calculate_features_from_point_cloud(file_path)
            
                # Find row index in feature table that matches area code and label
                row_index = features_df[(features_df['area_code'] == area_code) & (features_df['label'].astype(str) == label)].index
                if len(row_index) == 1:
                    row_index = row_index[0]
                    # Update the feature table
                    features_df.at[row_index, 'hmax'] = hmax
                    features_df.at[row_index, 'hrange'] = hrange
                    features_df.at[row_index, 'hmean'] = hmean
                    features_df.at[row_index, 'hstd'] = hstd
                    features_df.at[row_index, 'CA'] = CA
                    features_df.at[row_index, 'CV'] = CV
                    features_df.at[row_index, 'CD'] = CD
                    for i, hp in enumerate(HP10_to_HP90, start=1):
                        features_df.at[row_index, f'HP{i*10}'] = hp

    # Save the updated feature table
    features_df.to_csv(os.path.join(train_or_val_folder, 'features_h_intensity.csv'), index=False)

process_species_folder('train')
process_species_folder('val')
