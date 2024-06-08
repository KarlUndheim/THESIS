import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis

def voxel_densities(file_path, voxel_size):

    # load points
    points = pd.read_csv(file_path, header=None).values
    
    # Calculate voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Identify unique voxels and count how many points in each
    _, voxel_counts = np.unique(voxel_indices, return_counts=True, axis=0)

    # Calculate voxel features from the distribution
    V_mean = np.mean(voxel_counts)
    V_max = np.max(voxel_counts)
    V_min = np.min(voxel_counts)
    V_median = np.median(voxel_counts)
    V_std = np.std(voxel_counts)
    V_skewness  = skew(voxel_counts)
    V_kut = kurtosis(voxel_counts)
    V_p90 = np.percentile(voxel_counts, 90)
    total_points = len(points)
    
    return V_mean, V_max, V_min, V_median, V_std, V_skewness, V_kut, V_p90, total_points


# This step is to handle the specific folder structure i have the point clouds in. 
def process_species_folder(train_or_val_folder):
    if train_or_val_folder=="train":
        train_or_val_folder = 'Training_Data_Ref_BINEXP/split_data/mrg_train'

    else:
        train_or_val_folder = 'Training_Data_Ref_BINEXP/split_data/mrg_val'

    # Load feature table
    features_df = pd.read_csv(os.path.join(train_or_val_folder, 'features_h_intensity.csv'))

    voxel_sizes = [0.5]
    for i, voxel_size in enumerate(voxel_sizes):
        
        # Change this when you want multiple V features from different voxel size
        denote = 'V'

        # Initialize new features
        features_df[f'{denote}_min'] = 0
        features_df[f'{denote}_mean'] = 0
        features_df[f'{denote}_max'] = 0
        features_df[f'{denote}_median'] = 0
        features_df[f'{denote}_std'] = 0
        features_df[f'{denote}_sk'] = 0
        features_df[f'{denote}_kut'] = 0
        features_df[f'{denote}_p90'] = 0
        features_df['V_points'] = 0

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
                    mean, max, min, median, std, sk, kut, p90, pts = voxel_densities(file_path, voxel_size=voxel_size)
                    

                    # Find row index in feature table that matches area code and label
                    row_index = features_df[(features_df['area_code'] == area_code) & (features_df['label'].astype(str) == label)].index
                    if len(row_index) == 1:
                        # Update the feature table
                        features_df.at[row_index[0], f'{denote}_min'] = min
                        features_df.at[row_index[0], f'{denote}_mean'] = mean
                        features_df.at[row_index[0], f'{denote}_max'] = max
                        features_df.at[row_index[0], f'{denote}_median'] = median
                        features_df.at[row_index[0], f'{denote}_std'] = std
                        features_df.at[row_index[0], f'{denote}_sk'] = sk
                        features_df.at[row_index[0], f'{denote}_kut'] = kut
                        features_df.at[row_index[0], f'{denote}_p90'] = p90
                        features_df.at[row_index[0], 'V_points'] = pts
   
    # Save the updated feature table
    features_df.to_csv(os.path.join(train_or_val_folder, 'features_h_intensity.csv'), index=False)

     
process_species_folder('train')
print('Features updated')
process_species_folder('val')
print('Features updated')
