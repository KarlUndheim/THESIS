import pandas as pd
import os


def std_mean_generation(train_or_val_folder):

    if train_or_val_folder=="train":
        train_or_val_folder = 'Training_Data_Ref_BINEXP/split_data/mrg_train'

    else:
        train_or_val_folder = 'Training_Data_Ref_BINEXP/split_data/mrg_val'

    # Load the feature table
        
    data = pd.read_csv(os.path.join(train_or_val_folder, 'features_h_intensity.csv'))
    original_headers = data.columns.tolist()
    if 'area_code' in data:
        data.drop('area_code', axis=1, inplace=True)
        original_headers.remove('area_code')

    mean_values = data.mean().tolist()
    std_values = data.std().tolist()

    print(mean_values)

    output_df = pd.DataFrame([mean_values, std_values], columns=original_headers)
    output_df.insert(0, '', ['mean', 'std'])

    # Save the new dataframe to file
    output_df.to_csv(os.path.join(train_or_val_folder, 'features_h_intensity_mean_std.csv'), index=False)

    print("File has been updated with mean and std values for all features.")

std_mean_generation('train')
std_mean_generation('val')




