import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import os
import numpy as np
import pandas as pd
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.cfg_utils import parse_args_yaml
from utils.file_utils import create_folder

from TLSpecies_SimpleView.utils import get_used_feats_names, get_used_feats_names_2
from TLSpecies_SimpleView.cls_tree.utils_tree import warmup_lr
from torch.utils.data.sampler import WeightedRandomSampler


def load_feats(cfgs):
    """
    load features of data
    :param cfgs:
    :return:
        return pd's columns:
        area_code, label, h:height<_range, _max>, i: intensity<_max, _min, _mean, _sk, _kut, _p90>
    """
    cfgs_data = cfgs.data

    root_dir = cfgs_data.data_path
    # get features' filename
    feat_filename = cfgs_data.feat_filename.replace(".csv", "")

    # get training and validation data's names
    train_filename = os.path.join(root_dir, cfgs_data.datasplit.train, feat_filename+".csv")
    val_filename   = os.path.join(root_dir, cfgs_data.datasplit.val, feat_filename+".csv")

    # get normalization filename
    normvalue_filename = os.path.join(root_dir, cfgs_data.datasplit.train, feat_filename+"_mean_std.csv")

    # Features to exclude
    feats_to_use = cfgs_data.use_feats
    feats_cols = np.array(['hrange', 'hmax', 'imax_0', 'imin_0', 'imean_0', 'isk_0', 'ikut_0', 'ip90_0',
 'imax_1', 'imin_1', 'imean_1', 'isk_1', 'ikut_1', 'ip90_1', 'imax_2', 'imin_2',
 'imean_2', 'isk_2', 'ikut_2', 'ip90_2'])# np.array(["hrange", "hmax", "imax_1", "imin_1", "imax_2", "imin_2",])
    

    if cfgs.data.all_feats:
        feats_to_use = feats_cols
    else:
        feats_to_use = get_used_feats_names(feats_frame_columns=feats_cols, features_list=feats_to_use)

    # load data to pd
    train_pd = pd.read_csv(train_filename)
    val_pd   = pd.read_csv(val_filename)
    norm_pd  = pd.read_csv(normvalue_filename, index_col=0)

    train_pd_temp = train_pd[feats_to_use]

    # norm
    feats_name = train_pd_temp.columns.values # np.array ["area_code", "label", "hrange", ...]
    """ feats_name = feats_name[2:] """

    print("")
    print('FEATURES USED: ', feats_name)
    print("")

    if cfgs_data.scale_feats:
        train_pd[feats_name] = (train_pd[feats_name] - norm_pd.loc["mean", feats_name]) / norm_pd.loc["std", feats_name]
        val_pd[feats_name]   = (val_pd[feats_name] - norm_pd.loc["mean", feats_name]) / norm_pd.loc["std", feats_name]

    return train_pd, val_pd, feats_name


def load_data(cfgs):
    #################
    # load training and validation data with gt label
    # area_code, label, species
    #################
    train_gt = pd.read_csv(os.path.join(cfgs.data.data_path, "train_list.csv"))
    val_gt   = pd.read_csv(os.path.join(cfgs.data.data_path, "val_list.csv"))

    # get the data of considered species
    train_gt = train_gt[train_gt["species"].isin(cfgs.data.consider_species)]
    val_gt   = val_gt[val_gt["species"].isin(cfgs.data.consider_species)]
    print("train_gt: ", train_gt.shape, " val_gt: ", val_gt.shape)

    if cfgs.data.mapping:
        # Transform the 'species' column
        # Reclassify species 1 and 2 as 0 ('conifers'), and species 3 through 9 as 1 ('deciduous')
        species_mapping = cfgs.data.mapping
        train_gt['species'] = train_gt['species'].map(species_mapping)
        val_gt['species'] = val_gt['species'].map(species_mapping)


    #################
    # load features
    # area_code, label, h:height<range, max>, i: intensity<_max, _min, _mean, _sk, _kut, _p90>
    #################
    train_data, val_data, feats_name = load_feats(cfgs)
    print("train_data: ", train_data.shape)

    #################
    # add species (class label) to train and validation data
    #################
    train_data = train_data.merge(train_gt, how="inner", on=["area_code", "label"])
    val_data   = val_data.merge(val_gt, how="inner", on=["area_code", "label"])
    print("after merge: train: ", train_data.shape, "val: ", val_data.shape)
    print("after merge columns: ", train_data.columns)

    return train_data, val_data, feats_name

## MLP architecture
class FeatureModel(nn.Module):
    def __init__(self, input_features, hidden_dim, output_classes, dropout_rate = 0.25, all_species=None):
        super(FeatureModel, self).__init__()
        self.num_classes = output_classes
        self.species = all_species
        self.network = nn.Sequential(

            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, output_classes)

            
        )
        """ nn.Linear(input_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_classes) """



        """ nn.Linear(input_features, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.BatchNorm1d(hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        nn.Linear(hidden_dim // 2, output_classes) """
        

    def forward(self, x):
        return self.network(x)
    
def calc_accuracy(true_labels, pred_labels):
    correct = (true_labels == pred_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total
    return accuracy

def validate_model(model, val_loader):
    model.eval() 
    correct = 0
    total = 0

    num_classes = model.num_classes
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():  
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    accuracy = correct / total
    """ print(f'Validation Accuracy: {accuracy * 100:.2f}%') """
    return accuracy, confusion_matrix


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, scheduler=None, warmup=None):
    best = 0
    best_epoch = 0
    best_model = None

    if warmup:
        warmup_scheduler = warmup_lr(optimizer, 'linear', warm_period=int(epochs * len(train_loader) / 20))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        """ print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}') """
        
        if scheduler:
            scheduler.step()

        if warmup:
            if epoch < len(train_loader) - 1:
                with warmup_scheduler.dampening():
                    pass

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        """ print(f'Validation Loss: {val_loss/len(val_loader)}') """
        
        
        accuracy, cm = validate_model(model, val_loader)
        if accuracy>best:
            best = accuracy
            best_epoch = epoch
            best_model = cm
        print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy * 100:.2f}%')

        if warmup:
            with warmup_scheduler.dampening():
                scheduler.step()
    
    print("")
    print(f'Best accuracy: {best:.3f}\t at epoch {best_epoch}')
    print("")

    totals = best_model.sum(axis=1)

    accs = np.zeros(len(totals))
    classes = model.species
    for i in range(len(totals)):
        accs[i] = best_model[i, i] / totals[i]
        print(
            f"{classes[i]}: "
            f"Got {best_model[i, i]}/{totals[i]} with accuracy {accs[i] * 100:.2f}")
    
    print('Confusion matrix: \n', best_model)
    print('')

def main(cfgs):

    work_dir = os.path.join(cfgs.data.save_root,
                                f"{cfgs.exp_name}_titan_")


    train_data, val_data, feats_name = load_data(cfgs)

    from sklearn.preprocessing import LabelEncoder

    # Initialize the encoder
    encoder = LabelEncoder()

    all_species = train_data['species'].unique()

    # Fit and transform the 'species' column to get encoded labels
    train_data['species_encoded'] = encoder.fit_transform(train_data['species'])
    val_data['species_encoded'] = encoder.transform(val_data['species'])

    # Checking unique values 
    print(f"Unique species in training set: {train_data['species_encoded'].unique()}")



   
    # Convert features to PyTorch tensors
    train_feats_tensor = torch.FloatTensor(train_data[feats_name].values)
    val_feats_tensor = torch.FloatTensor(val_data[feats_name].values)

    train_labels_tensor = torch.LongTensor(train_data['species_encoded'].values)
    val_labels_tensor = torch.LongTensor(val_data['species_encoded'].values)

    train_dataset = torch.utils.data.TensorDataset(train_feats_tensor, train_labels_tensor)
    val_dataset = torch.utils.data.TensorDataset(val_feats_tensor, val_labels_tensor)

    num_features = train_dataset.tensors[0].shape[1]

    print(f"Number of different features in train_dataset: {num_features}")

    sampler = 'alanced'

    # Trying balanced sampling
    if sampler[0] == 'b':
        print("Using balanced sampling...")

        labels = torch.tensor(train_data['species_encoded']) #Counts over
        counts = torch.bincount(labels) #Training set only
        label_weights = 1 / counts
        
        sample_weights = torch.stack([label_weights[label] for label in labels])

        # Replacement is true by default for the weighted sampler
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # Dataloader using the weighted sampler
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                                    sampler=train_sampler)
    else:
        print("Using random sampling...")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64)

    # Model instantiation
    input_features = len(feats_name)
    hidden_dim = 512  # 128x128 best for hierarchical. 64x64 best for multi-class
    output_classes = len(train_data['species_encoded'].unique())
    model = FeatureModel(input_features, hidden_dim, output_classes, dropout_rate=0.0, all_species=all_species)


    species = cfgs.data.consider_species

    class_weight_dict = {
        1:1.0, 
        2:2.0, 
        3:1.0, 
        4:1.0, 
        5:1.0, 
        6:1.5, 
        7:1.5, 
        8:1.5, 
        9:1.5
        }
    
    weights = [class_weight_dict[i] for i in species]

    
    lr = 0.0005

    optimizer = 'adam'
    use_weights = False

    # Loss and optimizer
    if use_weights:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2, weight=torch.tensor(weights))
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    if optimizer =="sgd":
        print("Optimizing with SGD...")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer=="adam":
        print("Optimizing with AdaM...")
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler setup
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, scheduler=scheduler, warmup=False)

if __name__=="__main__":
    #
    # cfg_path = "cfg/cfg_ml_rf.yaml" # -> ~65%
    cfg_path = "cfg/cfg_ml_rf.yaml"
    cfgs = parse_args_yaml(cfg_path, usedot=True)

    main(cfgs)