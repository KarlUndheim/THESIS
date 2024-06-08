"""
@File           : dataset_titan_dv.py
@Author         : Gefei Kong
@Time:          : 22.11.2023 17:12
------------------------------------------------------------------------------------------------------------------------
@Description    : as below

"""

from re import X
import os
import torch
import sys

# KARL: Had to add the root directory to sys.path for the imports to work
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from TLSpecies_SimpleView.utils import get_used_feats_names

torch.pi = torch.acos(torch.zeros(1)).item() * 2

from tqdm import tqdm
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import TLSpecies_SimpleView.utils as utils


class TitanTreePointDataset_dv(Dataset):
    """Dataset for tree species classification from
    point cloud depth projection images"""

    def __init__(self, data_dir, split="train",
                 num_views:int=7, features: str or list="h", scale_feats: bool=True,
                 iimg_dim:int=256, oimg_dim:int=256, f:float=1, cam_dist:float=1.4):
        """
        Args:
            metadata_file (string): Path to the metadata file.
            root_dir (string): Directory with all the images.
        """
        self.data_dir = data_dir
        self.num_views = num_views if num_views>=6 else 6

        # get all features e.g., h, intensity
        self.features_list = features if isinstance(features, list) else [features]
        self.feats_frame = pd.read_csv(os.path.join(self.data_dir, split, "features_h_intensity.csv"))
        # get feats mean, std, etc. info for standarlization. as detailview
        if scale_feats:
            # only import the training data's mean and std
            self.feats_ms = pd.read_csv(os.path.join(self.data_dir, "mrg_train", "features_h_intensity_mean_std.csv"),
                                        index_col=0) # index: mean, std; col: label, feat0, feat1, ......


        # get species list
        self.catfile = os.path.join(self.data_dir, 'titan_tree_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)] # str
        self.classes = dict(zip(self.cat, range(len(self.cat))))  # a dict: {"shape_name_0": 0, ...}

        # get train, validation and test list info.
        # for each row in .csv file: area_code, label, species (label means treeid)
        # meta_frame = pd.read_csv(metadata_file, keep_default_na=False)
        assert ('train' in split or 'val' in split or 'test' in split)
        if "train" in split:
            meta_frame = pd.read_csv(os.path.join(self.data_dir, "train_list.csv"))  # each row: area_code, label, species
            split_type = "train"
        elif "val" in split:
            meta_frame = pd.read_csv(os.path.join(self.data_dir, "val_list.csv"))  # each row: area_code, label, species
            split_type = "val"
        else:
            meta_frame = pd.read_csv(os.path.join(self.data_dir, "test_list.csv"))  # each row: area_code, label, species
            split_type = "test"

        self.meta_frame = meta_frame

        self.point_clouds = []
        # self.labels = None
        self.file_names = []
        self.ids = []

        self.image_dim = iimg_dim # 256
        self.image_dim_out = oimg_dim
        self.camera_fov_deg = 90
        self.f = f # 1
        self.camera_dist = cam_dist # 1.4
        self.transforms = ['none']

        self.min_rotation = 0
        self.max_rotation = 2 * np.pi

        self.min_translation = 0
        self.max_translation = 0.5 # 0.93

        self.min_scale = 0.71
        self.max_scale = 1.71 # 1.51

        self.transform_prop = -1. # -1: don't consider the transform probability

        # filenames = list(filter(lambda t: t.endswith('.txt'), os.listdir(self.data_dir)))
        # no_files = len(filenames)

        no_files = len(meta_frame)
        self.labels = torch.zeros(no_files)

        # added, for additional features used in detailview ###########################
        # get_used_feats_names
        self.feats = None
        self.num_feats = 0
        if len(self.features_list)>0: # there are some additional features will be considered.
            used_feats_names = get_used_feats_names(self.feats_frame.columns.values, self.features_list)
            print("Used features:", used_feats_names)
            self.feats = torch.zeros(size=(no_files, len(used_feats_names)))
            self.num_feats = len(used_feats_names)

        for i, row in tqdm(meta_frame.iterrows(), total=no_files):  # For each file in the directory
            file_name = os.path.join(self.data_dir, split, str(row["species"]),
                                       f"{split.replace(f'{split_type}', '')}{row['area_code']}_{row['label']}.txt")
            self.file_names.append(file_name)  # Save the file name
            self.ids.append(row['label'])
            cloud = utils.pc_from_txt_comma(file_name)  # Load the point cloud
            # cloud = utils.center_and_scale(cloud)  # Center and scale it ---> move this step to create the depth image. for creating detail view

            # Add the point cloud to the dataset #################################
            self.point_clouds.append(torch.from_numpy(cloud))
            # Add the species label (int, index from self.species) to the list of labels ########################
            self.labels[i] = self.classes[str(row["species"])]
            # Add the features of this tree ######################################
            if self.feats is not None:
                # 0,1 is area_code and label
                feats_i  = self.feats_frame[(self.feats_frame["area_code"]==row['area_code']) &
                                            (self.feats_frame["label"]==row['label'])]
                self.feats[i] = torch.from_numpy(feats_i[used_feats_names].values[0]) # numpy -> tensor

        # scale feats
        if scale_feats and self.feats is not None:
            feats_mean = self.feats_ms.loc["mean",used_feats_names].values
            feats_std  = self.feats_ms.loc["std", used_feats_names].values
            self.feats = (self.feats - feats_mean) / feats_std

        self.feats = self.feats.float()
        self.labels = self.labels.long()
        self.counts = self.meta_frame['species'].value_counts()

        return


    def get_depth_image_dv(self, i, transforms=None, transform_prop=-1.):
        if transforms is None:
            transforms = self.transforms
        if transform_prop == -1:
            transform_prop = self.transform_prop

        points = self.point_clouds[i]

        if 'rotation' in transforms:
            points = self.random_rotation(points,
                                          min_rotation=self.min_rotation,
                                          max_rotation=self.max_rotation,
                                          transform_prop=transform_prop)

        if 'translation' in transforms:
            points = self.random_translation(points,
                                             min_translation=self.min_translation,
                                             max_translation=self.max_translation,
                                             transform_prop=transform_prop)

        if 'scaling' in transforms:
            points = self.random_scaling(points,
                                         min_scale=self.min_scale,
                                         max_scale=self.max_scale,
                                         transform_prop=transform_prop)

        return torch.unsqueeze(
            utils.get_depth_images_from_cloud_dv(points=points,
                                                 num_views=self.num_views,
                                                 image_dim=self.image_dim,
                                                 camera_fov_deg=self.camera_fov_deg,
                                                 f=self.f,
                                                 camera_dist=self.camera_dist
                                                )
            , 1)

    def remove_species(self, specie):

        idx = []  # Indices to keep

        classes_inv = {v:k for k,v in self.classes.items()}
        for i in range(len(self.labels)):  # Remove entries in images and labels for that species
            if not (classes_inv[int(self.labels[i])] == specie):
                idx.append(i)

        self.point_clouds = [self.point_clouds[i] for i in idx]  # Crop point clouds
        self.labels = self.labels[idx]  # Crop labels
        self.meta_frame = self.meta_frame.iloc[idx]  # Crop meta frame

        old_species = self.classes.copy()
        self.classes.pop(specie)  # Pop from species list
        self.classes = {k:i for i, (k,v) in enumerate(self.classes.items())}

        species_map = [self.classes[sp_k] if sp_k in self.classes.keys() else None for sp_k, sp_v in old_species.items()]

        for k in range(len(self.labels)):  # Apply species map to relabel
            self.labels[k] = torch.tensor(species_map[int(self.labels[k])])

        self.counts = self.counts.drop(int(specie),
                                       errors='ignore')  # remove from the counts series, ignore if it's not in there.

        return
    
    def group_species(self, mapping):
        """
        Re-label species into groups based on the provided mapping.

        Parameters:
        - self: the dataset instance
        - mapping: a dictionary where keys are original species IDs and
        values are the group IDs to which they should be mapped.
        """

        # Update labels according to the mapping
        for k in range(len(self.labels)):
            original_label = int(self.labels[k])
            if original_label in mapping:
                self.labels[k] = torch.tensor(mapping[original_label])
            else:
                # Handle case where original_label is not in mapping,
                # e.g., leave as is or raise an error
                pass

        # Update the classes attribute to reflect the new groups
        # Assuming the groups in 'mapping' are all the desired final groups
        unique_groups = set(mapping.values())
        self.classes = {str(group_id): group_id for group_id in unique_groups}

        # Update counts for each group
        # This requires recalculating counts based on the new group labels
        self.counts = {group_id: sum(1 for label in self.labels if label == group_id) for group_id in unique_groups}

        return


    def set_params(self,
                   image_dim=None,
                   camera_fov_deg=None,
                   f=None,
                   camera_dist=None,
                   transforms=None,
                   min_rotation=None,
                   max_rotation=None,
                   min_translation=None,
                   max_translation=None,
                   min_scale=None,
                   max_scale=None,
                   transform_prop=-1):

        if image_dim:
            self.image_dim = image_dim
        if camera_fov_deg:
            self.camera_fov_deg = camera_fov_deg
        if f:
            self.f = f
        if camera_dist:
            self.camera_dist = camera_dist
        if transforms:
            self.transforms = transforms
        if min_rotation:
            self.min_rotation = min_rotation
        if max_rotation:
            self.max_rotation = max_rotation
        if min_translation:
            self.min_translation = min_translation
        if max_translation:
            self.max_translation = max_translation
        if min_scale:
            self.min_scale = min_scale
        if max_scale:
            self.max_scale = max_scale
        if transform_prop:
            self.transform_prop = transform_prop

        return

    def random_rotation(self,
                        point_cloud,
                        min_rotation=0,
                        max_rotation=2 * torch.pi,
                        transform_prop=-1):

        if transform_prop!=-1 and torch.rand(1).item() < transform_prop:
            return point_cloud

        theta = torch.rand(1) * (max_rotation - min_rotation) + min_rotation

        Rz = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1],
        ]).double()

        return torch.matmul(point_cloud, Rz.t())

    def random_translation(self,
                           point_cloud,
                           min_translation=0,
                           max_translation=0.1,
                           transform_prop=-1):

        if transform_prop!=-1 and torch.rand(1).item() < transform_prop:
            return point_cloud

        sign = torch.sign(torch.rand(1) - 0.5)
        tran = torch.rand(3) * (max_translation - min_translation) + min_translation

        return point_cloud + sign * tran

    def random_scaling(self,
                       point_cloud,
                       min_scale=0.5,
                       max_scale=1.5,
                       transform_prop=-1):

        if transform_prop!=-1 and torch.rand(1).item() < transform_prop:
            return point_cloud

        scale = torch.rand(1) * (max_scale - min_scale) + min_scale

        return scale * point_cloud

    def __len__(self):
        assert len(self.labels) == len(self.point_clouds)
        assert len(self.meta_frame) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            num_trees = len(idx)
        elif type(idx) == int:
            num_trees = 1
        print("id: ", self.ids[idx], self.num_views)
        depth_images = torch.zeros(size=(num_trees, self.num_views, 1, self.image_dim, self.image_dim))

        if type(idx) == list:
            for i in range(len(idx)):
                if self.image_dim == self.image_dim_out:
                    depth_images[i] = self.get_depth_image_dv(int(idx[i]))
                else:
                    from torchvision import transforms
                    tresize = transforms.Resize((self.image_dim_out, self.image_dim_out))
                    depth_images[i] = tresize(self.get_depth_image_dv(int(idx[i])))
        elif type(idx) == int:
            if self.image_dim == self.image_dim_out:
                depth_images = self.get_depth_image_dv(idx)
            else:
                from torchvision import transforms
                tresize = transforms.Resize((self.image_dim_out, self.image_dim_out))
                depth_images = tresize(self.get_depth_image_dv(idx))



        labels = self.labels[idx]
        # added, get feats
        feats = self.feats[idx] if self.feats is not None else None

        sample = {'depth_images': depth_images, 'labels': labels, "feats": feats}

        return sample

if __name__ == "__main__":
    data_dir = '../../Training_Data/split_data/'
    split="mrg_val"
    num_views=6 # 7
    features =["h", "i"]
    train_dataset = TitanTreePointDataset_dv(data_dir=data_dir, split=split,
                                             num_views=num_views, features=features, scale_feats=True,
                                             iimg_dim=128, oimg_dim=128, f=1, cam_dist=1.4)

    if "train" in split:
        train_dataset.set_params(transforms=['rotation', 'translation',
                                             'scaling'],
                                 transform_prop=0.5)
        
    """ for specie in ['1', '2', '3', '4', '5', '6', '7', '8']:
        train_dataset.remove_species(specie) """



    # 'rotation', 'translation', 'scaling'
    # train_dataset = torch.load(f"/home/gefeik/PhD-work/z_temp_work/20231106-treeSpecies/codes_tree/"
    #                            f"TLSpecies_SimpleView/cls_tree/exp/depth_img_data/isize128_osize256/trees_mrg_train.pt")

    print(f"dataset_num_views:{train_dataset.num_views}, num_feats: {train_dataset.num_feats}")
    for epoch in range(1):
        for i, sample in enumerate(train_dataset):
            specie = sample['labels'].item()
            if specie !=4: continue
            print(epoch, i, sample["labels"], sample["depth_images"].size(), sample["feats"].size())
            print(sample["depth_images"].dtype, sample["feats"].dtype)
            print(sample["feats"])
            fig, ax = utils.plot_depth_images(sample["depth_images"], nrows=2)
            import matplotlib.pyplot as plt
            plt.show()
            plt.close()

    # utils.plot_depth_images(train_dataset.__getitem__(3)['depth_images'])
    # import matplotlib.pyplot as plt
    # plt.show()
    # plt.close()
