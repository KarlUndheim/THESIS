"""
@File           : main_titan_f.py
@Author         : Gefei Kong
@Time:          : 22.11.2023 16:45
------------------------------------------------------------------------------------------------------------------------
@Description    : as below
cls tree with more features
refer: https://github.com/JulFrey/DetailView
"""

import datetime
import os
import logging
import torch

from pprint import pformat

import sys
import pprint

# KARL: Had to add the root directory to sys.path for the imports to work
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from TLSpecies_SimpleView.cls_tree_f import train_titan_f
from TLSpecies_SimpleView.utils.dataset_titan_dv import TitanTreePointDataset_dv
from utils.cfg_utils import parse_args_yaml
from utils.file_utils import create_folder


def crt_save_viewimgs(data_dir:str, save_dir:str,
                      split:str="mrg_train",
                      num_views:int=7, features:str or list="h", scale_feats:bool=True,
                      iimg_dim:int=128, oimg_dim:int=256, transform_prop:float=-1,
                      is_debug:bool=False,
                      is_override:bool=True):
    # save info
    save_dir_data = os.path.join(save_dir, "depth_img_data", f"isize{iimg_dim}_osize{oimg_dim}")
    create_folder(save_dir_data)
    save_path = os.path.join(save_dir_data, f"trees_v{num_views}_f{'-'.join(features)}_{split}.pt")


    # if is_override (=True), the existed created dataset will be created again.
    # else, directly return the path of existed dataset
    if not is_override:
        if os.path.exists(save_path):
            print("the dataset has been created before, directly return path.")
            return save_path

    print("creating dataset....")
    crted_dataset = TitanTreePointDataset_dv(data_dir=data_dir, split=split,
                                             num_views=num_views, features=features, scale_feats=scale_feats,
                                             iimg_dim=iimg_dim, oimg_dim=oimg_dim,
                                             f=1, cam_dist=1.4)
    if "train" in split:
        crted_dataset.set_params(transforms=['rotation', 'translation',
                                             'scaling'],
                                 transform_prop=transform_prop)  # Other parameters can be changes - for example ...set_params(image_dim=128) .set_params(max_rotation=0.5) etc.

    # save
    torch.save(crted_dataset, save_path)

    # debug output: an example image, the head of meta_frame and the head labels of dataset
    if is_debug:
        from TLSpecies_SimpleView.utils import plot_depth_images
        import matplotlib.pyplot as plt
        plot_depth_images(crted_dataset.__getitem__(3)['depth_images'])
        plt.show()
        plt.close()
        print(f"{split} dataset.head():\n")
        print(crted_dataset.meta_frame.head())
        print(f"{split} dataset labels.head(): ", crted_dataset.labels[:5])

    return save_path


def crt_img_dataset(cfgs, is_debug:bool=False, is_override:bool=True):
    cfgs_data = cfgs.data
    iimage_size, oimage_size = cfgs_data.depth_image.iimg_size, cfgs_data.depth_image.oimg_size

    # train
    train_data_path = crt_save_viewimgs(cfgs_data.data_path,
                                        cfgs_data.save_root,
                                        cfgs_data.datasplit["train"],
                                        cfgs_data.num_views, cfgs_data.features, cfgs_data.scale_feats,
                                        iimage_size, oimage_size, cfgs_data.transform_prop,
                                        is_debug, is_override)
    # val
    val_data_path = crt_save_viewimgs(cfgs_data.data_path,
                                      cfgs_data.save_root,
                                      cfgs_data.datasplit["val"],
                                      cfgs_data.num_views, cfgs_data.features, cfgs_data.scale_feats,
                                      iimage_size, oimage_size, cfgs_data.transform_prop,
                                      is_debug, is_override)
    return train_data_path, val_data_path


def set_logger(log_dir:str):
    create_folder(log_dir)
    ######################
    # log info.
    ######################
    logger = logging.getLogger("Treespecies_simpleview_pytorch")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train_cls.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info(f"PARAMETER ...\n{pformat(cfgs)}")

    return logger

def set_logger_best(log_dir:str):
    create_folder(log_dir)
    ######################
    # log info.
    ######################
    logger = logging.getLogger("Treespecies_simpleview_pytorch_best")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'best_models.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def main(cfgs):
    work_dir = os.path.join(cfgs.data.save_root,
                            f"{cfgs.exp_name}_titan_" +
                            str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) +
                            f"_{cfgs.data.num_views}_dp{cfgs.model.dfmodel.dropout_ratio}_"
                            f"{cfgs.data.depth_image.iimg_size}_{cfgs.data.depth_image.oimg_size}_"
                            f"{cfgs.model.learning_rate[0] if isinstance(cfgs.model.learning_rate, list) else cfgs.model.learning_rate}_"
                            f"b{cfgs.model.batch_size}_e{cfgs.model.epochs}_o{cfgs.model.optimizer}_"
                            f"{cfgs.model.loss_fn}_{cfgs.model.train_sampler}_cls{len(cfgs.model.species_considered)}")
    work_dir = work_dir.replace("_smooth-loss_", "_sml_")

    logger = set_logger(os.path.join(work_dir, "log"))
    best_logger = set_logger_best(os.path.join(work_dir, "log"))
    ######################
    # 1. points -> images
    ######################
    train_data_path, val_data_path = crt_img_dataset(cfgs, is_debug=False, is_override=True)
    print(train_data_path)

    ######################
    # 2. train
    ######################
    params = cfgs.model
    # num_views = cfgs.data["num_views"]
    # num_features = len(cfgs.data["features"]) if isinstance(cfgs.data["features"], list) else 1
    model_dir = os.path.join(work_dir, "checkpoints")
    train_titan_f.train(train_data_path, val_data_path, model_dir, params, logger, best_logger)



if __name__=="__main__":
    # load configs
    #########################
    # convnext_tiny + detailview + h+i features.
    # detailview: front-back, features: hrange, hmax, imax, imin, imean, isk, ikut, ip90
    # cfg_path = "configs_f/cfgf_dfconvnextt_pretrain_v7f20_b8_adam_stlr_sml_balanced_warm_5e-4.yaml" -> best_val_acc: 68.54% --> best, but c3 reduce.
    #
    # detailview: front-back, features: hmax ==> totally the same as detailview
    # -> stop at 47epoch, because it's similar to the original simpleview
    # cfg_path = "configs_f/cfgf_dfconvnextt_pretrain_v7f1_b8_adam_stlr_sml_balanced_warm_5e-4.yaml" -> c3 better, imply intensity might affect c3 acc.
    #
    # detailview: all, features: hrange, hmax, imax, imin, imean, isk, ikut, ip90
    # cfg_path = "configs_f/cfgf_dfconvnextt_pretrain_v12f20_b4_adam_stlr_sml_balanced_warm_5e-4.yaml" -> best_val_acc: 67.61%
    #
    # detailview: all features: all, use detailview_dfmodel_sep (separatedly consider top-down and side views.)
    # cfg_path = "configs_f/cfgf_dfconvnextt_dvsep_pretrain_v7f20_b8_adam_stlr_sml_balanced_warm_5e-4.yaml" -> best_val_acc: 68.54%
    cfg_path = "configs_f/cfgf_dfconvnextt_pretrain_d05_v7f20_b8_adam_stlr_sml_balanced_warm_5e-4.yaml"
    cfgs = parse_args_yaml(cfg_path, usedot=True)

    # main
    main(cfgs)
