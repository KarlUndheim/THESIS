"""
@File           : train_titan_f.py
@Author         : Gefei Kong
@Time:          : 22.11.2023 16:49
------------------------------------------------------------------------------------------------------------------------
@Description    : as below

"""

import os
import sys
import copy
# from tkinter import wantobjects
from pprint import pformat

pdir = os.path.dirname(os.getcwd())
sys.path.append(pdir)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR

from timm.loss import LabelSmoothingCrossEntropy

from sklearn.metrics import confusion_matrix
from datetime import datetime

# import TLSpecies_SimpleView.utils as utils
from TLSpecies_SimpleView.cls_tree.utils_tree import load_checkpoint, save_checkpoints, warmup_lr
from TLSpecies_SimpleView.simpleview_pytorch import DetailView_DfModel, DetailView_DfModel_sep # SimpleView, SimpleView_DfModel


def get_dataset_sp_consider(train_data_path, val_data_path, sp_considered, logger, mapping=None):
    train_data = torch.load(train_data_path)
    val_data = torch.load(val_data_path)
    val_data.set_params(transforms=["none"])  # Turn off transforms for validation, test sets.

    assert train_data.classes == val_data.classes, "Error - train/val set species labels do not match. Check content and order"

    logger.info(f"The number of train data is: {len(train_data)}")
    logger.info(f"The number of val data is: {len(val_data)}")

    print('Training data:')
    print(train_data.counts)
    print('Total count: ', len(train_data))
    print('Species: ', train_data.classes)
    # print('Labels: ', train_data.labels)

    print('Validation data:')
    print(val_data.counts)
    print('Total count: ', len(val_data))
    print('Species: ', val_data.classes)
    # print('Labels: ', val_data.labels)

    if set(train_data.classes) != set(sp_considered):
        logger.info(f"only consider {len(sp_considered)} species: {sp_considered}")

        for specie in list(set(train_data.classes) - set(sp_considered)):
            print("Removing: {}".format(specie))
            train_data.remove_species(specie)
            val_data.remove_species(specie)

        print('Train Dataset:')
        print('Total count: ', len(train_data))
        print(train_data.counts)
        print('Species: ', train_data.classes)
        # print('Labels: ', train_data.labels)
        print()

        print('Validation Dataset:')
        print('Total count: ', len(val_data))
        print(val_data.counts)
        print('Species: ', val_data.classes)
        # print('Labels: ', val_data.labels)

        assert set(sp_considered) == set(train_data.classes)
    
    if mapping != None:
        train_data.group_species(mapping)
        val_data.group_species(mapping)

    print('Train Dataset:')
    print('Total count: ', len(train_data))
    print(train_data.counts)
    print('Species: ', train_data.classes)
    # print('Labels: ', train_data.labels)
    print()

    print('Validation Dataset:')
    print('Total count: ', len(val_data))
    print(val_data.counts)
    print('Species: ', val_data.classes)
    # print('Labels: ', val_data.labels)

    return train_data, val_data



def train(train_data_path, val_data_path, model_dir, params, logger, best_logger,
          fname_prefix=str(datetime.now())):
    """
    Trains a model

    train_file - location of train data file
    val_data - location of validation data file
    test_data - location of test data file

    saves trained model to disk at the folder model_dir/
    params to specify training parameters
    fname_prefix to determine model filenames - fname_prefix_best_test for best overall test accuracy etc.
    optional wandb logging by setting name of wandb project - should prompt for login automatically.
    """
    ##############
    # set device
    ##############
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ##############
    # load data
    ##############
    train_data, val_data = get_dataset_sp_consider(train_data_path, val_data_path, params["species_considered"], logger, params['mapping'])

    ##############
    # set random seed
    ##############
    torch.manual_seed(params['random_seed'])
    torch.cuda.manual_seed(params['random_seed'])
    torch.random.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])
    random.seed(params['random_seed'])

    ##############
    # create datalodar
    ##############
    # 1. train sampler:
    # whether use balanced sampling to ensure the balance sample of each class
    if params['train_sampler'] == "random":
        print("Using random/uniform sampling...")
        # Dataset without a sampler (uniform/deterministic if shuffled/unshuffled)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                                    shuffle=params["shuffle_dataset"])
    elif "balanced" in params['train_sampler']:
        if params['train_sampler'] == "balanced":
            print("Using balanced sampling...")
            labels = train_data.labels #Counts over
            counts = torch.bincount(labels) #Training set only
            label_weights = 1 / counts
        elif "balanced_T" in params['train_sampler']:
            samp_temperature = float(params['train_sampler'].split("_")[-1])
            print(f"Using balanced (with temprature = {samp_temperature}) sampling...")
            labels = train_data.labels  # Training set only
            counts = torch.bincount(labels)  # Counts over
            # get freq: refernce: DAFormer uda_dataset.py
            freq = counts / torch.sum(counts)
            freq = 1 - freq
            label_weights = torch.softmax(freq/samp_temperature, dim=-1)

        # Corresponding weight for each sample
        sample_weights = torch.stack([label_weights[label] for label in train_data.labels])

        # Replacement is true by default for the weighted sampler
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # Dataloader using the weighted sampler
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'],
                                                    sampler=train_sampler)

    # 2. val sampler
    # Val loader - never shuffled (shouldn't matter anyway)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=params['batch_size'])

    ##############
    # set model
    ##############
    # 1. model
    dropout_ratio = params.dfmodel["dropout_ratio"] if params.dfmodel.get("dropout_ratio") is not None else 0.
    if params["model"]=="DetailView_DfModel":
        modelname = params.dfmodel["name"]
        model = DetailView_DfModel(
            num_views=train_data.num_views,
            num_classes=len(params['species_considered']),
            num_feats=train_data.num_feats,
            selected_model=params.dfmodel["name"],
            imgnet_weight=params.dfmodel["pretrain_weight"],
            drop_rate=dropout_ratio
        )
    elif params["model"]=="DetailView_DfModel_sep":
        modelname = params.dfmodel["name"]
        model = DetailView_DfModel_sep(
            num_views=train_data.num_views,
            num_classes=len(params['species_considered']),
            num_feats=train_data.num_feats,
            selected_model=params.dfmodel["name"],
            imgnet_weight=params.dfmodel["pretrain_weight"],
            drop_rate=dropout_ratio
        )
    print(f"model dropout ratio: {dropout_ratio}")

    model = model.to(device=device)

    if params.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        # checkpoint = torch.load(args.pretrain)
        # start_epoch = checkpoint['epoch']
        # classifier.load_state_dict(checkpoint['model_state_dict'])
        model = load_checkpoint(model, logger, params.pretrain)
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    # 2. loss
    if params['loss_fn']=="cross-entropy":
        loss_fn = nn.CrossEntropyLoss()
        print("Using cross-entropy loss...")
    if params['loss_fn']=="smooth-loss":
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
        print("Using smooth-loss")
    if params['loss_fn']=="smooth-loss-timm":
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.2)
        print("Using smooth-loss-timm")


    # 3. lr
    if isinstance(params['learning_rate'], list):
        lr = params['learning_rate'][0]
        step_size = params['learning_rate'][1]
        gamma = params['learning_rate'][2]
    else:
        lr = params['learning_rate']

    # 4. optimizer
    if params['optimizer']=="sgd":
        print("Optimizing with SGD...")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=params['momentum'])
    elif params['optimizer']=="adam":
        print("Optimizing with AdaM...")
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif params['optimizer']=="adamw":
        print("Optimizing with AdamW...")
        from timm.optim import create_optimizer
        from argparse import Namespace
        params_opt = params.adamw_args
        params_opt.update({"opt": params.optimizer, "lr": lr})
        params_opt = Namespace(**dict(params_opt))
        optimizer = create_optimizer(params_opt, model)


    if params["optimizer"]!="adamw":
        if isinstance(params['learning_rate'], list):
            print("Using step LR scheduler...")
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        if params["warm_up"]!="none":
            warmup_scheduler = warmup_lr(optimizer, params["warm_up"], warm_period=int(params["epochs"] * len(train_loader) / 20))

    else: # adamw optimizer, for vit
        from timm.scheduler import create_scheduler
        from argparse import Namespace
        params_sched = params.adamw_lr
        params_sched.update({"opt": params.optimizer, "epochs":params.epochs, "seed":params.random_seed})
        params_sched = Namespace(**dict(params_sched))
        scheduler, _ = create_scheduler(params_sched, optimizer)



    ##############
    # start training
    ##############
    best_acc, best_min_acc, best_epoch = 0,0,0
    # best_val_acc, best_min_val_acc = 0,0
    for epoch in range(params['epochs']):  # loop over the dataset multiple times
        # Training loop============================================
        model.train()
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0), desc=f"{epoch}", total=len(train_loader), smoothing=0.9):
            depth_images = data['depth_images']
            labels = data['labels']
            feats  = data["feats"]
            depth_images = depth_images.to(device=device)
            labels = labels.to(device=device)
            feats  = feats.to(device=device)

            optimizer.zero_grad() # zero the parameter gradients

            # forward + backward + optimize
            outputs = model(depth_images, feats)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2))
                running_loss = 0.0

            # warm-up
            if params["warm_up"]!="none":
                if i < len(train_loader) - 1:
                    with warmup_scheduler.dampening():
                        pass

        # Test loop================================================
        num_train_correct, num_train_samples = 0, 0
        num_val_correct, num_val_samples = 0, 0

        running_train_loss, running_val_loss = 0, 0

        model.eval()
        with torch.no_grad():
            # Train set eval==============
            for data in train_loader:
                depth_images = data['depth_images']
                labels = data['labels']
                feats = data["feats"]

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)
                feats = feats.to(device=device)

                scores = model(depth_images, feats)
                _, predictions = scores.max(1)
                num_train_correct += (predictions == labels).sum()
                num_train_samples += predictions.size(0)

                running_train_loss += loss_fn(scores, labels)

            train_acc = float(num_train_correct) / float(num_train_samples)
            train_loss = running_train_loss / len(train_loader)

            # Val set eval===============
            all_labels = torch.tensor([]).to(device)
            all_predictions = torch.tensor([]).to(device)

            for data in val_loader:
                depth_images = data['depth_images']
                labels = data['labels']
                feats = data["feats"]

                depth_images = depth_images.to(device=device)
                labels = labels.to(device=device)
                feats = feats.to(device=device)

                scores = model(depth_images, feats)
                _, predictions = scores.max(1)

                all_labels = torch.cat((all_labels, labels))
                all_predictions = torch.cat((all_predictions, predictions))

                num_val_correct += (predictions == labels).sum()
                num_val_samples += predictions.size(0)

                running_val_loss += loss_fn(scores, labels)

            val_acc = float(num_val_correct) / float(num_val_samples)
            val_loss = running_val_loss / len(val_loader)

            print(f'{epoch} OVERALL (Val): Got {num_val_correct} / {num_val_samples} with accuracy {val_acc * 100:.2f}'
                  f' (best_val_acc={best_acc} in {best_epoch})')
            logger.info(f"{epoch} OVERALL: lr: {optimizer.param_groups[0]['lr']}, "
                        f"train_loss: {train_loss:.8f}, val_loss: {val_loss:.8f}, "
                        f"train_acc: {train_acc * 100:.2f}, val_acc: {val_acc * 100:.2f}, "
                        f"best_acc: {best_acc * 100:.2f}, best_epoch: {best_epoch}")

            cm = confusion_matrix(all_labels.cpu(), all_predictions.cpu())
            totals = cm.sum(axis=1)

            accs = np.zeros(len(totals))
            for i in range(len(totals)):
                accs[i] = cm[i, i] / totals[i]
                print(
                    f"{list(train_data.classes.keys())[i]}: "
                    f"Got {cm[i, i]}/{totals[i]} with accuracy {accs[i] * 100:.2f}")

            logger.info(f"{epoch} each lass: {accs}")

            # save model==============
            if val_acc >= best_acc:
                # best_model_state = copy.deepcopy(model.state_dict())
                best_acc = val_acc
                best_epoch = epoch
                best_logger.info(cm)
                best_logger.info(f'{best_epoch}::{best_acc}\n')
                save_checkpoints(model.state_dict(), best_acc, model_dir,
                                 epoch, minibatch=i, # i here is always equal to the number of class (because acc calculation uses i)
                                 fname_suffix="_best_acc")

            # if min(accs) >= best_min_acc:
            #     # best_min_model_state = copy.deepcopy(model.state_dict())
            #     best_min_acc = min(accs)
            #     save_checkpoints(model.state_dict(), best_min_acc, model_dir,
            #                      epoch, minibatch=i,
            #                      fname_suffix="_best_min_acc")


        if params["warm_up"]!="none":
            with warmup_scheduler.dampening():
                scheduler.step()
        elif "vit" in modelname:
            scheduler.step(epoch)
        else:
            scheduler.step()

    print('Finished Training')
    logger.info("Finish Training")
