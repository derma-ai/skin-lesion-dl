import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split

from subset import Subset

def setup_data(hparams):
    # This somehow makes the performance terrible.
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    )

    train_transform = get_train_transforms(hparams["t"])
    if hparams['path'] != None:
        root = os.path.join(hparams['path'])
    else:    
        root = os.path.join("/", "space", "derma-data")
    print(f"Trainig transforms: {train_transform}")
    dataset = datasets.ImageFolder(root, base_transforms)

    train_data_idx, val_data_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)
    weights, _ = compute_weights(dataset)
    train_data = Subset(dataset, train_data_idx, train_transform)
    val_data = Subset(dataset, val_data_idx)
    return train_data, val_data, weights


def get_train_transforms(flags=None):
    if flags is None or len(flags) == 0:
        # Default transforms
        return None
    return nn.Sequential(*build_transform_list(flags))

def build_transform_list(flags):
    transform_flags = flags.split(",")
    transforms_list = []
    transform = None # If there is no matching transform should remain None
    for flag in transform_flags:
        if flag == "r":
            transform = transforms.RandomRotation(degrees=(0, 180))
        elif flag == "vflip":
            transform = transforms.RandomVerticalFlip()
        elif flag == "hflip":
            transform = transforms.RandomHorizontalFlip()
            # add new cases here
        if transform is not None:
            transforms_list.append(transform)
    return transforms_list


def compute_weights(dataset):
    class_sample_count = np.unique(dataset.targets, return_counts=True)[1]
    weights = 1.0 / class_sample_count
    weights_per_sample = np.array([weights[t] for t in dataset.targets])
    return torch.from_numpy(weights).float(), torch.from_numpy(weights_per_sample).float()


def setup_data_loaders(train_data, val_data, batch_size, over_sampling_rate):
    if (over_sampling_rate <= 1):
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                num_workers=8,
                                                drop_last=False,
                                                timeout=30000,
                                                pin_memory=True)
    else:
        print("Use oversampling and weighted sampler")
        _, weights_per_sample = compute_weights(train_data.dataset)
        weights_per_sample = weights_per_sample[train_data.indices]
        weighted_sampler = WeightedRandomSampler(
            weights=weights_per_sample, num_samples=int(len(train_data) * over_sampling_rate))
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                sampler=weighted_sampler,
                                                num_workers=8,
                                                drop_last=False,
                                                timeout=30000,
                                                pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             num_workers=8,
                                             drop_last=False,
                                             shuffle=False,
                                             timeout=30000,
                                             pin_memory=True)
    return train_loader, val_loader
