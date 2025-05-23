import os
from platform import platform
import platform

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split

from data.subset import Subset
from data.color_constancy import compute_color_constancy

def setup_data(hparams, path=None):
    # This somehow makes the performance terrible.
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    base_transforms = transforms.Compose([
        transforms.ToTensor()
       #transforms.Lambda(compute_color_constancy)
    ]
    )

    train_transform = get_train_transforms(hparams["t"], hparams["d"])
    if(platform.system()=="Linux"):
        root_path = os.path.join("/","space","derma-data","isic_2019")
    else:
        root_path = os.path.expanduser("~/share-all/derma-data/")
    if hparams.get('d') == 'original':
        root = os.path.join(root_path, "clean")
    else:    
        root = os.path.join(root_path,"preprocessed")
    print(f"Using the {hparams['d']} dataset.")
    print(f"Trainig transforms: {train_transform}")
    dataset = datasets.ImageFolder(root, base_transforms)

    train_data_idx, val_data_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)
    np.save('train_idx_window_eval', train_data_idx)
    np.save('val_idx_window_eval', val_data_idx)
    # weight_scheme == 1 to use 1/n for WCE Loss 
    weights, _ = compute_weights(dataset, 1)
    train_data = Subset(dataset, train_data_idx, train_transform)
    val_data = Subset(dataset, val_data_idx, transforms.Resize((224,224)))
    return train_data, val_data, weights


def get_train_transforms(flags=None, dataset_flag=None):
    if flags is None or len(flags) == 0:
        # Default transforms
        return None
    return nn.Sequential(*build_transform_list(flags, dataset_flag))

def build_transform_list(flags, dataset_flag):
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
        elif flag == "gaussblur": # worked
            transform = transforms.GaussianBlur(kernel_size= (9,9))
        elif flag == "colorjitter": # worked
            transform = transforms.ColorJitter()
        elif flag == "grayscale": # worked
            transform = transforms.Grayscale(num_output_channels=3)
        elif flag == "randperspective": # worked 
            transform = transforms.RandomPerspective()
        elif flag == "randadjustsharpness":
            transform = transforms.RandomAdjustSharpness(np.random.randint(5))
        elif flag == "norm":
            if dataset_flag == "original":
                transform = transforms.Normalize(mean=[0.624, 0.520, 0.504], std=[0.242, 0.223, 0.231])
            elif dataset_flag == "preprocessed": 
                transform = transforms.Normalize(mean=[0.599, 0.578, 0.566], std=[0.185, 0.202, 0.211])
        if transform is not None:
            transforms_list.append(transform)
    transforms_list.append(transforms.Resize((224, 224)))
    return transforms_list


def compute_weights(dataset, weight_scheme = 0):
    class_sample_count = np.unique(dataset.targets, return_counts=True)[1]
    weights = 1.0 / class_sample_count
    # no weight scheme used
    if(weight_scheme == 0):
        return None, None
    elif(weight_scheme == 2):
        weights_avg = np.average(weights)
        weights = weights + weights_avg
        # Medically especially important classes BCC:1, MEL:4, NV:5 and SCC:6 recieve higher weights
    elif(weight_scheme == 3):
        weights = np.array([1,4,1,1,4,2,4,1])
    weights_per_sample = np.array([weights[t] for t in dataset.targets])
    return torch.from_numpy(weights).float(), torch.from_numpy(weights_per_sample).float()


def setup_data_loaders(train_data, val_data, batch_size, over_sampling_rate, weight_scheme):
    if (weight_scheme == 0):
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                num_workers=8,
                                                drop_last=False,
                                                timeout=30000,
                                                pin_memory=True)
    else:
        print("Use oversampling and weighted sampler")
        _, weights_per_sample = compute_weights(train_data.dataset, weight_scheme)
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
