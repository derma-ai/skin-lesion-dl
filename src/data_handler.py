import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split

from subset import Subset

def setup_data(hparams, path=None):
    # This somehow makes the performance terrible.
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    base_transforms = transforms.Compose([
        transforms.ToTensor()   
    ]
    )

    train_transform = get_train_transforms(hparams["t"])
    if hparams.get('path') != None:
        root = os.path.join(hparams['path'])
    else:    
        root = os.path.join("/", "space", "derma-data", "isic_2019", "clean")
    print(f"Trainig transforms: {train_transform}")
    dataset = datasets.ImageFolder(root, base_transforms)

    train_data_idx, val_data_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)
    # weight_scheme == 1 to use 1/n for WCE Loss 
    weights, _ = compute_weights(dataset, 1)
    train_data = Subset(dataset, train_data_idx, train_transform)
    val_data = Subset(dataset, val_data_idx, transforms.Resize((224,224)))
    return train_data, val_data, weights, dataset


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
        elif flag == "gaussblur":
            transform = transforms.GaussianBlur()
        elif flag == "colorjitter":
            transform = transforms.ColorJitter()
        elif flag == "grayscale":
            transform = transforms.Grayscale(num_output_channels=3)
        elif flag == "randperspective":
            transform = transforms.RandomPerspective()
        elif flag == "randposterize":
            transform = transforms.Compose([transforms.ToPILImage(mode="RGB"),transforms.RandomPosterize(bits=5),transforms.ToTensor()])
        elif flag == "randadjustsharpness":
            transform = transforms.RandomAdjustSharpness(np.randint(5))
        elif flag == "randomequalize": # currently not working because of TypeError: expects [0,255] but got [0,1] due to ToTensor
            transform = transforms.Compose([transforms.ToPILImage(mode="RGB"),transforms.RandomEqualize(),transforms.ToTensor()])
            # add new cases here
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
