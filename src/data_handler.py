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

    train_transform = get_train_transforms(hparams["t"], hparams["d"])
    root_path = os.path.join("/","space","derma-data","isic_2019")
    if hparams.get('d') == 'original':
        root = os.path.join(root_path, "clean")
    else:    
        root = os.path.join(root_path,"preprocessed")
    print(f"Using the {hparams['d']} dataset.")
    print(f"Trainig transforms: {train_transform}")
    dataset = datasets.ImageFolder(root, base_transforms)

    train_data_idx, val_data_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)
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
        elif flag == "gaussblur":
            transform = transforms.GaussianBlur()
        elif flag == "colorjitter": # worled
            transform = transforms.ColorJitter()
        elif flag == "grayscale": # worked
            transform = transforms.Grayscale(num_output_channels=3)
        elif flag == "randperspective": # worked 
            transform = transforms.RandomPerspective()
        elif flag == "randposterize":
            transform = nn.ModuleList([transforms.ToPILImage(mode="RGB"),transforms.RandomPosterize(bits=5),transforms.ToTensor()])
        elif flag == "randadjustsharpness":
            transform = transforms.RandomAdjustSharpness(np.random.randint(5))
        elif flag == "randomequalize": # currently not working because of TypeError: expects [0,255] but got [0,1] due to ToTensor
            transform = nn.ModuleList([transforms.ToPILImage(mode="RGB"),transforms.RandomEqualize(),transforms.ToTensor()])
        elif flag == "norm":
            if dataset_flag == "original":
                transform = transforms.Normalize(mean=[0.624, 0.520, 0.504], std=[0.242, 0.223, 0.231])
            elif dataset_flag == "preprocessed":
                transform = transforms.Normalize(mean=[0.657, 0.548, 0.532], std=[0.204, 0.197, 0.208])
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
