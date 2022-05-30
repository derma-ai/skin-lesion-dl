import os
from statistics import variance
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def compute_existing_resolutions():
    root = os.path.join(os.path.expanduser("~"), "share-all", "derma-data", "archive")
    dataset = datasets.ImageFolder(root, transform=transforms.ToTensor())
    resolution_sizes = []
    print(len(dataset))
    for i in range(len(dataset)):
        image = dataset[i][0]
        if(i%1000 == 0):
            print(f"Index {i}")
        if not image.shape in resolution_sizes:
            resolution_sizes.append(image.shape)
    print(resolution_sizes)



def main(device):
    dataset = setup_dataset()
    samples_per_class = compute_samples_per_class(dataset)
    print_samples_per_class(samples_per_class, dataset)

    per_channel_mean, per_channel_variance = compute_per_channel_mean_variance(dataset, device)
    print_mean_variance_per_channel(per_channel_mean, per_channel_variance)

def setup_dataset():
    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ]
    )

    root = os.path.join("/", "space", "derma-data")
    dataset = datasets.ImageFolder(root, base_transforms)
    return dataset

def compute_samples_per_class(dataset):
    class_sample_count = np.unique(dataset.targets, return_counts=True)[1]
    return class_sample_count

def print_samples_per_class(samples_per_class, dataset):
    print(f"Dataset contains a total of {len(dataset.classes)} classes, the samples are distributed over the classes in the following way:")
    for class_name, sample_count in zip(dataset.classes, samples_per_class):
        print(f"{class_name}: {sample_count}")
    
def compute_per_channel_mean_variance(dataset, device):
    mean, variance = compute_loader_gpu(dataset, device)
    return mean, variance

def compute_loader_gpu(dataset, device):
    mean = torch.zeros(3).to(device)
    variance = torch.zeros(3).to(device)
    loader = DataLoader(dataset, batch_size=32, num_workers=40)
    pixels_per_channel = dataset[0][0].shape[1] * dataset[0][0].shape[2]

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        mean += x.mean(dim=[0,2,3])
    mean = mean / len(dataset)
    mean = mean.to(device)
    for batch_idx, (x,y) in enumerate(loader):
        x = x.to(device)
        variance +=  (x - mean[:, None, None]).pow(2).sum(dim=[0,2,3]) / pixels_per_channel
    variance = variance / len(dataset)
    mean = mean.cpu()
    variance = variance.cpu()
    return mean, variance

def print_mean_variance_per_channel(per_channel_mean, per_channel_variance):
    print(f"We have a per channel mean of: {per_channel_mean}")
    print(f"We have a per channel variance of: {per_channel_variance}")

if __name__ == "__main__":
    device = "cuda:1"
    main(device)
