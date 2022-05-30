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



def main():
    dataset = setup_dataset()
    samples_per_class = compute_samples_per_class(dataset)
    print_samples_per_class(samples_per_class, dataset)

    per_channel_mean, per_channel_variance = compute_per_channel_statistics(dataset)

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
    
def compute_per_channel_statistics(dataset):
    start_loader_gpu = torch.cuda.Event(enable_timing=True)
    end_loader_gpu = torch.cuda.Event(enable_timing=True)
    
    print("Start GPU computation")
    # Issue: Black borders in some images will influence mean even though they provide no real information, we should remove the black borders!
    start_loader_gpu.record()
    mean, variance = compute_loader_gpu(dataset, 'cuda:1')
    end_loader_gpu.record()
    torch.cuda.synchronize()

    print(f"Time required with dataloader and gpu: {start_loader_gpu.elapsed_time(end_loader_gpu)}")
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

if __name__ == "__main__":
    main()
