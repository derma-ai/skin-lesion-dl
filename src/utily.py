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
    start_no_loader = torch.cuda.Event(enable_timing=True)
    end_no_loader = torch.cuda.Event(enable_timing=True)
    start_loader_cpu = torch.cuda.Event(enable_timing=True)
    end_loader_cpu = torch.cuda.Event(enable_timing=True)
    start_loader_gpu = torch.cuda.Event(enable_timing=True)
    end_loader_gpu = torch.cuda.Event(enable_timing=True)
    
    start_no_loader.record()
    # compute_simple(dataset)
    torch.cuda.synchronize()
    end_no_loader.record()

    start_loader_cpu.record()
    compute_loader(dataset)
    torch.cuda.synchronize()
    end_loader_cpu.record()

    start_loader_gpu.record()
    compute_loader_gpu(dataset)
    torch.cuda.synchronize()
    end_loader_gpu.record()

    print(f"Time required without dataloader: {start_no_loader.elapsed_time(end_no_loader)}")
    print(f"Time required with dataloader and cpu: {start_loader_cpu.elapsed_time(end_loader_cpu)}")
    print(f"Time required with dataloader and gpu: {start_loader_gpu.elapsed_time(end_loader_gpu)}")

def compute_simple(dataset):
    mean = torch.zeros(3)
    pixels_per_channel = dataset[idx][0].shape[1] * dataset[idx][0].shape[2]
    variance = torch.zeros(3)
    for idx in range(len(dataset)):
        mean += dataset[idx][0].mean(dim=[1,2])
    mean = mean / len(dataset)
    for idx in range(len(dataset)):
        variance +=  (dataset[idx][0] - mean).pow(2).sum(dim=[1,2]) / pixels_per_channel
    variance = variance / len(dataset)
    return mean, variance

def compute_loader(dataset):
    mean = torch.zeros(3)
    variance = torch.zeros(3)
    loader = DataLoader(dataset, batch_size=64, num_workers=8)
    pixels_per_channel = dataset[0][0].shape[1] * dataset[0][0].shape[2]

    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            print(batch.shape)  
        mean += batch[0].mean(dim=[0,2,3])
    mean = mean / len(dataset)
    for batch_idx, batch in enumerate(loader):
        variance +=  (batch - mean).pow(2).sum(dim=[1,2]) / pixels_per_channel
    variance = variance / len(dataset)
    return mean, variance

def compute_loader_gpu(dataset):
    mean = 0

if __name__ == "__main__":
    main()
