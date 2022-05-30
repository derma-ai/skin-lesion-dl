import os
import numpy as np
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
    print(samples_per_class)
    print(dataset.class_to_idx)


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

if __name__ == "__main__":
    main()
