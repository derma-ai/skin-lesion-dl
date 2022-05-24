import os
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