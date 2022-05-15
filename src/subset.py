from torch.utils.data import Dataset


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Writing custom variant, as the default Subset implementation
    does not provide transforms.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset[indices]
        # We just use the classes from the source dataset
        self.classes = self.dataset.classes
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample[0]), sample[1]
        return sample

    def __len__(self):
        return len(self.dataset)
