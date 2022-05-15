from argparse import ArgumentParser
from pyparsing import string
import os


import pytorch_lightning as pl
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torchsummary import summary

from subset import Subset
import model_loader

def set_seed(seed=15):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def setup_data():
    # This somehow makes the performance terrible.
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])
    root = os.path.join("/", "space", "derma-data")
    dataset = datasets.ImageFolder(root, transform=transforms.ToTensor())


    train_data_idx, val_data_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)

    train_data = Subset(dataset, train_data_idx, train_transform)
    val_data = Subset(dataset, val_data_idx, val_transform)
    print(len(train_data))
    print(len(val_data))
    return train_data, val_data


def setup_data_loaders(train_data, val_data, batch_size):
    train_sampler = get_train_sampler(train_data)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               drop_last=False,
                                               shuffle=True,
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

def get_train_sampler(dataset):
    dataset.

def train(hparams,
          version_name,
          checkpoint=None):

    train_data, val_data = setup_data()
    hparams["c"] = len(train_data.classes)
    train_loader, val_loader = setup_data_loaders(train_data, val_data, hparams["b"])

    logger = TensorBoardLogger(version=version_name, save_dir="./")
    trainer = pl.Trainer(gpus=[1],
                         max_epochs=hparams["e"],
                         logger=logger
                         )

    model = model_loader.load(hparams, checkpoint)
    trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint(f'model_{version_name}.ckpt')


def main():
    parser = ArgumentParser()
    parser.add_argument('-e', '--max_epochs', type=int, dest='max_epochs',
                        default=10, help="Number of training epochs")
    parser.add_argument('-b', '--batch_size', type=int,
                        dest='batch_size', default=16, help="Batch size")
    parser.add_argument('-lr', '--learning_rate', type=float,
                        dest='learning_rate', default=1e-3, help="Learning rate")
    parser.add_argument('-wd', '--weight_decay', type=float,
                        default=1e-8, dest="weight_decay", help="Weight decay")
    parser.add_argument('-ex', '--experiment', type=str,
                        default="", dest="experiment_name", help="Experiment name")
    parser.add_argument('-m', '--model', type=str,
                        default="", dest="model", help="Model name")
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None,
                        dest="checkpoint", help="Call model from checkpoint by version name")
    args = parser.parse_args()

    hparams = {
        "e": args.max_epochs,
        "b": args.batch_size,
        "lr": args.learning_rate,
        "wd": args.weight_decay,
        "m": args.model
    }

    set_seed()

    train(hparams,
          version_name=f'b={args.batch_size}-lr={args.learning_rate}-wd={args.weight_decay}-{args.experiment_name}',
          checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
