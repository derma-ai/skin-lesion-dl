from argparse import ArgumentParser
import imp
from pyparsing import string

import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torchsummary import summary

from subset import Subset

from model import ResNetClassifier

def setup_data():

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224))
    ])
    root = "./../archive"
    dataset = datasets.ImageFolder(root, train_transform)
    train_data_idx, val_data_idx = train_test_split(list(range(len(dataset))),test_size=0.2, stratify=dataset.targets)
    train_data = Subset(dataset, train_data_idx, train_transform)
    val_data = Subset(dataset, val_data_idx, val_transform)
    return train_data, val_data

def train(batch_size=16,
          learning_rate=1e-3,
          max_epochs=100,
          version_name = "1"):

    train_data, val_data = setup_data()

    train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           num_workers=8,
                                           drop_last=False,
                                           shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        drop_last=False,
                                        shuffle=False)

    model = ResNetClassifier(learning_rate=learning_rate,
                            num_classes=len(train_data.classes))


    trainer = pl.Trainer(devices=1,
                         accelerator='gpu',
                         max_epochs=max_epochs,
                         log_every_n_steps=50,
                         flush_logs_every_n_steps= 300
                        )
    logger = TensorBoardLogger(version=version_name, name="tensorboard_logs")
    trainer.fit(model, train_loader, val_loader, logger)

    trainer.save_checkpoint(f'model_{version_name}.ckpt')

def main():

    parser = ArgumentParser()
    parser.add_argument('-e', '--max_epochs', type=int, dest='max_epochs', default=10, help="Number of training epochs")
    parser.add_argument('-b', '--batch_size', type=int, dest='batch_size', default=16, help="Batch size")
    parser.add_argument('-lr', '--learning_rate', type=float, dest='learning_rate', default=1e-3, help="Learning rate")
    parser.add_argument('-v', '--version', type= string, dest='version_name', default="1", help="Name of current model version")

    args = parser.parse_args()

    train(batch_size=args.batch_size, max_epochs=args.max_epochs, learning_rate=args.learning_rate, version_name=args.version_name)
    
if __name__ == "__main__":
    main()