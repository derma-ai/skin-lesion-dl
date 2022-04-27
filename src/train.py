import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from sklearn.svm import SVC

import pytorch_lightning as pl
import torch

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from model import SimpleClassifier

def setup_data():
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.Normalize(norm_mean, norm_std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),
        transforms.Normalize(norm_mean, norm_std),
    ])

    train_data = torchvision.datasets.CIFAR10(root='./', download=True, transform=train_transform)
    val_data = torchvision.datasets.CIFAR10(root='./', download=True, train=False, transform=val_transform)
    
    return train_data, val_data

def train(batch_size=16,
          learning_rate=1e-3,
          max_epochs=100):

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

    model = SimpleClassifier(learning_rate=learning_rate,
                            num_classes=len(train_data.classes))

    trainer = pl.Trainer(devices=1,
                         accelerator='gpu',
                         max_epochs=max_epochs,
                        )

    trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint('saved_model.ckpt')

def main():

    parser = ArgumentParser()
    parser.add_argument('-e', '--max_epochs', type=int, dest='max_epochs', default=10, help="Number of training epochs")
    parser.add_argument('-b', '--batch_size', type=int, dest='batch_size', default=16, help="Batch size")
    parser.add_argument('-lr', '--learning_rate', type=float, dest='learning_rate', default=1e-3, help="Learning rate")

    args = parser.parse_args()

    train(batch_size=args.batch_size, max_epochs=args.max_epochs, learning_rate=args.learning_rate)
    
if __name__ == "__main__":
    main()