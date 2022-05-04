from argparse import ArgumentParser
from pyparsing import string
import os


import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torchsummary import summary
import glob

from subset import Subset

from model import ResNetClassifier


def setup_data(test_mode):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    root = os.path.join(os.path.expanduser("~"), "share-all", "derma-data", "archive")
    dataset = datasets.ImageFolder(root)
    train_data_idx, val_data_idx = train_test_split(
        list(range(len(dataset))), test_size=0.2, stratify=dataset.targets)
    
    if test_mode:
        # Limit dataset and make validation and test set the same
        train_data_idx = [32*n for n in range(512)]
        val_data_idx = [32*n for n in range(512)]

    train_data = Subset(dataset, train_data_idx, train_transform)
    val_data = Subset(dataset, val_data_idx, val_transform)
    return train_data, val_data

def train(batch_size=32,
          learning_rate=1e-3,
          weight_decay=1e-8,
          max_epochs=100,
          version_name="1",
          test_mode = False,
          checkpoint = None):

    train_data, val_data = setup_data(test_mode=test_mode)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               drop_last=False,
                                               shuffle=True,
                                               timeout= 30000,
                                               pin_memory = True)



    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             num_workers=8,
                                             drop_last=False,
                                             shuffle=False,
                                             timeout=30000,
                                             pin_memory=True)
    
    logger = TensorBoardLogger(version=version_name, save_dir="./")
    trainer = pl.Trainer(devices=1,
                         accelerator='gpu',
                         max_epochs=max_epochs,
                         logger=logger
                         )
    if(checkpoint is None):
        print("Checkpoint is None")
        model = ResNetClassifier(learning_rate=learning_rate,
                                 weight_decay=weight_decay, 
                                 num_classes=len(train_data.classes))
    elif(len(checkpoint) == 0):
        latest_ckpt = max(glob.glob('./checkpoints/*.ckpt'), key=os.path.getctime)
        model = ResNetClassifier.load_from_checkpoint(latest_ckpt,
                                                      learning_rate=learning_rate,
                                                      weight_decay=weight_decay, 
                                                      num_classes=len(train_data.classes))
    elif(len(checkpoint)>0):
        ckpt = max(glob.glob(f"./checkpoints/*{checkpoint}.ckpt"), key=os.path.getctime)
        print(ckpt)
        model = ResNetClassifier.load_from_checkpoint(ckpt,
                                                      learning_rate=learning_rate,
                                                      weight_decay=weight_decay, 
                                                      num_classes=len(train_data.classes))
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
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-8, dest="weight_decay", help="Weight decay")
    parser.add_argument('-c', '--comment', type=str, default="", dest="comment", help="Comment")
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, dest="checkpoint", help="Call model from checkpoint by version name")    
    args = parser.parse_args()

    train(batch_size=args.batch_size, 
          max_epochs=args.max_epochs,
          learning_rate=args.learning_rate, 
          weight_decay=args.weight_decay,
          version_name=f'bs={args.batch_size}-lr={args.learning_rate}-wd={args.weight_decay}-{args.comment}',
          checkpoint=args.checkpoint,
          test_mode=False)


if __name__ == "__main__":
    main()
