from argparse import ArgumentParser
import os

import pytorch_lightning as pl
import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from experiment_builder import ExperimentBuilder
import data_handler

def set_seed(seed=15):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(gpu,
          hparams,
          checkpoint=None):

    train_data, val_data, weights = data_handler.setup_data(hparams)

    num_classes = len(train_data.classes)
    train_loader, val_loader = data_handler.setup_data_loaders(
        train_data, val_data, hparams["b"], hparams["osr"])

    builder = ExperimentBuilder(
        hparams,
        num_classes=num_classes,
        class_weights=weights
    )
    model = builder.create(checkpoint)

    version_name=f'{hparams["ex"]}-{hparams["m"]}'
    logger = TensorBoardLogger(version=version_name,
                                save_dir="./",
                                log_graph=True
                                )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='./checkpoints',
                                                       filename=f'{version_name}'+'-{epoch}' + f'-bs:{hparams["b"]}' + '{val_acc:.2f}',
                                                       save_top_k=3,
                                                       every_n_epochs=5,
                                                       save_on_train_epoch_end=True,
                                                       monitor='val_acc',
                                                       mode='max'
                                                       )
    trainer = pl.Trainer(gpus=[gpu],
                         max_epochs=hparams["e"],
                         logger=logger,
                         callbacks=[checkpoint_callback]
                         )
    

    if hparams["lrf"] == 1:
         # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader, early_stop_threshold=3.0, max_lr=1e-4)
        new_lr = lr_finder.suggestion()
        hparams["lr"] = new_lr
        print(f"Found LR: {new_lr}")
    
        builder = ExperimentBuilder(
            hparams,
            num_classes=num_classes,
            class_weights=weights
        )
        model = builder.create(checkpoint)

    logger.log_hyperparams(hparams)
    trainer.fit(model, train_loader, val_loader)

def main():
    parser = ArgumentParser()
    parser.add_argument('-e', '--max_epochs', type=int, dest='max_epochs',
                        default=10, help="Number of training epochs")
    parser.add_argument('-b', '--batch_size', type=int,
                        dest='batch_size', default=16, help="Batch size")
    parser.add_argument('-lr', '--learning_rate', type=float,
                        dest='learning_rate', default=1e-3, help="Learning rate")
    parser.add_argument('-lrf', '--learning_rate_finder', type=int,
                        dest='learning_rate_finder', default=0, help="Learning rate finder enabled")
    parser.add_argument('-wd', '--weight_decay', type=float,
                        default=1e-8, dest="weight_decay", help="Weight decay")
    parser.add_argument('-ex', '--experiment', type=str,
                        default="", dest="experiment_name", help="Experiment name")
    parser.add_argument('-m', '--model', type=str,
                        default="", dest="model", help="Model name")
    parser.add_argument("-t", "--transforms", type=str, default=None, dest="transforms",
                        help="Comma separated list of transform flags, e.g. /'r,hflip,vflip/'")
    parser.add_argument("-osr", "--over_sampling_rate", type=float, default=1, dest="osr",
                        help="How many multiples of dataset size should be oversampled using a weighted sampler. At 1, no oversampling or weighted sampling is done.")
    parser.add_argument("-l", "--loss", type=str, default="", dest="loss",
                        help="Loss function'")
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None,
                        dest="checkpoint", help="Call model from checkpoint by version name")
    parser.add_argument('-gpu', '--gpu', type=int, default=None,
                        dest="gpu", help="On which GPU to train")
    args = parser.parse_args()

    print(f"Using {args.model} model")
    hparams = {
        "e": args.max_epochs,
        "b": args.batch_size,
        "lr": args.learning_rate,
        "wd": args.weight_decay,
        "m": args.model,
        "ex": args.experiment_name,
        "t": args.transforms,
        "l": args.loss,
        "osr": args.osr,
        "lrf": args.learning_rate_finder
    }

    set_seed()
    train(args.gpu,
          hparams,
          checkpoint=args.checkpoint
          )


if __name__ == "__main__":
    main()
