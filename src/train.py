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


def train(hparams,
          version_name,
          checkpoint=None):

    train_data, val_data, weights = data_handler.setup_data(hparams)

    hparams["c"] = len(train_data.classes)
    train_loader, val_loader = data_handler.setup_data_loaders(
        train_data, val_data, hparams["b"], hparams["osr"])

    builder = ExperimentBuilder(
        extractor_type=hparams["m"],
        loss=hparams["l"],
        num_classes=hparams["c"],
        learning_rate=hparams["lr"],
        class_weights=weights
    )

    model = builder.create(checkpoint)

    logger = TensorBoardLogger(version=version_name,
                               save_dir="./",
                               log_graph=True
                               )

    trainer = pl.Trainer(gpus=[1],
                         max_epochs=hparams["e"],
                         logger=logger
                         )

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
    parser.add_argument("-t", "--transforms", type=str, default=None, dest="transforms",
                        help="Comma separated list of transform flags, e.g. /'r,hflip,vflip/'")
    parser.add_argument("-osr", "--over_sampling_rate", type=float, default=1, dest="osr",
                        help="How many multiples of dataset size should be oversampled using a weighted sampler. At 1, no oversampling or weighted sampling is done.")
    parser.add_argument("-l", "--loss", type=str, default="", dest="loss",
                        help="Loss function'")
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None,
                        dest="checkpoint", help="Call model from checkpoint by version name")
    args = parser.parse_args()

    hparams = {
        "e": args.max_epochs,
        "b": args.batch_size,
        "lr": args.learning_rate,
        "wd": args.weight_decay,
        "m": args.model,
        "t": args.transforms,
        "l": args.loss,
        "osr": args.osr
    }

    set_seed()
    train(hparams,
          version_name=f'b={args.batch_size}-lr={args.learning_rate}-wd={args.weight_decay}-{args.experiment_name}',
          checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
