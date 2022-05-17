import glob
import os

import torchvision.models as models
import torch.nn as nn

from model import Classifier


def load(hparams, checkpoint, class_weights=None):
    if(checkpoint is None):
        print("Checkpoint is None")
        model = Classifier(
            model_name=hparams["m"],
            learning_rate=hparams["lr"],
            weight_decay=hparams["wd"],
            num_classes=hparams["c"],
            class_weights = class_weights)
    elif(len(checkpoint) == 0):
        latest_ckpt = max(glob.glob('./checkpoints/*.ckpt'),
                          key=os.path.getctime)
        model = Classifier.load_from_checkpoint(latest_ckpt,
                                                model_name=hparams["m"],
                                                learning_rate=hparams["lr"],
                                                weight_decay=hparams["wd"],
                                                num_classes=hparams["c"],
                                                class_weights = class_weights)
    else:
        ckpt = max(
            glob.glob(f"./checkpoints/*{checkpoint}.ckpt"), key=os.path.getctime)
        print(ckpt)
        model = Classifier.load_from_checkpoint(ckpt,
                                                model_name=hparams["m"],
                                                learning_rate=hparams["lr"],
                                                weight_decay=hparams["wd"],
                                                num_classes=hparams["c"],
                                                class_weights = class_weights)
    return model
