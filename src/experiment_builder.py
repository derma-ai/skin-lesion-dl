import glob
import os
import glob

import torchvision.models as models
import torch.nn as nn

from model import Classifier


class ExperimentBuilder:

    def __init__(self,
                 hparams,
                 num_classes=8,
                 class_weights=None
                 ):
        self.hparams = hparams
        self.num_classes = num_classes
        self.class_weights = class_weights

        layers = list(self.get_model().children())
        convolution_layers = layers[:-1]
        self.extractor = nn.Sequential(*convolution_layers)
        self.classifier = nn.Linear(
            layers[-1][1].in_features, self.num_classes)
        self.loss = self.configure_loss()

    def create(self, checkpoint):
        if(checkpoint is None):
            model = Classifier(
                hparams=self.hparams,
                extractor=self.extractor,
                classifier=self.classifier,
                loss=self.loss,
                num_classes=self.num_classes
            )
        else:
            model = self.load_checkpoint(checkpoint)

        return model

    def load_checkpoint(self, checkpoint):
        if(len(checkpoint) == 0):
            ckpt = max(glob.glob('./checkpoints/*.ckpt'), key=os.path.getctime)
        else:
            ckpt = max(
                glob.glob(f"./checkpoints/*{checkpoint}.ckpt"), key=os.path.getctime)
        model = Classifier.load_from_checkpoint(ckpt,
                                                hparams=self.hparams,
                                                extractor=self.extractor,
                                                classifier=self.classifier,
                                                loss=self.loss,
                                                num_classes=self.num_classes,
                                                learning_rate=self.learning_rate,
                                                weight_decay=self.weight_decay,
                                                )
        return model

    def get_model(self, pretrained=True):
        if self.hparams["m"] == "efficientnet_b0":
            return models.efficientnet_b0(pretrained)
        if self.hparams["m"] == "efficientnet_b1":
            return models.efficientnet_b1(pretrained)
        if self.hparams["m"] == "efficientnet_b2":
            return models.efficientnet_b2(pretrained)
        if self.hparams["m"] == "efficientnet_b3":
            return models.efficientnet_b3(pretrained)
        if self.hparams["m"] == "efficientnet_b4":
            return models.efficientnet_b4(pretrained)
        if self.hparams["m"] == "efficientnet_b5":
            return models.efficientnet_b5(pretrained)
        if self.hparams["m"] == "efficientnet_b6":
            return models.efficientnet_b6(pretrained)
        if self.hparams["m"] == "efficientnet_b7":
            return models.efficientnet_b7(pretrained)
        if self.hparams["m"] == "resnet50":
            return models.resnet50(pretrained)
        if self.hparams["m"] == "resnet101":
            return models.resnet101(pretrained)
        if self.hparams["m"] == "densenet121":
            return models.densenet121(pretrained)
        if self.hparams["m"] == "resnext50_32x4d":
            return models.resnext50_32x4d(pretrained)
        if self.hparams["m"] == "resnext101_32x8d":
            return models.resnext101_32x8d(pretrained)
        print("No model with that name, defaulting to efficientnet_b0")
        return models.efficientnet_b0(pretrained)

    def configure_loss(self):
        loss_name = self.hparams["l"]
        if loss_name == "wce":
            print("Using weighted ce loss")
            return nn.CrossEntropyLoss(weight=self.class_weights)
        if loss_name == "ce":
            print("Using standard ce loss")
            return nn.CrossEntropyLoss()
        print("No loss with that name, defaulting to CE Loss")
        return nn.CrossEntropyLoss()
