import glob
import os
import glob

import torchvision.models as models
import torch.nn as nn

from model import Classifier


class ExperimentBuilder:

    def __init__(self,
                 extractor_type,
                 loss,
                 num_classes=8,
                 learning_rate=1e-3,
                 weight_decay=1e-8,
                 class_weights=None
                 ):
        self.extractor_type = extractor_type
        self.loss_name = loss
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights

        layers = self.get_model().children()
        convolution_layers = layers[:-1]
        self.extractor = nn.Sequential(*convolution_layers)
        self.classifier = nn.Linear(
            layers[-1][1].in_features, self.num_classes)
        self.loss = self.configure_loss()

    def create(self, checkpoint):
        if(checkpoint is None):
            model = Classifier(
                extractor=self.extractor,
                classifier=self.classifier,
                loss=self.loss,
                num_classes=self.num_classes,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
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
                                                extractor=self.extractor,
                                                classifier=self.classifier,
                                                loss=self.loss,
                                                num_classes=self.num_classes,
                                                learning_rate=self.learning_rate,
                                                weight_decay=self.weight_decay,
                                                )
        return model

    def get_model(self, pretrained=True):
        if self.extractor_type == "efficientnet_b1":
            return models.efficientnet_b1(pretrained)
        if self.extractor_type == "efficientnet_b2":
            return models.efficientnet_b2(pretrained)
        if self.extractor_type == "efficientnet_b3":
            return models.efficientnet_b3(pretrained)
        if self.extractor_type == "efficientnet_b4":
            return models.efficientnet_b4(pretrained)
        if self.extractor_type == "efficientnet_b5":
            return models.efficientnet_b5(pretrained)
        if self.extractor_type == "efficientnet_b6":
            return models.efficientnet_b6(pretrained)
        if self.extractor_type == "efficientnet_b7":
            return models.efficientnet_b7(pretrained)
        if self.extractor_type == "resnet50":
            return models.resnet50(pretrained)
        if self.extractor_type == "resnet101":
            return models.resnet101(pretrained)
        if self.extractor_type == "densenet121":
            return models.densenet121(pretrained)
        if self.extractor_type == "resnext50_32x4d":
            return models.resnext50_32x4d(pretrained)
        if self.extractor_type == "resnext101_32x8d":
            return models.resnext101_32x8d(pretrained)
        # Default:
        return models.efficientnet_b0(pretrained)

    def configure_loss(self):
        if not self.class_weights is None:
            return nn.CrossEntropyLoss(weight=self.class_weights)
        # Default to CE loss
        return nn.CrossEntropyLoss()
