import torch
import torch.nn as nn
import torchmetrics
import torchvision
import pytorch_lightning as pl
import seaborn as sns
from torchsummary import summary


class ResNetClassifier(pl.LightningModule):
    """
    Classifier Model written in pytorch_lightning

    ...

    Attributes
    ----------
    numm_classes : int
        number of classes (size of the output layer)
    zero_prob : float
        dropout probability that a neuron is set to 0
    class_weights : numpy array
        arrays of weights to be used as class weights for the loss
    learning_rate : float
        learning rate for the optimizer
    weight_decay : float
        weight regularization parameter

    Methods
    -------

    """

    def __init__(self,
                 num_classes=8,
                 learning_rate=1e-3,
                 weight_decay=1e-8,
                 class_weights=None):
        super().__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.zero_prob = 0.5

        self.resnet_conv_layers = list(torchvision.models.resnet50(
            pretrained=True, progress=True).children())[:-1]
        self.extractor = nn.Sequential(*self.resnet_conv_layers)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 8)

        self.extractor.requires_grad_(False)

        self.train_acc = torchmetrics.Accuracy(
            num_classes=num_classes, average='macro')
        self.val_acc = torchmetrics.Accuracy(
            num_classes=num_classes, average='macro')
        self.conf_matrix = torchmetrics.ConfusionMatrix(self.num_classes)

        if class_weights is not None:
            self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.extractor(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.extractor.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.train_acc(logits, y)

        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 logger=True)
        self.log('train_acc',
                 self.train_acc,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)

        loss = self.loss(logits, y)
        self.val_acc(logits, y)
        self.log('val_acc',
                 self.val_acc,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        preds = nn.Softmax(dim=1)(logits)
        return y, preds

    def validation_epoch_end(self, validation_step_outputs):
        pred_step_tensors = []
        target__step_tensors = []

        for tuple in validation_step_outputs:
            target__step_tensors.append(tuple[0])
            pred_step_tensors.append(tuple[1])
        concat_targets = torch.cat(target__step_tensors)
        stacked_preds = torch.vstack(pred_step_tensors)
        
        confusion_matrix = self.conf_matrix(
            preds=stacked_preds, target=concat_targets)
        confusion_matrix_np = confusion_matrix.cpu().data.numpy()
        heat_map = sns.heatmap(confusion_matrix_np, annot=True)
        self.logger.experiment.add_figure("conf matrix", heat_map.get_figure())
