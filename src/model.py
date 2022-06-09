import torch
import torch.nn as nn
import torchmetrics
import torchvision
import torchvision.models as models
import pytorch_lightning as pl
from collections import Counter
import seaborn as sns
from torchsummary import summary

class Classifier(pl.LightningModule):
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
                 hparams,
                 extractor,
                 classifier,
                 loss,
                 num_classes=8
                 ):
        super().__init__()
        self.hparams.update(hparams)
        self.num_classes = num_classes
        self.zero_prob = 0.5

        self.extractor = extractor
        self.classifier = classifier
        self.loss = loss

        self.configure_metrics()
        print(self.hparams)

    def forward(self, x):
        x = self.extractor(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.extractor.parameters(),
                                     lr=self.hparams["lr"],
                                     weight_decay=self.hparams["wd"])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.train_acc(logits, y)
        train_acc_per_class = self.train_acc_per_class(logits,y)
        train_prec_per_class = self.train_prec_per_class(logits,y)
        train_rec_per_class = self.train_rec_per_class(logits,y)

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

        self.log_per_class(mode="train", metric="acc",
                           values=train_acc_per_class)
        self.log_per_class(mode="train", metric="prec",
                           values=train_prec_per_class)
        self.log_per_class(mode="train", metric="rec",
                           values=train_rec_per_class)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
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
        self.add_histogram()
        pred_step_tensors = []
        target__step_tensors = []
        for tuple in validation_step_outputs:
            target__step_tensors.append(tuple[0])
            pred_step_tensors.append(tuple[1])

        concat_targets = torch.cat(target__step_tensors)
        stacked_preds = torch.vstack(pred_step_tensors)

        val_acc_per_class = self.val_acc_per_class(stacked_preds,concat_targets)
        val_prec_per_class = self.val_prec_per_class( stacked_preds, concat_targets)
        val_rec_per_class = self.val_rec_per_class(stacked_preds,concat_targets)
        self.log_per_class(mode="val", metric="acc", values=val_acc_per_class)
        self.log_per_class(mode="val", metric="prec", values=val_prec_per_class)
        self.log_per_class(mode="val", metric="rec", values=val_rec_per_class)
        confusion_matrix = self.conf_matrix(
            preds=stacked_preds, target=concat_targets)
        confusion_matrix_np = confusion_matrix.cpu().data.numpy()
        heat_map = sns.heatmap(confusion_matrix_np, annot=True)
        self.logger.experiment.add_figure("conf matrix", heat_map.get_figure(), global_step=self.current_epoch)

    def on_train_epoch_start(self):
        if self.current_epoch == 8:
            self.extractor.requires_grad_(True)

    def log_per_class(self, mode, metric, values):
        for i in range(len(values)):
            self.log(f'{mode}_{metric}_class_{i}',
                     values[i],
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)

    def add_histogram(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def configure_metrics(self):
        # training metrics
        self.train_acc = torchmetrics.Accuracy(
            num_classes=self.num_classes, average='macro')
        self.train_acc_per_class = torchmetrics.Accuracy(
            num_classes=self.num_classes, average='none')
        self.train_prec_per_class = torchmetrics.Precision(
            num_classes=self.num_classes, average='none')
        self.train_rec_per_class = torchmetrics.Recall(
            num_classes=self.num_classes, average='none')

        # validation metrics
        self.val_acc = torchmetrics.Accuracy(
            num_classes=self.num_classes, average='macro')
        self.val_acc_per_class = torchmetrics.Accuracy(
            num_classes=self.num_classes, average='none')
        self.val_prec_per_class = torchmetrics.Precision(
            num_classes=self.num_classes, average='none')
        self.val_rec_per_class = torchmetrics.Recall(
            num_classes=self.num_classes, average='none')
        self.conf_matrix = torchmetrics.ConfusionMatrix(self.num_classes)
