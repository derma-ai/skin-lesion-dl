from itertools import count
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
                 model_name,
                 num_classes=8,
                 learning_rate=1e-3,
                 weight_decay=1e-8,
                 class_weights=None):
        super().__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.zero_prob = 0.5

        layers = list(get_model(model_name).children())
        self.resnet_conv_layers = layers[:-1]
        self.extractor = nn.Sequential(*self.resnet_conv_layers)
        self.classifier = nn.Linear(layers[-1][1].in_features, self.num_classes)

        # training metrics
        self.train_acc = torchmetrics.Accuracy(
            num_classes=num_classes, average='macro')
        self.train_acc_per_class = torchmetrics.Accuracy(
            num_classes=num_classes, average='none')
        self.train_prec_per_class = torchmetrics.Precision(
            num_classes=num_classes, average='none')
        self.train_rec_per_class= torchmetrics.Recall(
            num_classes=num_classes, average='none')
        
        # validation metrics
        self.val_acc = torchmetrics.Accuracy(
            num_classes=num_classes, average='macro')
        self.val_acc_per_class= torchmetrics.Accuracy(
            num_classes=num_classes, average='none')
        self.val_prec_per_class = torchmetrics.Precision(
            num_classes=num_classes, average='none')
        self.val_rec_per_class= torchmetrics.Recall(
            num_classes=num_classes, average='none')
        self.conf_matrix = torchmetrics.ConfusionMatrix(self.num_classes)

        self.example_input_array = torch.rand(10,3,224,224)

        if class_weights is not None:
            self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.extractor(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.extractor.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        class_counts = Counter(list(y.numpy()))
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

        self.log_per_class(mode="train", metric="acc", values=train_acc_per_class)
        self.log_per_class(mode="train", metric="prec", values=train_prec_per_class)
        self.log_per_class(mode="train", metric="rec", values=train_rec_per_class)

        return {"loss":loss, "class_counts": class_counts}
    
    def training_epoch_end(self, outputs) -> None:
        total_counts = Counter()
        for dict in outputs:
            total_counts += dict["class_counts"]
        print(total_counts.most_common())
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)

        #loss = self.loss(logits, y)
        self.val_acc(logits, y)
        val_acc_per_class = self.val_acc_per_class(logits, y)
        val_prec_per_class = self.val_prec_per_class(logits, y)
        val_rec_per_class = self.val_rec_per_class(logits,y)

        self.log('val_acc',
                 self.val_acc,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.log_per_class(mode="val", metric="acc", values=val_acc_per_class)
        self.log_per_class(mode="val", metric="prec", values=val_prec_per_class)
        self.log_per_class(mode="val", metric="rec", values=val_rec_per_class)

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
            self.logger.experiment.add_histogram(name,params, self.current_epoch)

def get_model(model_name, pretrained=True):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained)
        return model
    if model_name == "efficientnet_b1":
        return models.efficientnet_b1(pretrained)
    if model_name == "efficientnet_b2":
        return models.efficientnet_b2(pretrained)
    if model_name == "efficientnet_b3":
        return models.efficientnet_b3(pretrained)
    if model_name == "efficientnet_b4":
        return models.efficientnet_b4(pretrained)
    if model_name == "efficientnet_b5":
        return models.efficientnet_b5(pretrained)
    if model_name == "efficientnet_b6":
        return models.efficientnet_b6(pretrained)
    if model_name == "efficientnet_b7":
        return models.efficientnet_b7(pretrained)
    if model_name == "resnet50":
        return models.resnet50(pretrained)
    if model_name == "resnet101":
        return models.resnet101(pretrained)
    if model_name == "densenet121":
        return models.densenet121(pretrained)
    if model_name == "resnext50_32x4d":
        return models.resnext50_32x4d(pretrained)
    if model_name == "resnext101_32x8d":
        return models.resnext101_32x8d(pretrained)



