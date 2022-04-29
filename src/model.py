import torch
import torch.nn as nn
import torchmetrics
import torchvision
import pytorch_lightning as pl


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
                 class_weights=None):
        super().__init__()

        self.num_classes = num_classes
        self.zero_prob = 0.5
        
        self.extractor = torchvision.models.resnet50(pretrained=True, progress=True)
        # For now just freeze the entire model and use the pretrained conv and linear layers
        self.extractor.requires_grad_(False)
        self.fc1 = nn.Linear(1000, 600)
        self.dropout1 = nn.Dropout(self.zero_prob)
        self.fc2 = nn.Linear(600, 200)
        self.dropout2 = nn.Dropout(self.zero_prob)
        self.fc3 = nn.Linear(200, 8)

        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro')

        if class_weights is not None:
            self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.extractor(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.extractor.parameters(), 
                                    lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        logits = self.forward(x)
        loss = self.loss(logits, y)

        self.log('train_loss',
                loss,
                on_step=True,
                on_epoch=False,
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
        return loss