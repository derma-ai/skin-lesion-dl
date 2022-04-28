import torch
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
                 num_classes=7,
                 class_weights=None,
                 learning_rate=5e-5):
        super().__init__()

        self.num_classes = num_classes
        self.zero_prob = 0.5
        self.learning_rate = learning_rate

        self.backbone = torchvision.models.resnet50(pretrained=True, progress=True)
        self.backbone.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True)

        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, average='macro')

        if class_weights is not None:
            self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), 
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