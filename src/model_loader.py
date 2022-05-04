
def load(hparams, checkpoint):
    if(checkpoint is None):
         print("Checkpoint is None")
         model = ResNetClassifier(learning_rate=hparams["lr"],
                                  weight_decay=hparams["wd"], 
                                  num_classes=hparams["c"])
    elif(len(checkpoint) == 0):
         latest_ckpt = max(glob.glob('./checkpoints/*.ckpt'), key=os.path.getctime)
         model = ResNetClassifier.load_from_checkpoint(latest_ckpt,
                                                       learning_rate=hparams["lr"],
                                                       weight_decay=hparams["wd"], 
                                                       num_classes=hparams["c"])
    else:
         ckpt = max(glob.glob(f"./checkpoints/*{checkpoint}.ckpt"), key=os.path.getctime)
         print(ckpt)
         model = ResNetClassifier.load_from_checkpoint(ckpt,
                                                       learning_rate=hparams["lr"],
                                                       weight_decay=hparams["wd"], 
                                                       num_classes=hparams["c"])