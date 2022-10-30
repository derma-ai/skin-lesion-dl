from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, confusion_matrix, classification_report
import data_handler
from experiment_builder import ExperimentBuilder
import torch
import pytorch_lightning as pl
import pandas as pd
from argparse import ArgumentParser
import os
import numpy as np


def set_seed(seed=15):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def test(hparams, checkpoint_path, gpu):
    set_seed()
    _,dataset, _ = data_handler.setup_data(hparams, None)
    builder = ExperimentBuilder(hparams)
    model = builder.create()

    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
    # This destroys loading for some reason?
    checkpoint["state_dict"].pop("loss.weight")
    model.load_state_dict(checkpoint['state_dict'])

    trainer = pl.Trainer(gpus=[gpu],
                         logger=False
                         )

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=16,
                                              num_workers=8,
                                              drop_last=False,
                                              timeout=30000,
                                              pin_memory=True)

    predictions = trainer.predict(model, data_loader)

    features, labels = zip(*predictions)

    features = torch.vstack(features)

    # features is a batch_size X output_size tensor
    labels = torch.cat(labels)

    print("\n\n############\nCNN Classification\n############")

    clf_report = pd.DataFrame(classification_report(labels, torch.argmax(
        features, dim=1), output_dict=True, zero_division=0.0))
    print("Classification report:\n{}".format(clf_report))

    filepath = os.path.expanduser(
        f'~/IDP/derma_ai_idp/model_reports')
    os.makedirs(filepath, exist_ok=True)
    clf_report.to_csv(os.path.join(
        filepath, f"{hparams['m']=}-{hparams['res']=}-{hparams['l']=}-{hparams['t']=}.csv"))


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str,
                        default="", dest="model", help="Model name")
    parser.add_argument("-t", "--transforms", type=str, default=None, dest="transforms",
                        help="Comma separated list of transform flags, e.g. /'r,hflip,vflip/'")
    parser.add_argument("-l", "--loss", type=str, default="", dest="loss",
                        help="Loss function'")
    parser.add_argument("-wi", "--width", type=int, default=224, dest="width",
                        help="Resolution width of images'")
    parser.add_argument("-he", "--height", type=int, default=224, dest="height",
                        help="Resolution height of images'")
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None,
                        dest="checkpoint", help="Path to checkpoint")
    parser.add_argument('-gpu', '--gpu', type=int, default=None,
                        dest="gpu", help="On which GPU to train")
    parser.add_argument('-p', '--path', type=str, default=None,
                        dest='path', help="Path to dataset root folder")
    parser.add_argument('-d', '--dataset', type=str, default="preprocessed",
                        dest='dataset', help="Version of ISIC2019 of dataset to use")             
    args = parser.parse_args()

    hparams = {
        "e": 10,
        "b": 16,
        "lr": 1e-3,
        "wd": 1e-8,
        "m": args.model,
        "t": args.transforms,
        "l": args.loss,
        "d": args.dataset,
        "res": (args.width, args.height)
    }
    test(hparams, args.checkpoint, args.gpu)


if __name__ == "__main__":
    main()
