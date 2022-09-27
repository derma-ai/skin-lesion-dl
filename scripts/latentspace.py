import os
import sys

from torch.utils.data import DataLoader

import platform
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from torch import Tensor

script_dir = os.path.dirname('__file__')
sys.path.append(os.path.join(script_dir, '..', 'src'))

from color_constancy import compute_color_constancy
from model import Classifier
from experiment_builder import ExperimentBuilder

"""
    Extract latent space representation of data points

    model: A pretrained model
    data_loader: A dataloader that provides the data to be visualized
"""
def compute_latent_space_representation(model, data_loader:DataLoader):
    labels = np.zeros((len(data_loader.dataset),1))
    for idx, (X, y) in enumerate(data_loader):
        if(idx==0):
            latent_points = np.zeros((len(data_loader.dataset),model.extractor(X).size(dim=1)))
        if(idx % 50 == 0):
            print(f"Currently at index {idx} of {len(data_loader)}")
        latent_variables = model.extractor(X).squeeze()
        labels[idx*y.size(dim=0): (idx+1)*y.size(dim=0)] = y.detach().numpy()[:,None]
        latent_points[idx*y.size(dim=0): (idx+1)*y.size(dim=0)] = latent_variables.detach().numpy()
    return latent_points, labels

    
def visualize_latent_space(latent_points, labels, idx_to_class):
    pca = PCA(n_components=2)
    pca.fit(latent_points)
    dim_red_latent_points = pca.transform(latent_points)
    dim_red_latent_points = np.transpose(dim_red_latent_points)
    colors_dict = {0: '#F5B041',1: '#ABEBC6',2: '#465AFF',3: '#FF4233',4: '#A44AFE',5: '#FF60C3',6: '#A4FF60', 7: '#116F95'}
    scatter_points_x = dim_red_latent_points[0]
    scatter_points_y = dim_red_latent_points[1]
    fig, ax =plt.subplots((3,3))
    for class_index in np.unique(labels):
        index_x = np.squeeze(labels == class_index)
        ax[(int(class_index/3)),(class_index%3)].scatter(scatter_points_x[index_x], scatter_points_y[index_x], c=colors_dict[class_index], label= idx_to_class[class_index])
        ax[2,2].scatter(scatter_points_x[index_x], scatter_points_y[index_x], c=colors_dict[class_index], label= idx_to_class[class_index])
    ax.legend()
    plt.show()

def get_dataloader(dataset:str):
    if(platform.system()=="Linux"):
        root_path = os.path.join("/","space","derma-data","isic_2019")
    else:
        root_path = os.path.expanduser("~/share-all/derma-data/")
    preprocessed_dataset = datasets.ImageFolder(os.path.join(root_path, dataset), transforms.Compose([transforms.ToTensor(), transforms.Lambda(compute_color_constancy), transforms.Resize((224,224))]))
    
    # define trainloader
    return DataLoader(preprocessed_dataset,
                                batch_size=16,
                                num_workers=8,
                                drop_last=False,
                                timeout=30000,
                                pin_memory=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('-ckpt', '--model_checkpoint', type=str, dest='model_checkpoint',
                        default=None, help="Path to checkpoint of model")
    parser.add_argument('-d', '--dataset', type=str, dest='dataset',
                        default='preprocessed', help="Which variant of dataset ISIC2019 to use, e.g. 'preprocessed', 'clean'")
    args = parser.parse_args()

    hparams = {
        "e": 100,
        "b": 32,
        "d": "preprocessed",
        "lr": 10e-3,
        "wd": 10e-8,
        "ws": 1,
        "ex": "Testing latent space visualization",
        "t": None,
        "l": "wce",
        "m": "efficientnet_b0",
        "osr": 1.5,
        "lrf": 0
    }
    experimentbuilder = ExperimentBuilder(hparams, 8, Tensor([1,1,1,1,1,1,1,1]))
    model = Classifier.load_from_checkpoint(args.model_checkpoint, hparams=hparams, classifier= experimentbuilder.classifier,extractor= experimentbuilder.extractor , loss= experimentbuilder.loss)
    dataloader:DataLoader = get_dataloader(args.dataset)
    latent_space_points, labels = compute_latent_space_representation(model=model,data_loader=dataloader)
    class_to_idx = dataloader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    visualize_latent_space(latent_points=latent_space_points, labels=labels, idx_to_class=idx_to_class)


if __name__ == "__main__":
    main()
