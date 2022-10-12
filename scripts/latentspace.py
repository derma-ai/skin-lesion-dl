import os
import sys

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import platform
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
def compute_latent_space_representation(model, data_loader:DataLoader, ex_name:str):

    #Put model in eval mode
    model.eval()
    model.to('cuda')

    # List for pointers to the result arrays
    latent_points = []
    labels = []

    for idx, (X, y) in enumerate(data_loader):
        X = X.to('cuda')
        y = y.to('cuda')

        print(f"Currently at index {idx} of {len(data_loader)}")

        latent_variables = model.extractor(X).squeeze()
        labels.append(y.cpu().detach()[:,None])
        latent_points.append(latent_variables.cpu().detach())

    # Stack output
    latent_points = torch.vstack(latent_points).numpy()
    labels = torch.vstack(labels).numpy()

    # Save output
    if(platform.system()=="Linux"):
        root_path = os.path.join("/","space","derma-data")
    else:
        root_path = os.path.expanduser("~/share-all/derma-data/")

    if not os.path.isdir(f"{root_path}{ex_name}"):
        os.mkdir(f"{root_path}{ex_name}")
    np.save(f"{root_path}{ex_name}/labels.npy",labels)
    np.save(f"{root_path}{ex_name}/latent_vectors.npy",latent_points)

    return latent_points, labels


def visualize_latent_space(idx_to_class, ex_name, method='pca', do_scaling=False):
    if(platform.system()=="Linux"):
        root_path = os.path.join("/","space","derma-data")
    else:
        root_path = os.path.expanduser("~/share-all/derma-data/")
    # Load generated latent variables
    latent_points = np.load(f"{root_path}{ex_name}/latent_vectors.npy")
    labels = np.load(f"{root_path}{ex_name}/labels.npy")

    # Dirty hack to speed up computation. Choose only every 10th element. Can be deleted to allow for the whole dataset to be considered
    latent_points = latent_points[::10]
    labels = labels[::10]

    if do_scaling:
        latent_points = StandardScaler(with_mean=True,with_std=True).fit_transform(latent_points)

    if method == 'pca':
        pca = PCA(n_components=3)
    else:
        pca = TSNE(n_components=3)

    # Compute SVD to check singular values
    singular_values = torch.linalg.svdvals(torch.from_numpy(latent_points[:100,:]))
    print("Singular values:",singular_values)

    dim_red_latent_points = pca.fit_transform(latent_points)



    dim_red_latent_points = np.transpose(dim_red_latent_points)
    colors_dict = {0: '#F5B041',1: '#ABEBC6',2: '#465AFF',3: '#FF4233',4: '#A44AFE',5: '#FF60C3',6: '#A4FF60', 7: '#116F95'}
    scatter_points_x = dim_red_latent_points[0]
    scatter_points_y = dim_red_latent_points[1]
    scatter_points_z = dim_red_latent_points[2]

    
    lower_lim = -25
    upper_lim = 25


    # Generate 2D plots of latent space points
    fig, ax = plt.subplots(3,3)
    for class_index in np.unique(labels):
        index_x = np.squeeze(labels == class_index)
        centroid_x = np.mean(scatter_points_x[index_x])/len(scatter_points_x[index_x])
        centroid_y = np.mean(scatter_points_y[index_x])/len(scatter_points_y[index_x])
        ax[(int(class_index/3)),int((class_index%3))].scatter(centroid_x,centroid_y, c='#db5b00')
        ax[(int(class_index/3)),int((class_index%3))].scatter(scatter_points_x[index_x], scatter_points_y[index_x], c=colors_dict[class_index], label= idx_to_class[class_index])
        ax[2,2].scatter(scatter_points_x[index_x], scatter_points_y[index_x], c=colors_dict[class_index], label= idx_to_class[class_index])
        ax[(int(class_index/3)),int((class_index%3))].set_xlim([lower_lim,upper_lim])
        ax[(int(class_index/3)),int((class_index%3))].set_ylim([lower_lim,upper_lim])
        ax[(int(class_index/3)),int((class_index%3))].legend()
    ax[2,2].set_xlim([lower_lim, upper_lim])
    ax[2,2].set_ylim([lower_lim, upper_lim])
    ax[2,2].legend()
    fig.set_size_inches(20,10)
    plt.savefig(f"{method}_2-components{'_scaled' if do_scaling else ''}.png", bbox_inches='tight', dpi=300)

    # Generate 2d projections
    fig, ax =plt.subplots(1,3)
    for class_index in np.unique(labels):
        index_x = np.squeeze(labels == class_index)
        ax[0].scatter(scatter_points_x[index_x], scatter_points_y[index_x], c=colors_dict[class_index], label= idx_to_class[class_index])
    ax[0].legend()
    ax[0].set_xlabel('X', fontsize=10)
    ax[0].set_ylabel('Y', fontsize=10)
    ax[0].set_title('X-Y')
    ax[0].set_xlim([lower_lim, upper_lim])
    ax[0].set_ylim([lower_lim, upper_lim])

    for class_index in np.unique(labels):
        index_x = np.squeeze(labels == class_index)
        ax[1].scatter(scatter_points_x[index_x], scatter_points_z[index_x], c=colors_dict[class_index], label= idx_to_class[class_index])
    ax[1].legend()
    ax[1].set_xlabel('X', fontsize=10)
    ax[1].set_ylabel('Z', fontsize=10)
    ax[1].set_title('X-Z')
    ax[1].set_xlim([lower_lim, upper_lim])
    ax[1].set_ylim([lower_lim, upper_lim])

    for class_index in np.unique(labels):
        index_x = np.squeeze(labels == class_index)
        ax[2].scatter(scatter_points_y[index_x], scatter_points_z[index_x], c=colors_dict[class_index], label= idx_to_class[class_index])
    ax[2].legend()
    ax[2].set_xlabel('Y', fontsize=10)
    ax[2].set_ylabel('Z', fontsize=10)
    ax[2].set_title('Y-Z')
    ax[2].set_xlim([lower_lim, upper_lim])
    ax[2].set_ylim([lower_lim, upper_lim])
    fig.set_size_inches(40,10)
    plt.savefig(f"{method}_3-components{'_scaled' if do_scaling else ''}_2d-projections.png", bbox_inches='tight', dpi=300)

    # Generate 3d scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for class_index in np.unique(labels):
        index_x = np.squeeze(labels == class_index)
        ax.scatter(scatter_points_x[index_x], scatter_points_y[index_x], scatter_points_z[index_x], c=colors_dict[class_index], label= idx_to_class[class_index])
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_xlim([lower_lim, upper_lim])
    ax.set_ylim([lower_lim, upper_lim])
    ax.set_zlim([lower_lim, upper_lim])
    ax.legend()
    fig.set_size_inches(10,10)
    plt.savefig(f"{method}_3-components{'_scaled' if do_scaling else ''}_3d.png", bbox_inches='tight', dpi=300)

def get_dataloader(dataset:str):
    if(platform.system()=="Linux"):
        root_path = os.path.join("/","space","derma-data","isic_2019")
    else:
        root_path = os.path.expanduser("~/share-all/derma-data/")
    preprocessed_dataset = datasets.ImageFolder(os.path.join(root_path, dataset), transforms.Compose([transforms.ToTensor(), transforms.Lambda(compute_color_constancy), transforms.Resize((224,224))]))
    
    # define trainloader
    return DataLoader(preprocessed_dataset,
                                batch_size=16,
                                num_workers=12,
                                drop_last=False,
                                timeout=30000,
                                pin_memory=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('-ckpt', '--model_checkpoint', type=str, dest='model_checkpoint',
                        default=None, help="Path to checkpoint of model")
    parser.add_argument('-d', '--dataset', type=str, dest='dataset',
                        default='preprocessed', help="Which variant of dataset ISIC2019 to use, e.g. 'preprocessed', 'clean'")
    parser.add_argument('-ne', '--name_experiment', type=str, dest='name_ex',
                    default="ex_0815", help="Keep track of saved latent points")
    parser.add_argument('-m', '--method', type=str, dest='method',
                        default='pca', help="PCA or TSNE method")
    parser.add_argument('-ds', '--do_scaling', type=bool, dest='do_scaling',
                    default=False, help="Perform scaling before applying PCA or TSNE")
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

    # Comment this out if the latent_variables have already been generated
    latent_space_points, labels = compute_latent_space_representation(model=model,data_loader=dataloader, ex_name=args.name_ex)
    class_to_idx = dataloader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # visualize_latent_space(latent_points=latent_space_points, labels=labels, idx_to_class=idx_to_class)
    visualize_latent_space(idx_to_class=idx_to_class, ex_name=args.name_ex,  method=args.method, do_scaling=args.do_scaling)


if __name__ == "__main__":
    main()
