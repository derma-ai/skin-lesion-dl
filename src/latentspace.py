import os
from statistics import mean
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
from sklearn.model_selection import train_test_split
import seaborn as sns
from torch import Tensor

script_dir = os.path.dirname('__file__')
sys.path.append(os.path.join(script_dir, '..', 'src'))

from color_constancy import compute_color_constancy
from model import Classifier
from experiment_builder import ExperimentBuilder
from subset import Subset





def set_seed(seed=15):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

"""
    Extract latent space representation of data points

    model: A pretrained model
    data_loader: A dataloader that provides the data to be visualized
"""
def compute_latent_space_representation(model, data_loader:DataLoader, ex_name:str, val:bool):

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
    np.save(f"{root_path}{ex_name}/{'validation_set_' if val else 'training_set_'}labels.npy",labels)
    np.save(f"{root_path}{ex_name}/{'validation_set_' if val else 'training_set_'}latent_vectors.npy",latent_points)
    return latent_points, labels


def visualize_latent_space_classes(idx_to_class, ex_name, method='pca', do_scaling=False):
    if(platform.system()=="Linux"):
        root_path = os.path.join("/","space","derma-data")
    else:
        root_path = os.path.expanduser("~/share-all/derma-data/")

    # Load generated latent variables
    train_latent_points = np.load(f"{root_path}{ex_name}/training_set_latent_vectors.npy")
    train_labels = np.load(f"{root_path}{ex_name}/training_set_labels.npy")

    val_latent_points = np.load(f"{root_path}{ex_name}/validation_set_latent_vectors.npy")
    val_labels = np.load(f"{root_path}{ex_name}/validation_set_labels.npy")
    
    train_latent_points = train_latent_points[::10]
    train_labels = train_labels[::10]

    val_latent_points = val_latent_points[::10]
    val_labels = val_labels[::10]

    if do_scaling:
        train_latent_points = StandardScaler(with_mean=True,with_std=True).fit_transform(train_latent_points)
        val_latent_points = StandardScaler(with_mean=True,with_std=True).fit_transform(val_latent_points)
    if method == 'pca':
        pca = PCA(n_components=3)
    else:
        pca = TSNE(n_components=3)

    train_dim_red_latent_points = pca.fit_transform(train_latent_points)
    val_dim_red_latent_points = pca.fit_transform(val_latent_points)

    train_dim_red_latent_points = np.transpose(train_dim_red_latent_points)
    
    val_dim_red_latent_points = np.transpose(val_dim_red_latent_points)
    
    train_colors_dict = {0: '#ad6c02',1: '#169e4f',2: '#0315a8',3: '#ab0909',4: '#5603a8',5: '#c41a84',6: '#63b327', 7: '#045a7d'}
    val_colors_dict ={0:'#f5a727', 1: '#8eedb6', 2:'#6d7bf2',3: '#eb7a7a',4: '#b371f5', 5:'#f277c4',6:'#acf277',7:'#68c3e8'}
    
    
    train_scatter_points_x = train_dim_red_latent_points[0]
    train_scatter_points_y = train_dim_red_latent_points[1]

    val_scatter_points_x = val_dim_red_latent_points[0]
    val_scatter_points_y = val_dim_red_latent_points[1]


    # Generate 2D plots of latent space points
    fig, ax = plt.subplots(3,3)
    columns_headers = []
    for class_index in np.unique(train_labels):
        train_index_x = np.squeeze(train_labels == class_index)
        
        val_index_x = np.squeeze(val_labels == class_index)

               
        columns_headers.append(idx_to_class[class_index])
        # set up individual legend
        leftmarker, =  ax[(int(class_index/3)),int((class_index%3))].plot([],[], c=train_colors_dict[class_index], marker='s', fillstyle="left",linestyle='none', markersize=15)
        rightmarker, =  ax[(int(class_index/3)),int((class_index%3))].plot([],[], c=val_colors_dict[class_index],marker='s', fillstyle="right",linestyle='none', markersize=15)
        
        ax[(int(class_index/3)),int((class_index%3))].legend(((leftmarker,rightmarker),), (idx_to_class[class_index],""), numpoints=1,loc="upper right")

        # scatter training set points
        ax[(int(class_index/3)),int((class_index%3))].scatter(train_scatter_points_x[train_index_x], train_scatter_points_y[train_index_x], c=train_colors_dict[class_index], label= idx_to_class[class_index])
        #ax[2,2].scatter(train_scatter_points_x[index_x], train_scatter_points_y[index_x], c=train_colors_dict[class_index], label= idx_to_class[class_index]) 
        
        # scatter validation set points
        ax[(int(class_index/3)),int((class_index%3))].scatter(val_scatter_points_x[val_index_x],val_scatter_points_y[val_index_x] , c=val_colors_dict[class_index], label= idx_to_class[class_index])
        ax[2,2].scatter(val_scatter_points_x[val_index_x], val_scatter_points_y[val_index_x], c=val_colors_dict[class_index], label= idx_to_class[class_index])
        
        

    fig.set_size_inches(20,12)
    fig.suptitle("Latent space representations of training and validation set points after training. Samples of the training set are marked by the darker color, samples of  the validation set by the lighter.", verticalalignment='baseline', x=0.5, y= 0.05)
    plt.savefig(f"{ex_name}_{method}_2-components{'_scaled' if do_scaling else ''}_validation_vs_training.png", bbox_inches='tight', dpi=300)




def visualize_latent_space(idx_to_class, ex_name, val, method='pca', do_scaling=False):
    if(platform.system()=="Linux"):
        root_path = os.path.join("/","space","derma-data")
    else:
        root_path = os.path.expanduser("~/share-all/derma-data/")

    # Load generated latent variables
    latent_points = np.load(f"{root_path}{ex_name}/{'validation_set_' if val else 'training_set_'}latent_vectors.npy")
    labels = np.load(f"{root_path}{ex_name}/{'validation_set_' if val else 'training_set_'}labels.npy")

    # Dirty hack to speed up computation. Choose only every 10th element. Can be deleted to allow for the whole dataset to be considered
    latent_points = latent_points[::10]
    labels = labels[::10]
    centroids = np.zeros((8,latent_points.shape[1]))
    mean_distances = np.ndarray((8,8))
    # Compute centroid for each class
    for class_index in np.unique(labels):
        indices = (labels == class_index)
        centroids[class_index]= np.mean(latent_points[indices[:,0],:], axis=0)
    # Compute mean distances of class specific latent points to each centroid
    for class_index in np.unique(labels):
        indices = (labels == class_index)
        distances = np.linalg.norm(latent_points[np.newaxis,indices[:,0],:] - centroids[:,np.newaxis,:], axis=2)
        mean_distances[class_index] = np.mean(distances, axis=1)
    mean_distances = mean_distances.round(4)
    # Heatmap of mean distances
    fig = plt.figure(figsize=(8,5))
    sns.heatmap(mean_distances,annot= mean_distances.astype(int),cmap="vlag",fmt="g", center=60, xticklabels=["AK-C","BCC-C","BKL-C","DF-C","MEL-C","NV-C","SCC-C","VASC-C"],yticklabels=["AK","BCC","BKL","DF","MEL","NV","SCC","VASC"])
    plt.savefig(f"{ex_name}_latent_space_mean_distances_to_centroids{'_validation_set' if val else ''}.png", bbox_inches='tight', dpi=300)

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
    colors_dict = {0: '#F5B041',1: '#ABEBC6',2: '#465AFF',3: '#f7dd2f',4: '#A44AFE',5: '#FF60C3',6: '#A4FF60', 7: '#116F95'}
    scatter_points_x = dim_red_latent_points[0]
    scatter_points_y = dim_red_latent_points[1]
    scatter_points_z = dim_red_latent_points[2]

    # set global ax limits
    lower_lim = -50
    upper_lim = 50


    # Generate 2D plots of latent space points
    fig, ax = plt.subplots(3,3)
    columns_headers = []
    for class_index in np.unique(labels):
        index_x = np.squeeze(labels == class_index)
        centroid_x = np.mean(scatter_points_x[index_x])
        centroid_y = np.mean(scatter_points_y[index_x])
        columns_headers.append(idx_to_class[class_index])
        
        ax[(int(class_index/3)),int((class_index%3))].scatter(scatter_points_x[index_x], scatter_points_y[index_x], c=colors_dict[class_index], label= idx_to_class[class_index])
        ax[2,2].scatter(scatter_points_x[index_x], scatter_points_y[index_x], c=colors_dict[class_index], label= idx_to_class[class_index])
        ax[(int(class_index/3)),int((class_index%3))].set_xlim([lower_lim,upper_lim])
        ax[(int(class_index/3)),int((class_index%3))].set_ylim([lower_lim,upper_lim])
        ax[(int(class_index/3)),int((class_index%3))].scatter(centroid_x,centroid_y, c='#db5b00')
        ax[(int(class_index/3)),int((class_index%3))].legend()
    
    ax[2,2].set_xlim([lower_lim, upper_lim])
    ax[2,2].set_ylim([lower_lim, upper_lim])
    ax[2,2].legend()
    fig.set_size_inches(20,10)
    plt.savefig(f"{ex_name}_{method}_2-components{'_scaled' if do_scaling else ''}{'_validation_set' if val else ''}.png", bbox_inches='tight', dpi=300)

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
    plt.savefig(f"{ex_name}_{method}_3-components{'_scaled' if do_scaling else ''}_2d-projections{'_validation_set' if val else ''}.png", bbox_inches='tight', dpi=300)

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
    plt.savefig(f"{ex_name}_{method}_3-components{'_scaled' if do_scaling else ''}_3d{'_validation_set' if val else ''}.png", bbox_inches='tight', dpi=300)

def get_dataloader(dataset:str, colorconst):
    if(platform.system()=="Linux"):
        root_path = os.path.join("/","space","derma-data","isic_2019")
    else:
        root_path = os.path.expanduser("~/share-all/derma-data/")
    
    if dataset == 'original':
        root = os.path.join(root_path, "clean")
        print("Using the original dataset")
    else:    
        root = os.path.join(root_path,"preprocessed")
    print("CC",colorconst)
    # apply correct transforms
    if colorconst:
        preprocessed_dataset = datasets.ImageFolder(root, transforms.Compose([transforms.ToTensor(), transforms.Lambda(compute_color_constancy), transforms.Resize((224,224))]))
    else:
        preprocessed_dataset = datasets.ImageFolder(root, transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))]))
    
    #split train and test data
    train_data_idx, val_data_idx = train_test_split(
        list(range(len(preprocessed_dataset))), test_size=0.2, stratify=preprocessed_dataset.targets)
    
    train_data = Subset(preprocessed_dataset, train_data_idx)
    val_data = Subset(preprocessed_dataset, val_data_idx)
     
    return (DataLoader(train_data,
                                batch_size=16,
                                num_workers=12,
                                drop_last=False,
                                timeout=30000,
                                pin_memory=True),
    DataLoader(val_data,
                                batch_size=16,
                                num_workers=12,
                                drop_last=False,
                                timeout=30000,
                                pin_memory=True))

def main():
    parser = ArgumentParser()
    parser.add_argument('-ckpt', '--model_checkpoint', type=str, dest='model_checkpoint',
                    default=None, help="Path to checkpoint of model")
    parser.add_argument('-d', '--dataset', type=str, dest='dataset',
                    default='preprocessed', help="Which variant of dataset ISIC2019 to use, e.g. 'preprocessed', 'clean'")
    parser.add_argument('-ne', '--name_experiment', type=str, dest='name_ex',
                    default="ex_0815", help="Keep track of saved latent points")
    parser.add_argument('-m', '--method', type=str, dest='method',
                    default='tsne', help="PCA or TSNE method")
    parser.add_argument('-ds', '--do_scaling', type=int, dest='do_scaling',
                    default=0, help="Perform scaling before applying PCA or TSNE, either 0 or 1")
    parser.add_argument('-cc', '--color_constancy', type=int, dest='colorconst',
                    default=1, help="Apply color constancy base transform, either 1 for cc or else 0")
    parser.add_argument('-val', '--validation_set',type=int, dest='val',
                    default=0, help="Choose between training and validation set, either 0 or 1")
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
    set_seed()
    experimentbuilder = ExperimentBuilder(hparams, 8, Tensor([1,1,1,1,1,1,1,1]))
    model = Classifier.load_from_checkpoint(args.model_checkpoint, hparams=hparams, classifier= experimentbuilder.classifier,extractor= experimentbuilder.extractor , loss= experimentbuilder.loss)
    train_dataloader, val_dataloader = get_dataloader(args.dataset, bool(args.colorconst))

    # Comment this out if the latent_variables have already been generated
    #compute_latent_space_representation(model=model,data_loader=train_dataloader, ex_name=args.name_ex, val=False)
    #compute_latent_space_representation(model,val_dataloader,ex_name=args.name_ex, val=True)
    class_to_idx = train_dataloader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # visualize_latent_space(latent_points=latent_space_points, labels=labels, idx_to_class=idx_to_class)
    visualize_latent_space_classes(idx_to_class=idx_to_class, ex_name=args.name_ex, method=args.method, do_scaling=bool(args.do_scaling))
    visualize_latent_space(idx_to_class=idx_to_class, ex_name=args.name_ex,  method=args.method, do_scaling=bool(args.do_scaling), val=bool(args.val))


if __name__ == "__main__":
    main()
