from cgitb import grey
import os
import torch
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image


"""
Preprocess images to eliminate black edged images


"""
classes = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
classes_dict = {}
file_names_per_class = {}

def preprocess_images(batch):
    print(type(batch), len(batch))
    edge = 0.01
    changed_images = []
    for k in range(len(batch[0])):
        image = batch[k][0]
        label = batch[k][1]
        grey_img = (torch.sum(image, dim=0) / 3).numpy()
        print("Shape of grey image:", grey_img.shape)
        has_black_edges = ((np.average(grey_img[0,:]) < edge) and (np.average(grey_img[-1,:]) < edge) and (np.average(grey_img[:,0]) < edge) and (np.average(grey_img[:,-1]) < edge))
        if(has_black_edges):
            blwh_img = (1.0 * (grey_img >= 0.12))
            cov_m = np.cov(blwh_img)
            box_idx = np.argwhere((cov_m) > 0.001)
            outer_points = np.argmax(box_idx, axis=0)
            height= np.abs(box_idx[outer_points[0],0] - box_idx[outer_points[0],1])
            width = np.abs(box_idx[outer_points[1],0] - box_idx[outer_points[1],1])
            changed_images.append((tv.transforms.functional.resized_crop(image, top=box_idx[0,0], left=box_idx[0,1],height= height, width= width, size=224), dataset[k][1]))
        else:
            changed_images.append((image, label))
    return changed_images

def color_constancy():
    return

def create_dataset_dir(path):
    orig_path = os.path.join(os.path.join("/", "space"),"derma-data")

    if not os.path.isdir(path):
        os.mkdir(path)

    for img_class in classes:
        class_directory = os.path.join(path, img_class)
        original_class_directory = os.path.join(orig_path, img_class)
        classes_dict[img_class] = class_directory
        file_names_per_class[img_class] = os.listdir(original_class_directory) 
    if not os.path.isdir(class_directory):
        os.mkdir(class_directory)
    return

def save_images_to_directory(batch, idx):

    for count, image, label in enumerate(batch):
        img = transforms.ToPILImage()
        img_class = classes[label]
        file_name = file_names_per_class[img_class][count + idx]
        img.save(os.path.join(classes_dict[img_class],file_name),image)


def main():
    # load dataset untransformed 
    root = os.path.join("/", "space")
    whole_dataset = datasets.ImageFolder(os.path.join(root,"derma-data"), transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(whole_dataset,
                                                batch_size=32,
                                                num_workers=8,
                                                drop_last=False,
                                                timeout=30000,
                                                pin_memory=True)
                                                
    for idx, batch in enumerate(data_loader):
        eliminate_blackedged_images = preprocess_images(batch)
        create_dataset_dir(os.path.join(root, "derma-data-preprocessed"), eliminate_blackedged_images, idx)
        save_images_to_directory(batch, idx)
if __name__ == "__main__":
    main()