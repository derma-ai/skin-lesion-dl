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

def preprocess_images(dataset):
    processed_images = []
    edge = 0.01
    for k in range(len(dataset)):
        image, label = dataset[k]
        # average over rgb-channel to creat grey image
        grey_img = (torch.sum(image, dim=0) / 3).numpy()
        # average over image edges to see if they are (mostly) black
        has_black_edges = ((np.average(grey_img[0][:]) < edge) and (np.average(grey_img[-1][:]) < edge) and (np.average(grey_img[:][0]) < edge) and (np.average(grey_img[:][-1]) < edge))
        if(has_black_edges):
            # create black and white image with certain threshold
            blwh_img = (1.0 * (grey_img >= 0.12))
            cov_m = np.cov(blwh_img)
            box_idx = np.argwhere((cov_m) > 0.001)
            outer_points = np.argmax(box_idx, axis=0)
            height= np.abs(box_idx[outer_points[0],0] - box_idx[outer_points[0],1])
            width = np.abs(box_idx[outer_points[1],0] - box_idx[outer_points[1],1])
            processed_images.append((tv.transforms.functional.resized_crop(image, top=box_idx[0,0], left=box_idx[0,1],height= height, width= width, size=224), label))
        else:
            processed_images.append((image, label))
    return processed_images

def color_constancy():
    return

def create_dataset_dir(path, dataset):
    if not os.path.isdir(path):
        os.mkdir(path)
    for img_class in classes:
        class_directory = os.path.join(path, img_class)
        classes_dict[img_class] = class_directory
    if not os.path.isdir(class_directory):
        os.mkdir(class_directory)
    for image, label in dataset:
        img = transforms.ToPILImage()
        img_class = classes[label]
        img.save()
    return

def main():
    # load dataset untransformed 
    root = os.path.join("/", "space")
    whole_dataset = datasets.ImageFolder(os.path.join(root,"derma-data"), transforms.ToTensor())

    eliminate_blackedged_images = preprocess_images(whole_dataset)
    create_dataset_dir(os.path.join(root, "derma-data-preprocessed"), eliminate_blackedged_images)

if __name__ == "__main__":
    main()