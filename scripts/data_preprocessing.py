from cgitb import grey
import os
import torch
import cv2
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.datasets as datasets



classes = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
classes_dict = {}
file_names_per_class = {}



"""
Preprocess image to eliminate possible black edges around the image.

  Keyword arguments:
  sample  -- (torch.Tensor, int) - tuple of a labelled image dataset
"""

def preprocess_image(sample):
    edge = 0.01
    image, label = sample
    grey_img = (torch.sum(image, dim=0) / 3).numpy()
    has_black_edges = ((np.average(grey_img[0,:]) < edge) and (np.average(grey_img[-1,:]) < edge) and (np.average(grey_img[:,0]) < edge) and (np.average(grey_img[:,-1]) < edge))
    if(has_black_edges):
        blwh_img = (1.0 * (grey_img >= 0.4))
        cov_m = np.cov(blwh_img)
        box_idx = np.argwhere((cov_m) > 0.001)
        outer_points = np.argmax(box_idx, axis=0)
        height= np.abs(box_idx[outer_points[0],0] - box_idx[outer_points[0],1])
        width = np.abs(box_idx[outer_points[1],0] - box_idx[outer_points[1],1])
        sample = (tv.transforms.functional.resized_crop(image, top=box_idx[0,0], left=box_idx[0,1],height= height, width= width, size=grey_img.shape), label)
    return sample
""" Adjust illumination of image with the Shades of Gray Color Constancy algorithm, adopted from:

    Nick Shawn, Shades_of_Gray-color_constancy_transformation, 2018, Github Repository
    Link: https://github.com/nickshawn/Shades_of_Gray-color_constancy_transformation

    Keyword arguments:
    image -- pytorch tensor image
    p_power -- power of Minkowski - norm
    gamma -- gamma - value for gamma correction
"""
def color_constancy(image, p_power= 6, gamma = None):
    image = image.numpy()
        
    if gamma is not None:
        image = np.round(image * 255.0)
        image = image.astype('uint8')
        look_up_table = np.zeros((256,1), dtype='uint8')
        for i in range(256):
            look_up_table[i][0] = 255*pow(i/255, 1/gamma)
        image = cv2.LUT(image, look_up_table)
        image = image * (1/255.0)

    img = image.astype('float32')
    img_power = np.power(img, p_power)
    rgb_vec = np.power(np.mean(img_power, (1,2)), 1/p_power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec[:, None, None])
    np.where(image > 1, image, 1.0)
    return torch.from_numpy(img)

"""Create new empty directory for dataset given original dataset directory.

    Keyword arguments:
    target_path -- the path into which the new directory is copied
    orig_path -- the original path of the dataset
    """
def create_dataset_dir(target_path, orig_path):

    if not os.path.isdir(target_path):
        os.mkdir(target_path)

    for img_class in classes:
        class_directory = os.path.join(target_path, img_class)
        original_class_directory = os.path.join(orig_path, img_class)
        classes_dict[img_class] = class_directory
        file_names_per_class[img_class] = os.listdir(original_class_directory) 
        if not os.path.isdir(class_directory):
            os.mkdir(class_directory)

"""Save image given label and index in dataset into its corresponding class directory.

    Keyword arguments:
    sample  -- (torch.Tensor, int) - tuple of a labelled image dataset
    idx     -- index of sample in dataset 
"""
def save_image_to_directory(sample, idx):
    image, label = sample
    transform = transforms.ToPILImage()
    image = transform(image)
    img_class = classes[label]
    file_name = file_names_per_class[img_class][idx]
    image.save(os.path.join(classes_dict[img_class],file_name),"JPEG")

def main():
    # load dataset untransformed 
    root = os.path.expanduser("~/share-all/derma-data")
    whole_dataset = datasets.ImageFolder(os.path.join(root,"archive"), transforms.ToTensor())
    orig_path = os.path.join(os.path.expanduser("~/share-all/derma-data"),"archive")
    create_dataset_dir(os.path.join(root,"archive-preprocessed"), orig_path)
    count = 0
    sample_count = 1
    for idx, sample in enumerate(whole_dataset):
        if((sample[1] == sample_count) and ((idx-count) == len(file_names_per_class[classes[sample[1] - 1]]))):
            count = count + len(file_names_per_class[classes[sample[1] - 1]])
            sample_count += 1
        changed_sample = preprocess_image(sample)
        changed_sample = color_constancy(changed_sample)
        save_image_to_directory(changed_sample, idx-count)
if __name__ == "__main__":
    main()