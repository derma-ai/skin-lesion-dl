from cgitb import grey
import os
import torch
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

def color_constancy():
    return

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
    root = os.path.join("/","space","derma-data", "isic_2019")
    whole_dataset = datasets.ImageFolder(os.path.join(root,"clean"), transforms.ToTensor())
    orig_path = os.path.join(root,"clean")
    create_dataset_dir(os.path.join(root,"preprocessed"), orig_path)
    count = 0
    sample_count = 1
    for idx, sample in enumerate(whole_dataset):
        if((sample[1] == sample_count) and ((idx-count) == len(file_names_per_class[classes[sample[1] - 1]]))):
            count = count + len(file_names_per_class[classes[sample[1] - 1]])
            sample_count += 1
            print("Current index:",idx,"Current idx-count:",idx - count,"Current class label:", sample[1],"Current count value:", count,"Current number of samples of the previous class:", len(file_names_per_class[classes[sample[1] - 1]]))
        changed_sample = preprocess_image(sample)
        save_image_to_directory(changed_sample, idx-count)
if __name__ == "__main__":
    main()
