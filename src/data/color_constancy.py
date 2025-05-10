import numpy as np
import torch

""" Adjust illumination of image with the Shades of Gray Color Constancy algorithm, adopted from:

    Nick Shawn, Shades_of_Gray-color_constancy_transformation, 2018, Github Repository
    Link: https://github.com/nickshawn/Shades_of_Gray-color_constancy_transformation

    Keyword arguments:
    image -- pytorch tensor image
    p_power -- power of Minkowski - norm
    gamma -- gamma - value for gamma correction
"""
def compute_color_constancy(image, p_power= 6):
    image = image.numpy()
    img = image.astype('float32')
    img_power = np.power(img, p_power)
    rgb_vec = np.power(np.mean(img_power, (1,2)), 1/p_power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec[:, None, None])
    np.where(image > 1, image, 1.0)
    return torch.from_numpy(img)