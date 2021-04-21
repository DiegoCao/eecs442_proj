import torch 
import os 
import torch.nn as nn
import numpy as np
import cv2

import torchvision 
def downloadImages(whichdata = 'cifar10', data_dir = './data'):
    if os.path.exists(data_dir) == False:
        os.makedirs(data_dir)
    
    images = None 
    if whichdata == "cifar10":
        # load images and transforms them to pytorch tensors
        images = torchvision.datasets.CIFAR10(root = data_dir, transform = torchvision.transforms.ToTensor())

    # set up warnings 
    if images == None:
        raise FileNotFoundError

    return images 


def lab2rgb(imgs):
    lab_img = np.transpose(imgs.numpy(), (0, 2, 3, 1))
    rgb_img = np.zeros(lab_img.shape)
    N = lab_img.shape[0]
    # print(lab_img)
    for i in range(N): 
        img = lab_img[i]
        mean = np.array([0.5])
        img = img/2
        img += mean
        img *= 255
        img[:, :, 2] -= 128
        img[:, :, 1] -= 128
        img[:, :, 0] /= 2.55
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        img = img.astype(np.uint8)
        rgb_img[i] = img
    tensor_rgb_image = torch.from_numpy(np.transpose(rgb_img, (0, 3, 1, 2)))
    # print(tensor_rgb_image)
    return tensor_rgb_image