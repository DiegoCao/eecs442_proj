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

def torch2numpy(rgb_image):
    numpy_rgb_image = np.transpose(rgb_image.numpy(), (1, 2, 0))
    return numpy_rgb_image

def lab2rgb(imgs):
    lab_imgs = imgs
    rgb_imgs = lab_imgs
    N = lab_imgs.shape[0]
    # print(rgb_img.shape)
    for i in range(N): 
        img = lab_imgs[i]
        img = np.transpose(img.numpy(), (1, 2, 0))
        img *= 255
        img[ :, :, 2] -= 128
        img[ :, :, 1] -= 128
        img[ :, :, 0] /= 2.55
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        rgb_img = torch.from_numpy(np.transpose(img, (2, 0, 1)))*255
        rgb_imgs[i] = rgb_img
    return rgb_imgs