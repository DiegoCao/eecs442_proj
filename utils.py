import torch 
import os 
import torch.nn as nn

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


def rgb2lab()