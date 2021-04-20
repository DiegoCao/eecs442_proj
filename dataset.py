# Create by Hangrui Cao, load dataset, pytorch version implementation of paper:
# Thanks to efforts by https://github.com/ImagingLab/Colorizing-with-GANs (Code in tensorflow, more complex version)

import os 
import numpy as np

import torch 
import torchvision 

class Dataset():
    def __init__(self, name, images):
        self.name = name
        self.images = images
    
    def __len__(self):
        return len(self.images)
        

    
    
