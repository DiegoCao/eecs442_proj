# Create by Hangrui Cao and Ruiyu Li, load dataset, pytorch version implementation of paper:
# Thanks to efforts by https://github.com/ImagingLab/Colorizing-with-GANs (Code in tensorflow, more complex version)

# import os 
# import numpy as np

# import torch 
# import torchvision 

# class Dataset():
#     def __init__(self, name, images):
#         self.name = name
#         self.images = images

#     def __len__(self):
#         return len(self.images)


import json

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

with open("params.json", "r") as read_file:
    params = json.load(read_file)

if params["dataset"] == "flower":
    tensor_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    trainset = torchvision.datasets.ImageFolder(root='./flower_data/train', transform=tensor_transform)
else:
    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors.
    tensor_transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=tensor_transform)
    


def processImages(imageset):
    """
        Given image set, return the testimages
    """
    rgb_images = []
    np_lab_images = []
    for image, label in imageset:
        rgb_images.append(image)

    
    for rgb_image in rgb_images:
        np_rgb_image = np.transpose(rgb_image.numpy(), (1, 2, 0))
        # Convert it to LAB

        np_lab_image = cv2.cvtColor(numpy_rgb_image, cv2.COLOR_RGB2LAB)
        np_lab_images.append(np_lab_image)

    for np_lab_image in np_lab_images:
        np_lab_image[:, :, 0] *= 255 / 100
        np_lab_image[:, :, 1] += 128
        np_lab_image[:, :, 2] += 128
        np_lab_image /= 255
        torch_lab_image = torch.from_numpy(np.transpose(numpy_lab_image, (2, 0, 1)))
        lab_images.append(torch_lab_image)



# classes = ('plane', 'car', 'bbird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#########################################################################
# Transform the images to CieLAB color space by the use of OpenCV library.
rgb_images = []
numpy_lab_images = []

testmode = False
if params["testmode"] == "True":
    testmode = True
    tensor_transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(root='./testdata', train=False,
                                       download=True, transform=tensor_transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                      shuffle=False, num_workers=2)


if testmode == False:
    for image, label in trainset:
        rgb_images.append(image)
        print(image.shape)

if testmode == True:
    for image, label in testset:
        rgb_images.append(image)
        print(image.shape)


# f1 = np.transpose(rgb_images[2].numpy(), (1, 2, 0))
# f1 = cv2.cvtColor(f1, cv2.COLOR_RGB2LAB)
# # print(f1)
# f1[:, :, 0] *= 255 / 100
# f1[:, :, 1] += 128
# f1[:, :, 2] += 128
# f1 /= 255
# torch_f1 = torch.from_numpy(np.transpose(f1, (2, 0, 1)))
# lab_f1 = np.transpose(torch_f1.numpy(), (1, 2, 0))
# lab_f1 *= 255
# lab_f1[ :, :, 2] -= 128
# lab_f1[ :, :, 1] -= 128
# lab_f1[ :, :, 0] /= 2.55
# rgb_f1 = cv2.cvtColor(f1, cv2.COLOR_LAB2RGB)
# cv2.imwrite("f1.png", rgb_f1*255)

for rgb_image in rgb_images:
    numpy_rgb_image = np.transpose(rgb_image.numpy(), (1, 2, 0))
    numpy_lab_image = cv2.cvtColor(numpy_rgb_image, cv2.COLOR_RGB2LAB)
    numpy_lab_images.append(numpy_lab_image)

######################################################################
# Transform the numpy lab images to images of range [0, 1] and further
# convert them to tensors.
lab_images = []
for numpy_lab_image in numpy_lab_images:
    numpy_lab_image[:, :, 0] *= 255 / 100
    numpy_lab_image[:, :, 1] += 128
    numpy_lab_image[:, :, 2] += 128
    numpy_lab_image /= 255
    torch_lab_image = torch.from_numpy(np.transpose(numpy_lab_image, (2, 0, 1)))
    lab_images.append(torch_lab_image)


#######################################################################
# Make a custom CieLAB dataset and a data loader that iterates over the
# custom dataset with shuffling and a batch size of 128.
class CieLABDataset(torch.utils.data.Dataset):
    """CieLab dataset."""

    def __len__(self):
        return len(lab_images)

    def __getitem__(self, index):
        img = lab_images[index]
        return img




cielab_dataset = CieLABDataset()
if params["dataset"] == "flower":
    cielab_loader = torch.utils.data.DataLoader(cielab_dataset, batch_size=16,
                                                shuffle=True, num_workers=2)
else:
    cielab_loader = torch.utils.data.DataLoader(cielab_dataset, batch_size=128,
                                            shuffle=True, num_workers=2)

# test_loader = CieLABDataset()

# if params["testmode"] == "True"
#     test_loader = 
