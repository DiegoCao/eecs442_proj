import torch
import torchvision


import os 
from torchsummary import summary
import evaluation
from torch.utils.tensorboard import SummaryWriter

import dataset
import evaluation
import model
import utils

writer_test = SummaryWriter(f"testlog/test")
writer_real = SummaryWriter(f"testlog/test")

def test():
    G = model.Generator() 
    G.load_state_dict(torch.load('./model/G.pt'))

    batch_size = 128
    testset = torchvision.datasets.CIFAR10(root='./testdata', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    for i, data in enumerate(testloader):
        lab_images = data
        print('step i ', i)
        
        l_images = lab_images[:, 0:1, :, :]
        # c_images = lab_images[:, 1:, :, :]
            # shift the source and target images into the range [-0.5, 0.5].
        mean = torch.Tensor([0.5])

        l_images = l_images - mean.expand_as(l_images)
        l_images = 2 * l_images

        # c_images = c_images - mean.expand_as(c_images)
        # c_images = 2 * c_images
        
        test_res_image = G(l_images)
        
        with torch.no_grad():
            rgb_images_test = utils.lab2rgb(test_res_images[:32].cpu())
        
            rgb_images_real = utils.lab2rgb(lab_images[:32].cpu())
            img_grid_real = torchvision.utils.make_grid(rgb_images_real, normalize=True)
            img_grid_test = torchvision.utils.make_grid(rgb_images_test, normalize=True)
            writer_real.add_image("Real", img_grid_real, global_step=i)
            writer_fake.add_image("Test", img_grid_test, global_step=i)
        

if __name__ == "__main__":
    test()
    