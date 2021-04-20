import torch 
import utils 
import model 
import torch.nn as nn 
import torch.optim as optim

from torchsummary import summary 
import json



def train(images):
    generator = model.Generator()

    # load gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = generator.to(device)
    summary(generator, input_size = (1, 32, 32))

    discriminator = model.Discriminator()
    

    
    


if __name__ == "__main__":
    images = utils.downloadImages()

