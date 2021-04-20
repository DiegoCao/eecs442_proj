import os 
import numpy as np
import torch 
import torchvision

import torch.nn as nn
# create an abtract class base model
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, norm = True):

        super(Encoder, self).__init__()
        self.Norm = norm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel, stride = stride, padding = padding)
        if self.Norm == True:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        # parameters!
    
    def forward(self, x1):
        x1 = self.conv(x1)
        if self.Norm:
            x1 = self.batchnorm(x1)
        x1 = self.leakyrelu(x1)
        return x1


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, norm = True):

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size= kernel, stride = stride, padding = 1)
        self.relu = nn.ReLU()
    
    def forward(self, x1, x2):

        x1 = self.upsample(x1)
        x1 = torch.cat((x1, x2), dim = 1)
        x1 = self.relu(x1)

# def encode(in_channels, out_channels, kernel, padding, stride):

#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel, stride = stride, padding=padding),
#         nn.BatchedNorm2d(out_channels),
#         # add a batched normed layer
#         # Use Leaky ReLU
#         nn.LeakyReLU(inplace=True),
#     )

#     pass 



class Generator(nn.Module):
    def __init__(self):
        self.num_of_layers = 10
        self.kernelsize = 3
        self.stride = 2
        self.paddingsize = 1
        # the encode
        


        
        self.encode_layers = []
        self.encode_layers.append(Encoder(1, 64, 3, 1, 1)) # THE STRIDE IS 1 here!
        self.encode_layers.append(Encoder(64, 128, 3, 2, 1))
        self.encode_layers.append(Encoder(128, 256, 3, 2, 1))
        self.encode_layers.append(Encoder(256, 512, 3, 2, 1))
        self.encode_layers.append(Encoder(512, 512, 3, 2, 1))


        self.decode_layers = []
        self.decode_layers.append(Decoder(512, 512, 3, 2, 1))
        self.decode_layers.append(Decoder(512, 256, 3, 2, 1))
        self.decode_layers.append(Decoder(256, 128, 3, 2, 1))
        self.decode_layers.append(Decoder(128, 64, 3, 2, 1))
        
        
        self.last_conv = nn.Conv2d(64, 2, kernel_size = 1)
        


    def forward(self, input):
        x1 = self.encode_layers[0](input)
        x2 = self.encode_layers[1](x1)
        x3 = self.encode_layers[2](x2)
        x4 = self.encode_layers[3](x3)
        f = self.encode_layers[4](x4)

        d4 = self.decode_layers[3](f, x4)
        d3 = self.decode_layers[2](d4, x3)
        d2 = self.decode_layers[1](d3, x2)
        d1 = self.decode_layers[0](d2, x1)
        out = self.last_conv(d1)

        return out
        



        # self.last_layer = nn.Conv2d(64, )
class Discriminator(nn.Module):
    def __init__(self):
        self.layer0 = nn.Conv2d(1, )
    