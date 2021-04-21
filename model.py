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
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2)
        # parameters!
    
    def forward(self, x1):
        x1 = self.conv(x1)
        if self.Norm:
            x1 = self.batchnorm(x1)
        x1 = self.leakyrelu(x1)
        return x1


class Decoder(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels, kernel, stride, padding, norm = True):
        super(Decoder, self).__init__()
        self.Norm = norm
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size= kernel, stride = stride, padding = 1)
        self.conv = nn.Conv2d(cat_channels, out_channels, kernel_size = 4, stride = 1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        '''need to fix dim'''
        x1 = torch.cat((x1, x2), dim = 3)
        x1 = self.cov(x1)
        if self.Norm:
            x1 = self.batchnorm(x1)
        x1 = self.relu(x1)
        return x1



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode_layers = []
        # THE STRIDE IS 1 here! No BatchNorm in first layer
        self.encode_layers.append(Encoder(1, 64, 4, 1, 0)) # [32, 32, 1] -> [32, 32, 64]
        self.encode_layers.append(Encoder(64, 128, 4, 2, 1)) # [32, 32, 64] -> [16, 16, 128]
        self.encode_layers.append(Encoder(128, 256, 4, 2, 1)) # [16, 16, 128] -> [8, 8, 256]
        self.encode_layers.append(Encoder(256, 512, 4, 2, 1)) # [8, 8, 256] -> [4, 4, 512]
        self.encode_layers.append(Encoder(512, 512, 4, 2, 1)) # [4, 4, 512] -> [2, 2, 512]

        # self.decode_layers = []
        # self.decode_layers.append(Decoder(512, 512+512, 512, 4, 2, 1)) # [2, 2, 512] -> [4, 4, 512] -> [4, 4, 512]
        # self.decode_layers.append(Decoder(512, 256+256, 256, 4, 2, 1)) #[4, 4, 512] -> [8, 8, 256] ->[8, 8, 256]
        # self.decode_layers.append(Decoder(256, 128+128, 128, 4, 2, 1)) # [8, 8, 256] -> [16, 16, 128] -> [16, 16, 128]
        # self.decode_layers.append(Decoder(128, 64+64, 64, 4, 2, 1)) #[16, 16, 128] -> [32, 32, 64] -> [32, 32, 64]
        
        # # last layer is 1 times 1 conv without batchNorm and with tanh activation fucntion
        # # [32, 32, 64] -> [32, 32, 2]
        # self.last_layer = nn.Sequential (
        #     nn.Conv2d(64, 2, kernel_size = 1), 
        #     nn.Tanh()
        # )


    def forward(self, input):
        x1 = self.encode_layers[0](input)
        x2 = self.encode_layers[1](x1)
        x3 = self.encode_layers[2](x2)
        x4 = self.encode_layers[3](x3)
        out = self.encode_layers[4](x4)

        # d4 = self.decode_layers[0](f, x4)
        # d3 = self.decode_layers[1](d4, x3)
        # d2 = self.decode_layers[2](d3, x2)
        # d1 = self.decode_layers[3](d2, x1)
        # out = self.last_layer(d1)
        return out
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer0 = Encoder(2, 64, 4, 2, 0)  # [32, 32, 2] -> [16, 16, 64]  First layer has no BatchNorm
        self.layer1 = Encoder(64, 128, 4, 2, 1) # [16, 16, 64] -> [8, 8, 128]
        self.layer2 = Encoder(128, 256, 4, 2, 1) # [8, 8, 128] -> [4, 4, 256]
        self.layer3 = Encoder(256, 512, 4, 1, 1) # [4, 4, 256] -> [4, 4, 512]
        self.last_layer = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size = 4 ), # stride = 1
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.last_layer(x)
        return x


