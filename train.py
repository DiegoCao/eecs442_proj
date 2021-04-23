import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os 
from torchsummary import summary
import evaluation
from torch.utils.tensorboard import SummaryWriter

import dataset
import evaluation
import model
import utils

with open("params.json", "r") as read_file:
    params = json.load(read_file)

learning_rate = params["learning_rate"]
num_of_epochs = params["num_of_epochs"]

# load gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

generator = model.Generator()
discriminator = model.Discriminator()
# summary(generator, input_size = (1, 32, 32))
# summary(discriminator, input_size = (2, 32, 32))
G = generator.to(device)
D = discriminator.to(device)

G_opt = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_opt = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

criterion = nn.BCELoss()
reg_criterion = nn.L1Loss()

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")


def saveModel(model, PATH):
    if os.path.exists(PATH) == FALSE:
        os.makedirs(PATH)

    torch.save(model.state_dict(), PATH)


import pickle


def writeData(data, file):
    with open(file, 'wb') as fp:
        pickle.dump(data, fp)


def train():
    step = 0
    g_lambda = 100
    smooth = 0.1

    Running_accu = []
    # loop over the dataset multiple times.
    for epoch in range(num_of_epochs):
        # the generator and discriminator losses are summed for the entire epoch.
        d_running_loss = 0.0
        g_running_loss = 0.0
        for i, data in enumerate(dataset.cielab_loader):
            lab_images = data

            # split the lab color space images into luminescence and chrominance channels.
            l_images = lab_images[:, 0:1, :, :]
            c_images = lab_images[:, 1:, :, :]
            # shift the source and target images into the range [-0.5, 0.5].
            mean = torch.Tensor([0.5])

            l_images = l_images - mean.expand_as(l_images)
            l_images = 2 * l_images

            c_images = c_images - mean.expand_as(c_images)
            c_images = 2 * c_images
            # allocate the images on the default gpu device.
            batch_size = l_images.shape[0]
            l_images = l_images.to(device)
            c_images = c_images.to(device)
            mean = mean.to(device)

            # fake images are generated by passing them through the generator.
            fake_images = generator(l_images)
            fake_images = fake_images.to(device)

            fake_lab_images = torch.cat((l_images, fake_images), dim=1) / 2 + mean

            # Train the discriminator. The loss would be the sum of the losses over
            # the source and fake images, with greyscale images as the condition.
            D_opt.zero_grad()
            d_loss = 0
            logits = discriminator(torch.cat([l_images, c_images], 1)).reshape(-1).cuda()
            d_real_loss = criterion(logits, ((1 - smooth) * torch.ones(batch_size)).cuda())

            logits = discriminator(torch.cat([l_images, fake_images], 1)).reshape(-1)
            d_fake_loss = criterion(logits, (torch.zeros(batch_size)).cuda())

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward(retain_graph=True)
            D_opt.step()

            # Train the generator. The loss would be the sum of the adversarial loss
            # due to the GAN and L1 distance loss between the fake and target images. 
            G_opt.zero_grad()
            g_loss = 0
            fake_logits = discriminator(torch.cat([l_images, fake_images], 1)).reshape(-1)
            g_fake_loss = criterion(fake_logits, (torch.ones(batch_size).cuda()))
            g_image_distance_loss = g_lambda * reg_criterion(fake_images, c_images).cuda()
            g_loss = g_fake_loss + g_image_distance_loss
            g_loss.backward(retain_graph=True)
            G_opt.step()

            # print statistics on pre-defined intervals.
            d_running_loss += d_loss.item()
            g_running_loss += g_loss.item()
            # print(fake_lab_images[3])
            mod_const = 200
            if i % mod_const == 0:
                print('[%d, %5d] d_loss: %.5f g_loss: %.5f' % (
                    epoch + 1, i + 1, d_running_loss / mod_const, g_running_loss / mod_const))
                d_running_loss = 0.0
                g_running_loss = 0.0
                with torch.no_grad():
                    rgb_images_real = utils.lab2rgb(lab_images[:32].cpu())
                    rgb_images_fake = utils.lab2rgb(fake_lab_images[:32].cpu())
                    accu = evaluation.evaluate_batch(rgb_images_real, rgb_images_fake)
                    print('The accuracy with theresh %5: ', accu)
                    Running_accu.append(accu)

                    # print(fake_lab_images[3])
                    # print(rgb_images_fake[3])
                    img_grid_real = torchvision.utils.make_grid(rgb_images_real, normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(rgb_images_fake, normalize=True)
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                step += 1

    writeData(Running_accu, 'accu_correct.txt')
    saveModel(G, './model')
    saveModel(D, './model')
    pass


if __name__ == "__main__":
    train()
