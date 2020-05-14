import argparse
import os
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import imageio

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision.utils as vutils


import torch.nn as nn
import torch.nn.functional as F
import torch

from models.dcgan import Generator, Discriminator

os.makedirs("images", exist_ok=True)

n_epochs = 10000
batch_size = 128
img_size = 64
lr = 1E-4
latent_dim = 100
channels = 3
n_critic = 5
sample_interval = 100
ngpu = 1
ngf = 128 #generator filter
ndf = 128 #generator filter
beta1 = 0.5 # Using Adam
clip_value = 0.01

cuda = True if torch.cuda.is_available() else False

device = 'cuda:0'
img_shape = (channels, img_size, img_size)

to_image = transforms.ToPILImage()

trans = transforms.Compose([
    
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


G = Generator(latent_dim, ngf, channels).cuda(device)
D = Discriminator(ndf, channels).cuda(device)


G.apply(weights_init)
D.apply(weights_init)

dataloader = torch.utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(
                                            root ='C:/Users/우리집/MyWGAN/images',
                                            transform = trans),batch_size = batch_size )

Optimizer_D = torch.optim.RMSprop(D.parameters(), lr=lr)
Optimizer_G = torch.optim.RMSprop(G.parameters(), lr=lr)
# Optimizer_G = torch.optim.Adam(G.parameters(),lr=lr, betas=(beta1, 0.999))
# Optimizer_D = torch.optim.Adam(D.parameters(),lr=lr, betas=(beta1, 0.999))

G_losses = []
D_losses = []
images = []

input = torch.Tensor(batch_size, 3, img_size, img_size)
noise = torch.Tensor(batch_size, latent_dim, 1, 1)
fixed_noise = torch.Tensor(batch_size, latent_dim, 1, 1).normal_(0,1).cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

batches_done = 0

for epoch in range(n_epochs):

    for i , (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.cuda(device)
        real_cpu  = imgs
        
        Optimizer_D.zero_grad()

        noise.resize_(batch_size, latent_dim, 1, 1).normal_(0, 1)
        z = noise.cuda().detach()
        fake_imgs = G(z)
            
        loss_D = - torch.mean(D(real_imgs)) + torch.mean(D(fake_imgs))
        
        loss_D.backward()
        Optimizer_D.step()
        
        for p in D.parameters():
            p.data.clamp_(-clip_value, clip_value)
     
        if i % n_critic == 0:
            
            Optimizer_G.zero_grad()
            
            gen_imgs = G(z)            
            loss_G = - torch.mean(D(gen_imgs))
            
            loss_G.backward()
            Optimizer_G.step()
        if i % 10 ==0:
            G_losses.append(loss_G)
            D_losses.append(loss_D)
            plt.clf()
            plt.plot(G_losses, label='Generator Losses')
            plt.plot(D_losses, label='Critic Losses')
            plt.legend()
            plt.savefig('loss.png')
            plt.show()            
            
    if epoch % 5 == 0:

        img = G(fixed_noise).detach().cpu()
        img = img.data.mul(0.5).add(0.5)
        img = make_grid(img)
        images.append(to_image(img))
        imageio.mimsave('progress.gif', [np.array(i) for i in images])
        #print('Epoch {}: G_loss: {:.4f} D_loss: {:.4f}'.format(epoch, loss_G, loss_D))
        print('{}/{} G_loss: {:.4f}  D_loss: {:.4f}'
              .format(epoch, n_epochs, loss_G, loss_D))
        if batches_done % 10 == 0:
            #real_cpu = real_cpu.mul(0.5).add(0.5)
            #vutils.save_image(real_cpu, '{0}/real_samples.png'.format('samples'))
            fake = G(fixed_noise.detach())
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format('samples', batches_done))    
    batches_done += 1
    
print('Training Finished')
torch.save(G.state_dict(), 'cifar10_generator.pth')
torch.save(D.state_dict(), 'cifar10_discriminator.pth')
