import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
import torch.utils.data
import torchvision.transforms as transforms
import copy
from PIL import Image
import warnings
import itertools
warnings.filterwarnings("ignore")


class VAE(nn.Module):
    
    def __init__(self, z_dim, nch_input, nch=64, device="cpu"):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nch_input, nch, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(nch),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(nch, nch*2, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(nch*2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(nch*2, nch*4, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(nch*4),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(nch*4, nch*8, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(nch*8),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(nch*8, nch*16, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(nch*16),
                nn.ReLU()
            )
        ])
        self.encmean = nn.Conv2d(nch*16, z_dim, kernel_size=(4,4), stride=1, padding=0)
        self.encvar = nn.Conv2d(nch*16, z_dim, kernel_size=(4,4), stride=1, padding=0)
        
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(z_dim, nch*16, kernel_size=(4,4), stride=1, padding=0),
                nn.BatchNorm2d(nch*16),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch*16, nch*8, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(nch*8),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch*8, nch*4, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(nch*4),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch*4, nch*2, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(nch*2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch*2, nch*1, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(nch),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(nch, nch_input, kernel_size=(4,4), stride=2, padding=1),
                nn.BatchNorm2d(nch_input)
            ),
        ])
        
    def _encoder(self, x):
        for layer in self.encoder:
            x = layer(x)
        mean = self.encmean(x)
        var = F.softplus(self.encvar(x))
        return mean, var
    
    def _decoder(self, z):
        for layer in self.decoder:
            z = layer(z)
        z = torch.tanh(z)
        z = torch.clamp(z, min=-1+1e-8, max=1-1e-8)
        return z
    
    def _samplez(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        return mean + torch.sqrt(var) * epsilon
    
    def forward(self, x):
        mean, var = self._encoder(x)
        z = self._samplez(mean, var)
        y = self._decoder(z)
        return y, z
    
    def loss(self, x, beta=1):
        mean, var = self._encoder(x)
        # KL divergence
        KL = -0.5 * torch.sum(1 + torch.log(var+1e-8) - mean**2 - var)  / x.shape[0]
        z = self._samplez(mean, var)
        y = self._decoder(z)
        # reconstruction error
        recon = F.mse_loss(y.view(-1), x.view(-1), size_average=False) / x.shape[0]
        # combine them
        lower_bound = [beta*KL, recon]
        return sum(lower_bound)
    
    
class Discriminator(nn.Module):
    
    def __init__(self, nch_input, nch_output, nch=64, n_layers=4):
        super(Discriminator, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(nch_input, nch, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(nch))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        
        in_nch = nch
        for _ in range(n_layers):
            out_nch = in_nch * 2
            layers.append(nn.Conv2d(in_nch, out_nch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_nch))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            in_nch = out_nch
            
        layers.append(nn.Conv2d(out_nch, nch_output, kernel_size=1, stride=1, padding=0))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x).squeeze()
    
    
class Generator(nn.Module):
    
    def __init__(self, ndim, nch_output, nch=64, n_layers=4):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(ndim, nch*2**n_layers, kernel_size=4, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(nch*2**n_layers))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        
        in_nch = nch*2**n_layers
        for _ in range(n_layers):
            out_nch = in_nch // 2
            layers.append(nn.ConvTranspose2d(in_nch, out_nch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_nch))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
            in_nch = out_nch
            
        layers.append(nn.ConvTranspose2d(in_nch, nch_output, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(nch_output))
        layers.append(nn.Tanh())
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.layers(z)
