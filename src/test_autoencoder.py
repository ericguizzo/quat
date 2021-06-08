import torch
from torch import nn
import numpy as np
from models import r2he, simple_autoencoder
from torchsummary import summary


x = torch.rand(4,1,128, 128)
model = simple_autoencoder()
print ('input_dim', x.shape)
x=model(x)
print ('output_dim', x.shape)
'''
model = r2he(verbose=True,
             latent_dim=100,
             quat=True,
             #flattened_dim=524288,
             architecture='VGG16')
print (model)

print ('TESTING DIMENSIONS')
print ('input_dim', x.shape)
x,v,a,d=model(x)
'''
