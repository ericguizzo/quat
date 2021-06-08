import torch
from torch import nn
import numpy as np
from models import r2he
from torchsummary import summary


x = torch.rand(1,1,512, 128)

model = r2he(verbose=True,latent_dim=20, quat=False)
print (model)

print ('TESTING DIMENSIONS')
print ('input_dim', x.shape)
x,v,a,d=model(x)
