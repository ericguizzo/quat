import torch
from torch import nn
import numpy as np
from models import r2he
from torchsummary import summary


x = torch.rand(1,1,128,512)
print ('input_dim', x.shape)

#torch.manual_seed(0)
model = r2he(verbose=True,latent_dim=20)
x,v,a,d=model(x)
