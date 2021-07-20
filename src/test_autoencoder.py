import torch
from torch import nn
import numpy as np
from models import simple_autoencoder, VGGNet, AlexNet, resnet50
from torchsummary import summary
from torchvision import models


x = torch.rand(5,4,512, 128)
model = resnet50(num_classes=4, quat=True)
print (model)
x = model(x)
print (x.shape)
model_params = sum([np.prod(p.size()) for p in model.parameters()])
print ('Total paramters: ' + str(model_params))

'''
model = simple_autoencoder(quat=True)
print ('input_dim', x.shape)
x, pred = model(x)
print ('output_dim', x.shape)

#compute number of parameters
model_params = sum([np.prod(p.size()) for p in model.parameters()])
print ('Total paramters: ' + str(model_params))
'''
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
