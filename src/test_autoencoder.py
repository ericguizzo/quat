import torch
from torch import nn
import numpy as np
from models import simple_autoencoder, r2he
from torchsummary import summary


x = torch.rand(5,1,512, 128)
model = r2he(quat=True, classifier_quat=True, batch_normalization=False)
#model = simple_autoencoder(quat=True)
print (model)

#model = simple_autoencoder(quat=True)
#d = model.state_dict().keys()
#print (d  == d)
print ('input_dim', x.shape)
y, pred = model(x)
print ('output_dim', y.shape)
l = model.get_embeddings(x)
print ('embeddings: ', l.shape)


#compute number of parameters
model_params = sum([np.prod(p.size()) for p in model.parameters()])

print ('Total paramters: ' + str(model_params))
