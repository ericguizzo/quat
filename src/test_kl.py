import torch
import torch.nn.functional as F
import numpy as np

a = torch.rand(1,1,1,128)  #output quaternion spectrogram (dims=[batch,quat,time,freq])
b = torch.empty(1,1,1,128).normal_(mean=0,std=0.9)  #quaternion-valued label (dims=[batch,emotion:(none,valence arousal,dominance),none,value])
b1 = torch.ones(1,1,1,1) * 20


kl = F.kl_div(torch.log(a),b)
kl1 = F.kl_div(torch.log(a), b1)

el = torch.mean(-0.5 * torch.sum(1 + a - b ** 2 - a.exp(), dim = 1), dim = 0)
print('\nEL', torch.mean(el))
print ('mean', torch.mean(a))
print ('kl divergence: ', kl, kl1)
