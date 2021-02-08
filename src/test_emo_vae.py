import torch
import torch.nn.functional as F
import numpy as np
from models import emo_vae

def loss_f(input, recon, truth, emo_preds, beta):
    input = input.repeat(1,4,1,1)
    recon_loss = F.binary_cross_entropy(recon, input)
    emo_loss = beta * F.binary_cross_entropy(emo_preds, truth)

    return recon_loss + emo_loss

batch_size = 1
beta = 0.2
label = torch.rand(batch_size,3,requires_grad=True)

x = torch.rand(batch_size,1,512,128)  #dummy batch, same size as spectrograms
model = emo_vae()

outs, preds = model(x)

loss = loss_f(x, outs, label, preds, beta)

print (loss)
