import torch
import torch.nn.functional as F
import numpy as np
from models import emo_vae

def loss_f(input, recon, truth, emo_preds, beta):
    input = input.repeat(1,4,1,1)
    recon_loss = F.binary_cross_entropy(recon, input)
    valence_loss = F.binary_cross_entropy(emo_preds[:,0],truth[:,0])
    arousal_loss = F.binary_cross_entropy(emo_preds[:,1],truth[:,1])
    dominance_loss = F.binary_cross_entropy(emo_preds[:,2],truth[:,2])

    emo_loss = beta * (valence_loss + arousal_loss + dominance_loss)
    total_loss = recon_loss + emo_loss

    return {'total':total_loss, 'emo':emo_loss, 'valence':valence_loss,
            'arousal':arousal_loss, 'dominance':dominance_loss}

batch_size = 1
beta = 0.2
label = torch.rand(batch_size,3)

x = torch.ones(batch_size,1,512,128)  #dummy batch, same size as spectrograms

model = emo_vae(verbose=True)
print ("\nInput dim:", x.shape)
print ('\nAutoencoder dims:')
outs, preds = model(x)

loss = loss_f(x, outs, label, preds, beta)

print ('\nLoss dict:')
print (loss)
