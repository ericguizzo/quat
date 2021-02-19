import torch
import torch.nn.functional as F
import numpy as np

def emo_loss_old(input, recon, truth, emo_preds, beta):
    input = input.repeat(1,4,1,1)
    recon_loss = F.binary_cross_entropy(recon, input)
    valence_loss = F.mse_loss(emo_preds[:,0],truth[:,0])
    arousal_loss = F.mse_loss(emo_preds[:,1],truth[:,1])
    dominance_loss = F.mse_loss(emo_preds[:,2],truth[:,2])

    emo_loss = beta * (valence_loss + arousal_loss + dominance_loss)
    total_loss = recon_loss + emo_loss

    return {'total':total_loss, 'recon': recon_loss.detach(), 'emo':emo_loss.detach(),
        'valence':valence_loss.detach(),'arousal':arousal_loss.detach(), 'dominance':dominance_loss.detach()}

def emo_loss(input, recon, truth, v, a, d, beta):
    #input = input.repeat(1,4,1,1)
    recon_loss = F.binary_cross_entropy(input, recon[:,0,:,:])

    valence_loss = F.mse_loss(v, truth[:,0])
    arousal_loss = F.mse_loss(a, truth[:,1])
    dominance_loss = F.mse_loss(d, truth[:,2])

    emo_loss = beta * (valence_loss + arousal_loss + dominance_loss)
    total_loss = recon_loss + emo_loss

    return {'total':total_loss, 'recon': recon_loss, 'emo':emo_loss,
        'valence':valence_loss,'arousal':arousal_loss, 'dominance':dominance_loss}

def simple_loss(input, recon, truth, v, a, d, beta):
    #just for testing. simplest reconstruction loss
    recon_loss = F.binary_cross_entropy(input, recon)

    return {'total':recon_loss, 'recon': 0, 'emo':0,
        'valence':0,'arousal':0, 'dominance':0}
