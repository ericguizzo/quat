import torch
import torch.nn.functional as F
import numpy as np




def emo_loss(input, recon, truth, v, a, d, beta):
    #split activation (sum quat channels)
    recon = torch.unsqueeze(torch.sum(recon, axis=1), dim=1)

    recon_loss = F.binary_cross_entropy(recon, input)
    '''
    valence_loss = F.mse_loss(v, truth[:,0])
    arousal_loss = F.mse_loss(a, truth[:,1])
    dominance_loss = F.mse_loss(d, truth[:,2])

    emo_loss = beta * (valence_loss + arousal_loss + dominance_loss)
    total_loss = recon_loss + emo_loss

    return {'total':total_loss, 'recon': recon_loss.detach().item(), 'emo':emo_loss.detach().item(),
        'valence':valence_loss.detach().item(),'arousal':arousal_loss.detach().item(), 'dominance':dominance_loss.detach().item()}
    '''
    return {'total':recon_loss}

def simple_loss(input, recon, truth, v, a, d, beta):
    #just for testing. simplest reconstruction loss
    recon_loss = F.mse_loss(input, recon)

    return {'total':recon_loss, 'recon': torch.tensor([0]), 'emo':torch.tensor([0]),
        'valence':torch.tensor([0]),'arousal':torch.tensor([0]), 'dominance':torch.tensor([0])}

def simplest_loss(input, recon, truth, v, a, d, beta):
    #just for testing. simplest reconstruction loss
    recon_loss = F.mse_loss(input, recon)

    return {'total':recon_loss}
