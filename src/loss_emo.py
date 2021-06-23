import torch
import torch.nn.functional as F
import numpy as np




def emo_loss(recon, sounds, truth, pred, beta):
    recon = torch.sum(recon, axis=1) / 4.
    recon_loss = F.binary_cross_entropy(recon, sounds.squeeze())
    emo_loss = beta * F.cross_entropy(pred, torch.argmax(truth, axis=1).long())
    total_loss = (recon_loss) + emo_loss
    acc = torch.sum(torch.argmax(pred, axis=1) == torch.argmax(truth, axis=1)) / pred.shape[0]

    return {'total':total_loss, 'recon': recon_loss.detach().item(), 'emo':emo_loss.detach().item(),
        'valence':acc.detach().item(),'arousal':0, 'dominance':0}

def emotion_recognition_loss(pred, truth):
    loss = F.cross_entropy(pred, torch.argmax(truth, axis=1).long())
    acc = torch.sum(torch.argmax(pred, axis=1) == torch.argmax(truth, axis=1)) / pred.shape[0]

    return {'loss':total_loss, 'acc': recon_loss.detach().item()}
