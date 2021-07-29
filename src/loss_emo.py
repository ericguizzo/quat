import torch
import torch.nn.functional as F
import numpy as np



def emo_loss(recon, sounds, truth, pred, beta, at_term=0):
    #split activation (sum quat channels)

    #recon = torch.unsqueeze(torch.sum(recon, axis=1), dim=1) / 4.
    #recon = recon[:,0,:,:]
    #recon = torch.unsqueeze(torch.sum(recon, axis=1), dim=1) / 4.
    #recon = torch.sum(recon**2, axis=1)
    recon = torch.sum(recon, axis=1) / 4.

    #recon_loss = F.binary_cross_entropy_with_logits(recon, sounds.squeeze())

    recon_loss = F.binary_cross_entropy(recon.squeeze(), sounds.squeeze())
    #recon_loss = F.mse_loss(recon, sounds.squeeze())

    #valence_loss = F.mse_loss(v[:,0].squeeze(), truth[:,0].squeeze())
    #arousal_loss = F.mse_loss(a[:,1].squeeze(), truth[:,1].squeeze())
    #dominance_loss = F.mse_loss(d[:,2].squeeze(), truth[:,2].squeeze())

    #emo_loss = beta * (valence_loss + arousal_loss + dominance_loss)
    #emo_loss = beta * F.mse_loss(truth, pred)
    #print ('IMBECILLE', truth.shape, pred.shape)
    emo_loss = beta * F.cross_entropy(pred, torch.argmax(truth, axis=1).long())
    total_loss = (recon_loss) + emo_loss + at_term

    acc = torch.sum(torch.argmax(pred, axis=1) == torch.argmax(truth, axis=1)) / pred.shape[0]
    #total_loss = recon_loss
    #recon_loss = torch.tensor(0)
    #emo_loss = torch.tensor(0)
    #return {'total':total_loss, 'recon': recon_loss.detach().item(), 'emo':emo_loss.detach().item(),
    #    'valence':valence_loss.detach().item(),'arousal':arousal_loss.detach().item(), 'dominance':dominance_loss.detach().item()}
    return {'total':total_loss, 'recon': recon_loss.detach().item(), 'emo':emo_loss.detach().item(),
        'acc':acc.item(),'at':at_term.detach().item()}

    #return {'total':recon_loss}

def emotion_recognition_loss(pred, truth):
    loss = F.cross_entropy(pred, torch.argmax(truth, axis=1).long())
    acc = torch.sum(torch.argmax(pred, axis=1) == torch.argmax(truth, axis=1)) / pred.shape[0]

    return {'loss':loss, 'acc': acc.detach().item()}
