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
    if isinstance(at_term, int):
        at_term = 0.
    elif isinstance(at_term, float):
        pass
    else:
        at_term = at_term.detach().item()
    return {'total':total_loss, 'recon': recon_loss.detach().item(), 'emo':emo_loss.detach().item(),
        'acc':acc.item(),'at':at_term}

    #return {'total':recon_loss}


def emo_loss_vad(recon, sounds, truth, pred, beta, beta_vad, at_term=0):

    recon = torch.sum(recon, axis=1) / 4.
    recon_loss = F.binary_cross_entropy(recon.squeeze(), sounds.squeeze())
    c_p, v_p, a_p, d_p = pred

    print ('cccaaaaaaCVERGIWJFEOVIWFNVS', torch.argmax(c_p, axis=1), truth[:,0].long())
    valence_loss = F.binary_cross_entropy(v_p.squeeze(), truth[:,1].squeeze())
    arousal_loss = F.binary_cross_entropy(a_p.squeeze(), truth[:,2].squeeze())
    dominance_loss = F.binary_cross_entropy(d_p.squeeze(), truth[:,3].squeeze())
    classification_loss = F.cross_entropy(torch.argmax(c_p, axis=1), truth[:,0].long())
    vad_loss = beta_vad * (valence_loss + arousal_loss + dominance_loss)
    emo_loss = beta * (classification_loss + vad_loss)

    total_loss = recon_loss + emo_loss + at_term

    acc = torch.sum(torch.argmax(pred["class"], axis=1) == torch.argmax(truth, axis=1)) / pred["class"].shape[0]
    acc_valence = torch.sum(pred["valence"] == truth["valence"]) / pred["valence"].shape[0]
    acc_arousal = torch.sum(pred["arousal"] == truth["arousal"]) / pred["arousal"].shape[0]
    acc_dominance = torch.sum(pred["dominance"] == truth["dominance"]) / pred["dominance"].shape[0]

    if isinstance(at_term, int):
        at_term = 0.
    elif isinstance(at_term, float):
        pass
    else:
        at_term = at_term.detach().item()

    output =  {'total':total_loss, 'recon': recon_loss.detach().item(), 'emo':emo_loss.detach().item(),
        'acc':acc.item(),'at':at_term, 'vad':vad_loss.detach().item(), 'valence':valence_loss.detach().item(),
        'arousal':arousal_loss.detach().item(), 'dominance':dominance_loss.detach().item(),
        'acc_valence':acc_valence.detach().item(), 'acc_arousal':acc_arousal.detach().item(),
        'acc_dominance':acc_dominance.detach().item()}

    return output


def emotion_recognition_loss(pred, truth):
    loss = F.cross_entropy(pred, torch.argmax(truth, axis=1).long())
    acc = torch.sum(torch.argmax(pred, axis=1) == torch.argmax(truth, axis=1)) / pred.shape[0]

    return {'loss':loss, 'acc': acc.detach().item()}
