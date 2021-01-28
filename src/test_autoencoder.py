import torch
import define_models_torch as mod
from torch import nn
import numpy as np
import time

#test if pretrained model and preprocessed iemocap dataset work
iemocap_path = '../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy'
model_path = '../pretraining_vgg/4secs_inv/model'
#gpu_ID = 1
#device = 'cuda:' + str(gpu_ID)
device = 'cpu'



print ('loading data')
#load input sound from iemicap
iem = np.load(iemocap_path, allow_pickle=True).item()
k = list(iem.keys())
s = k[0]
input = iem[s][0]
input = torch.tensor(input.reshape(1, 1, input.shape[0], input.shape[1])).float().to(device)


#torch.manual_seed(0)
emo_model, p = mod.autoencoder_q(0,1,['output_classes=3'])
emo_model = emo_model.to(device)

start = time.perf_counter()
