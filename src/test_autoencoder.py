import torch
import define_models_torch as mod
from torch import nn
import numpy as np
import time

#test if pretrained model and preprocessed iemocap dataset work
predictors_path = '/Users/eric/Desktop/sapienza/quat/temp/iemocap_randsplit_spectrum_fast_test_predictors_fold_0.npy'
target_path = '/Users/eric/Desktop/sapienza/quat/temp/iemocap_randsplit_spectrum_fast_test_target_fold_0.npy'
#gpu_ID = 1
#device = 'cuda:' + str(gpu_ID)
device = 'cpu'

predictors = np.load(predictors_path)
target = np.load(target_path)

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
