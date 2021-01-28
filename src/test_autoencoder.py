import torch
import define_models_torch as mod
from torch import nn
import numpy as np

#test if pretrained model and preprocessed iemocap dataset work
predictors_path = '/Users/eric/Desktop/sapienza/quat/temp/iemocap_randsplit_spectrum_fast_test_predictors_fold_0.npy'
target_path = '/Users/eric/Desktop/sapienza/quat/temp/iemocap_randsplit_spectrum_fast_test_target_fold_0.npy'
#gpu_ID = 1
#device = 'cuda:' + str(gpu_ID)
device = 'cpu'
input_id = 1

predictors = np.load(predictors_path)
target = np.load(target_path)

x = predictors[input_id]
y = target[input_id]
x = torch.tensor(x.reshape(1, 1, x.shape[0], x.shape[1])).float().to(device)


print (x.shape)

#torch.manual_seed(0)
model, p = mod.autoencoder_q(0,1,['verbose=True'])
model = model.to(device)
model(x)
