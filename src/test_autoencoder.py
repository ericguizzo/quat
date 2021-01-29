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
padding_dims = [512, 128]
predictors = np.load(predictors_path)
target = np.load(target_path)

#fit to correct size for the autoencoders
predictors_padded = np.zeros(shape=(predictors.shape[0], padding_dims[0], padding_dims[1]))  #zero pad time
predictors_padded[:,:predictors.shape[1], :predictors.shape[2]] = predictors[:,:,:predictors.shape[2]-1] #cut 1 freq bin

#print (predictors.shape)
#print (predictors_padded.shape)

x = predictors_padded[input_id]
y = target[input_id]
x = torch.tensor(x.reshape(1, 1, x.shape[0], x.shape[1])).float().to(device)

x = torch.rand(1,1,512,128)
print ('input_dim', x.shape)

#torch.manual_seed(0)
model, p = mod.autoencoder_q(0,1,['verbose=True', 'latent_dim=20'])
model = model.to(device)
model(x)
