import torch
import torch.nn.functional as F
import numpy as np


training_predictors = torch.rand(1,1,501,129)  #output quaternion spectrogram (dims=[batch,quat,time,freq])
validation_predictors = torch.rand(1,1,501,129)  #output quaternion spectrogram (dims=[batch,quat,time,freq])
test_predictors = torch.rand(1,1,501,129)
print (training_predictors.shape)
print (validation_predictors.shape)
print (test_predictors.shape)
print ('/////////')
time_dim = 512
freq_dim = 128

curr_time_dim = training_predictors.shape[2]
curr_freq_dim = training_predictors.shape[3]

#zero-pad/cut time tim
if time_dim > curr_time_dim:
    #
    training_predictors_padded = torch.zeros(training_predictors.shape[0],
                                             training_predictors.shape[1],
                                             time_dim,
                                             training_predictors.shape[3])
    training_predictors_padded[:,:,:curr_time_dim,:] = training_predictors
    training_predictors = training_predictors_padded
    #
    validation_predictors_padded = torch.zeros(validation_predictors.shape[0],
                                             validation_predictors.shape[1],
                                             time_dim,
                                             validation_predictors.shape[3])
    validation_predictors_padded[:,:,:curr_time_dim,:] = validation_predictors
    validation_predictors = validation_predictors_padded
    #
    test_predictors_padded = torch.zeros(test_predictors.shape[0],
                                             test_predictors.shape[1],
                                             time_dim,
                                             test_predictors.shape[3])
    test_predictors_padded[:,:,:curr_time_dim,:] = test_predictors
    test_predictors = test_predictors_padded

elif time_dim < curr_time_dim:
    training_predictors = training_predictors[:,:,:time_dim,:]
    validation_predictors = validation_predictors[:,:,:time_dim,:]
    test_predictors = test_predictors[:,:,:time_dim,:]
else:
    pass

#zero-pad/cut freq tim
if freq_dim > curr_freq_dim:
    #
    training_predictors_padded = torch.zeros(training_predictors.shape[0],
                                             training_predictors.shape[1],
                                             training_predictors.shape[2],
                                             freq_dim)
    training_predictors_padded[:,:,:,:curr_freq_dim] = training_predictors
    training_predictors = training_predictors_padded
    #
    validation_predictors_padded = torch.zeros(validation_predictors.shape[0],
                                             validation_predictors.shape[1],
                                             validation_predictors.shape[2],
                                             freq_dim)
    validation_predictors_padded[:,:,:,:curr_freq_dim] = validation_predictors
    validation_predictors = validation_predictors_padded
    #
    test_predictors_padded = torch.zeros(test_predictors.shape[0],
                                             test_predictors.shape[1],
                                             test_predictors.shape[2],
                                             freq_dim)
    test_predictors_padded[:,:,:,:curr_freq_dim] = test_predictors
    test_predictors = test_predictors_padded
elif freq_dim < curr_freq_dim:
    training_predictors = training_predictors[:,:,:,:freq_dim]
    validation_predictors = validation_predictors[:,:,:,:freq_dim]
    test_predictors = test_predictors[:,:,:,:freq_dim]
else:
    pass

print (training_predictors.shape)
print (validation_predictors.shape)
print (test_predictors.shape)
