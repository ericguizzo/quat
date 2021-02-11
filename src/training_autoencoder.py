import sys, os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import models
import torch.utils.data as utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models import *
from loss_emo import *
import utility_functions as uf

parser = argparse.ArgumentParser()
#saving parameters
parser.add_argument('--experiment_name', type=str, default='test')
parser.add_argument('--results_folder', type=str, default='../results')
#dataset parameters
parser.add_argument('--predictors_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy')
parser.add_argument('--target_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_target.npy')
parser.add_argument('--train_perc', type=float, default=0.7)
parser.add_argument('--val_perc', type=float, default=0.2)
parser.add_argument('--test_perc', type=float, default=0.1)
parser.add_argument('--normalize_predictors', type=bool, default=True)
parser.add_argument('--time_dim', type=int, default=512)
parser.add_argument('--freq_dim', type=int, default=128)
parser.add_argument('--fast_test', type=bool, default=True)
#training parameters
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--learning_rate', type=float, default=0.00005)
parser.add_argument('--regularization_lambda', type=float, default=0.)
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--save_model_metric', type=str, default='total_loss')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--load_pretrained', type=str, default=None)
#loss parameters
parser.add_argument('--loss_function', type=str, default='emo_loss')
parser.add_argument('--loss_beta', type=int, default=1.)
#model parameters
parser.add_argument('--model_name', type=str, default='emo_vae')
parser.add_argument('--model_cnn_structure', type=str, default='[32, 64, 128, 256, 512]')
parser.add_argument('--model_classifier_structure', type=str, default='[2000,1000,500,100]')
parser.add_argument('--model_latent_dim', type=int, default=20)
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--model_quat', type=bool, default=True)
#grid search parameters


#eval string args

args = parser.parse_args()

#output filenames
results_folder = os.path.join(args.results_folder, args.experiment_name)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
model_path = os.path.join(results_folder, 'model')
results_path = os.path.join(results_folder, 'results.npy')
figure_path = os.path.join(results_folder, 'figure.png')

if args.use_cuda:
    device = 'cuda:' + str(args.gpu_id)
else:
    device = 'cpu'


print ('Loading dataset')
loading_start = float(time.perf_counter())

#PREDICTORS_LOAD = os.path.join(args.dataset_path, 'iemocap_randsplit_spectrum_fast_predictors.npy')
#TARGET_LOAD = os.path.join(args.dataset_path, 'iemocap_randsplit_spectrum_fast_target.npy')
PREDICTORS_LOAD = args.predictors_path
TARGET_LOAD = args.target_path

dummy = np.load(TARGET_LOAD,allow_pickle=True)
dummy = dummy.item()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#JUST WRITE A FUNCTION TO RE-ORDER foldable_list TO SPLIT
foldable_list = list(dummy.keys())
fold_actors_list = uf.folds_generator(1, foldable_list, [args.train_perc, args.val_perc, args.test_perc])
train_list = fold_actors_list[int(0)]['train']
val_list = fold_actors_list[int(0)]['val']
test_list = fold_actors_list[int(0)]['test']
del dummy

predictors_merged = np.load(PREDICTORS_LOAD,allow_pickle=True)
target_merged = np.load(TARGET_LOAD,allow_pickle=True)
predictors_merged = predictors_merged.item()
target_merged = target_merged.item()

print ('\n building dataset for current fold')
print ('\n training:')
training_predictors, training_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, train_list)
print ('\n validation:')
validation_predictors, validation_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, val_list)
print ('\n test:')
test_predictors, test_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, test_list)


if args.fast_test:
    #take only 100 datapoints, just for quick testing
    bound = 100
    training_predictors = training_predictors[:bound]
    training_target = training_target[:bound]
    validation_predictors = validation_predictors[:bound]
    validation_target = validation_target[:bound]
    test_predictors = test_predictors[:bound]
    test_target = test_target[:bound]


if args.normalize_predictors:
    #normalize to 0 mean and 1 std
    tr_mean = np.mean(training_predictors)
    tr_std = np.std(training_predictors)
    training_predictors = np.subtract(training_predictors, tr_mean)
    training_predictors = np.divide(training_predictors, tr_std)
    validation_predictors = np.subtract(validation_predictors, tr_mean)
    validation_predictors = np.divide(validation_predictors, tr_std)
    test_predictors = np.subtract(test_predictors, tr_mean)
    test_predictors = np.divide(test_predictors, tr_std)



#reshaping for cnn
training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1],training_predictors.shape[2])
validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1], validation_predictors.shape[2])
test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1], test_predictors.shape[2])

#zero-pad/cut time tim
curr_time_dim = training_predictors.shape[2]
curr_freq_dim = training_predictors.shape[3]

if args.time_dim > curr_time_dim:
    #
    training_predictors_padded = np.zeros((training_predictors.shape[0],
                                             training_predictors.shape[1],
                                             args.time_dim,
                                             training_predictors.shape[3]))
    training_predictors_padded[:,:,:curr_time_dim,:] = training_predictors
    training_predictors = training_predictors_padded
    #
    validation_predictors_padded = np.zeros((validation_predictors.shape[0],
                                             validation_predictors.shape[1],
                                             args.time_dim,
                                             validation_predictors.shape[3]))
    validation_predictors_padded[:,:,:curr_time_dim,:] = validation_predictors
    validation_predictors = validation_predictors_padded
    #
    test_predictors_padded = np.zeros((test_predictors.shape[0],
                                             test_predictors.shape[1],
                                             args.time_dim,
                                             test_predictors.shape[3]))
    test_predictors_padded[:,:,:curr_time_dim,:] = test_predictors
    test_predictors = test_predictors_padded

elif args.time_dim < curr_time_dim:
    training_predictors = training_predictors[:,:,:args.time_dim,:]
    validation_predictors = validation_predictors[:,:,:args.time_dim,:]
    test_predictors = test_predictors[:,:,:args.time_dim,:]
else:
    pass

#zero-pad/cut freq tim
if args.freq_dim > curr_freq_dim:
    #
    training_predictors_padded = np.zeros((training_predictors.shape[0],
                                             training_predictors.shape[1],
                                             training_predictors.shape[2],
                                             args.freq_dim))
    training_predictors_padded[:,:,:,:curr_freq_dim] = training_predictors
    training_predictors = training_predictors_padded
    #
    validation_predictors_padded = np.zeros((validation_predictors.shape[0],
                                             validation_predictors.shape[1],
                                             validation_predictors.shape[2],
                                             args.freq_dim))
    validation_predictors_padded[:,:,:,:curr_freq_dim] = validation_predictors
    validation_predictors = validation_predictors_padded
    #
    test_predictors_padded = np.zeros((test_predictors.shape[0],
                                             test_predictors.shape[1],
                                             test_predictors.shape[2],
                                             args.freq_dim))
    test_predictors_padded[:,:,:,:curr_freq_dim] = test_predictors
    test_predictors = test_predictors_padded
elif args.freq_dim < curr_freq_dim:
    training_predictors = training_predictors[:,:,:,:args.freq_dim]
    validation_predictors = validation_predictors[:,:,:,:args.freq_dim]
    test_predictors = test_predictors[:,:,:,:args.freq_dim]
else:
    pass

print ('\nPadded dims:')
print ('Training predictors: ', training_predictors.shape)
print ('Validation predictors: ', validation_predictors.shape)
print ('Test predictors: ', test_predictors.shape)


#convert to tensor
train_predictors = torch.tensor(training_predictors).float()
val_predictors = torch.tensor(validation_predictors).float()
test_predictors = torch.tensor(test_predictors).float()
train_target = torch.tensor(training_target).float()
val_target = torch.tensor(validation_target).float()
test_target = torch.tensor(test_target).float()

#build dataset from tensors
tr_dataset = utils.TensorDataset(train_predictors, train_target)
val_dataset = utils.TensorDataset(val_predictors, val_target)
test_dataset = utils.TensorDataset(test_predictors, test_target)

#build data loader from dataset
tr_data = utils.DataLoader(tr_dataset, args.batch_size, shuffle=True, pin_memory=True)
val_data = utils.DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=True)
test_data = utils.DataLoader(test_dataset, args.batch_size, shuffle=False, pin_memory=True)  #no batch here!!

#load model

model = locals()[args.model_name](structure=eval(args.model_cnn_structure),
               classifier_structure=eval(args.model_classifier_structure),
               latent_dim=args.model_latent_dim,
               verbose=args.verbose,
               quat=args.model_quat)

model = model.to(device)

#print (model)

#compute number of parameters
model_params = sum([np.prod(p.size()) for p in model.parameters()])
print ('Total paramters: ' + str(model_params))

#define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.regularization_lambda)
loss_function = locals()[args.loss_function]

#init history
train_loss_hist = []
val_loss_hist = []

loading_time = float(time.perf_counter()) - float(loading_start)
print ('\nLoading time: ' + str(np.round(float(loading_time), decimals=1)) + ' seconds')


for epoch in range(args.num_epochs):
    epoch_start = time.perf_counter()
    model.train()
    print ('\n')
    string = 'Epoch: [' + str(epoch+1) + '/' + str(args.num_epochs) + '] '
    #iterate batches
    for i, (sounds, truth) in enumerate(tr_data):
        optimizer.zero_grad()
        sounds = sounds.to(device)
        truth = truth.to(device)

        recon, emo_preds = model(sounds)
        loss = loss_function(sounds, recon, truth, emo_preds, args.loss_beta)
        loss['total'].backward()

        optimizer.step()

        #print progress
        perc = int(i / len(tr_data) * 20)
        inv_perc = int(20 - perc - 1)
        loss_print_t = str(np.round(loss['total'].item(), decimals=5))
        string_progress = string + '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' loss: ' + loss_print_t
        print ('\r', string_progress, end='')

    #create history
    train_batch_losses = []
    val_batch_losses = []
    with torch.no_grad():
        model.eval()
        #training data
        for i, (sounds, truth) in enumerate(tr_data):
            sounds = sounds.to(device)
            truth = truth.to(device)

            recon, emo_preds = model(sounds)
            loss = loss_function(sounds, recon, truth, emo_preds, args.loss_beta)

            train_batch_losses.append(loss)
        #validation data
        for i, (sounds, truth) in enumerate(val_data):
            sounds = sounds.to(device)
            truth = truth.to(device)

            recon, emo_preds = model(sounds)
            loss = loss_function(sounds, recon, truth, emo_preds, args.loss_beta)

            val_batch_losses.append(loss)
    #append to history and print
    train_epoch_loss = {'total':[], 'emo':[], 'recon':[],
                        'valence':[],'arousal':[], 'dominance':[]}
    val_epoch_loss = {'total':[], 'emo':[], 'recon':[],
                      'valence':[],'arousal':[], 'dominance':[]}

    for i in train_batch_losses:
        for j in i:
            name = j
            value = i[j]
            train_epoch_loss[name].append(value.item())
    for i in val_batch_losses:
        for j in i:
            name = j
            value = i[j]
            val_epoch_loss[name].append(value.item())

    for i in train_epoch_loss:
        train_epoch_loss[i] = np.mean(train_epoch_loss[i])
        val_epoch_loss[i] = np.mean(val_epoch_loss[i])
    print ('\n EPOCH LOSSES:')
    print ('\n Training:')
    print (train_epoch_loss)
    print ('\n Validation:')
    print (val_epoch_loss)


    train_loss_hist.append(train_epoch_loss)
    val_loss_hist.append(val_epoch_loss)

    #print ('\n  Train loss: ' + str(np.round(train_epoch_loss.item(), decimals=5)) + ' | Val loss: ' + str(np.round(val_epoch_loss.item(), decimals=5)))

    #compute epoch time
    epoch_time = float(time.perf_counter()) - float(epoch_start)
    print ('\n Epoch time: ' + str(np.round(float(epoch_time), decimals=1)) + ' seconds')

    #save best model (metrics = validation loss)
    if epoch == 0:
        torch.save(model.state_dict(), model_path)
        print ('\nModel saved')
        saved_epoch = epoch + 1
    else:
        if args.save_model_metric == 'total_loss':
            best_loss = min([i['total'] for i in val_loss_hist[:-1]])
            #best_loss = min(val_loss_hist['total'].item()[:-1])  #not looking at curr_loss
            curr_loss = val_loss_hist[-1]['total']
            if curr_loss < best_loss:
                torch.save(model.state_dict(), model_path)
                print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                saved_epoch = epoch + 1


        else:
            raise ValueError('Wrong metric selected')

    if args.early_stopping and epoch >= args.patience+1:
        patience_vec = [i['total'] for i in val_loss_hist[-args.patience+1:]]
        #patience_vec = val_loss_hist[-args.patience+1:]
        best_l = np.argmin(patience_vec)
        if best_l == 0:
            print ('Training early-stopped')
            break


#COMPUTE
model.load_state_dict(torch.load(model_path), strict=False)  #load best model
train_batch_losses = []
val_batch_lesses = []
test_batch_losses = []

model.eval()
with torch.no_grad():
    #TRAINING DATA
    for i, (sounds, truth) in enumerate(tr_data):
        sounds = sounds.to(device)
        truth = truth.to(device)

        recon, emo_preds = model(sounds)
        loss = loss_function(sounds, recon, truth, emo_preds, args.loss_beta)

        train_batch_losses.append(loss)

    #VALIDATION DATA
    for i, (sounds, truth) in enumerate(val_data):
        sounds = sounds.to(device)
        truth = truth.to(device)

        recon, emo_preds = model(sounds)
        loss = loss_function(sounds, recon, truth, emo_preds, args.loss_beta)

        val_batch_losses.append(loss)

    #TEST DATA
    for i, (sounds, truth) in enumerate(test_data):
        sounds = sounds.to(device)
        truth = truth.to(device)

        recon, emo_preds = model(sounds)
        loss = loss_function(sounds, recon, truth, emo_preds, args.loss_beta)

        test_batch_losses.append(loss)

#compute final mean of batch losses for train, validation and test set
train_loss = {'total':[], 'emo':[], 'recon':[],
              'valence':[], 'arousal':[], 'dominance':[]}
val_loss = {'total':[], 'emo':[], 'recon':[],
            'valence':[],'arousal':[], 'dominance':[]}
test_loss = {'total':[], 'emo':[], 'recon':[],
             'valence':[],'arousal':[], 'dominance':[]}
for i in train_batch_losses:
    for j in i:
        name = j
        value = i[j]
        train_loss[name].append(value.item())
for i in val_batch_losses:
    for j in i:
        name = j
        value = i[j]
        val_loss[name].append(value.item())
for i in test_batch_losses:
    for j in i:
        name = j
        value = i[j]
        test_loss[name].append(value.item())
for i in train_loss:
    train_loss[i] = np.mean(train_loss[i])
    val_loss[i] = np.mean(val_loss[i])
    test_loss[i] = np.mean(test_loss[i])


#save results in temp dict file
temp_results = {}

#save loss
temp_results['train_loss_total'] = train_loss['total']
temp_results['val_loss_total'] = val_loss['total']
temp_results['test_loss_total'] = test_loss['total']

temp_results['train_loss_recon'] = train_loss['recon']
temp_results['val_loss_recon'] = val_loss['recon']
temp_results['test_loss_recon'] = test_loss['recon']

temp_results['train_loss_emo'] = train_loss['emo']
temp_results['val_loss_emo'] = val_loss['emo']
temp_results['test_loss_emo'] = test_loss['emo']

temp_results['train_loss_valence'] = train_loss['valence']
temp_results['val_loss_valence'] = val_loss['valence']
temp_results['test_loss_valence'] = test_loss['valence']

temp_results['train_loss_arousal'] = train_loss['arousal']
temp_results['val_loss_arousal'] = val_loss['arousal']
temp_results['test_loss_arousal'] = test_loss['arousal']

temp_results['train_loss_dominance'] = train_loss['dominance']
temp_results['val_loss_dominance'] = val_loss['dominance']
temp_results['test_loss_dominance'] = test_loss['dominance']

temp_results['train_loss_hist'] = train_loss_hist
temp_results['val_loss_hist'] = train_loss_hist
temp_results['parameters'] = vars(args)


np.save(results_path, temp_results)

#print  results
print ('\nRESULTS:')
keys = list(temp_results.keys())
keys.remove('parameters')
keys.remove('train_loss_hist')
keys.remove('val_loss_hist')

train_keys = [i for i in keys if 'train' in i]
val_keys = [i for i in keys if 'val' in i]
test_keys = [i for i in keys if 'test' in i]


print ('\n train:')
for i in train_keys:
    print (i, ': ', temp_results[i])
print ('\n val:')
for i in val_keys:
    print (i, ': ', temp_results[i])
print ('\n test:')
for i in test_keys:
    print (i, ': ', temp_results[i])
