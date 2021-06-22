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
from tqdm import tqdm

parser = argparse.ArgumentParser()
#saving parameters
parser.add_argument('--experiment_name', type=str, default='test')
parser.add_argument('--results_folder', type=str, default='../results')
parser.add_argument('--results_path', type=str, default='../results/results.npy')
parser.add_argument('--model_path', type=str, default='../results/model')
#dataset parameters
#'../new_experiments/experiment_1_beta0.txt/models/model_xval_iemocap_exp1_beta0.txt_run1_fold0'
parser.add_argument('--predictors_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy')
parser.add_argument('--target_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_target.npy')
parser.add_argument('--train_perc', type=float, default=0.7)
parser.add_argument('--val_perc', type=float, default=0.2)
parser.add_argument('--test_perc', type=float, default=0.1)
parser.add_argument('--normalize_predictors', type=str, default='True')
parser.add_argument('--time_dim', type=int, default=512)
parser.add_argument('--freq_dim', type=int, default=128)
parser.add_argument('--fast_test', type=str, default='True')
parser.add_argument('--fast_test_bound', type=int, default=5)

#training parameters
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.00001)
parser.add_argument('--regularization_lambda', type=float, default=0.)
parser.add_argument('--early_stopping', type=str, default='True')
parser.add_argument('--save_model_metric', type=str, default='total_loss')
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--load_pretrained', type=str, default=None)
parser.add_argument('--num_folds', type=int, default=1)
parser.add_argument('--num_fold', type=int, default=0)
parser.add_argument('--fixed_seed', type=str, default='True')


#loss parameters
parser.add_argument('--loss_function', type=str, default='emo_loss')
parser.add_argument('--loss_beta', type=float, default=1.)
parser.add_argument('--emo_loss_holes', type=int, default=None)  #emo loss is deactivated every x epochs
parser.add_argument('--emo_loss_warmup_epochs', type=int, default=None)  #warmup ramp length


#model parameters
parser.add_argument('--model_name', type=str, default='r2he')
parser.add_argument('--model_quat', type=str, default='True')
parser.add_argument('--model_classifier_quat', type=str, default='True')
parser.add_argument('--model_in_channels', type=int, default=1)
parser.add_argument('--model_flattened_dim', type=int, default=32768)
parser.add_argument('--model_latent_dim', type=int, default=1000)
parser.add_argument('--model_verbose', type=str, default='False')
parser.add_argument('--model_architecture', type=str, default='VGG16')
parser.add_argument('--model_classifier_dropout', type=float, default=0.5)

#grid search parameters
#SPECIFY ONLY IF PERFORMING A GRID SEARCH WITH exp_instance.py SCRIPT
parser.add_argument('--script', type=str, default='training_autoencoder.py')
parser.add_argument('--comment_1', type=str, default='none')
parser.add_argument('--comment_2', type=str, default='none')
parser.add_argument('--experiment_description', type=str, default='none')
parser.add_argument('--dataset', type=str, default='none')
parser.add_argument('--num_experiment', type=int, default=0)
#        "load_pretrained": "'../new_experiments/experiment_9_5samples.txt/models/model_xval_iemocap_exp9_5samples.txt_run1_fold0'"

#eval string args
args = parser.parse_args()
#output filenames

args.fast_test = eval(args.fast_test)
args.normalize_predictors = eval(args.normalize_predictors)
args.use_cuda = eval(args.use_cuda)
args.early_stopping = eval(args.early_stopping)
args.fixed_seed = eval(args.fixed_seed)
args.model_verbose = eval(args.model_verbose)
args.model_quat = eval(args.model_quat)
args.model_classifier_quat = eval(args.model_classifier_quat)


if args.use_cuda:
    device = 'cuda:' + str(args.gpu_id)
else:
    device = 'cpu'

#load data loaders
tr_data, val_data, test_data = uf.load_datasets(args)

#load model
print ('\nMoving model to device')
if args.model_name == 'r2he':
    model = locals()[args.model_name](batch_normalization=args.model_batchnorm)
if args.model_name == 'simple_autoencoder':
    print ('AAAAAFJFJFJFJFJFJFJFJFJFJFJFJ')
    model = locals()[args.model_name]()


model = model.to(device)

#compute number of parameters
model_params = sum([np.prod(p.size()) for p in model.parameters()])
print ('Total paramters: ' + str(model_params))

#load pretrained model if desired


#args.load_pretrained = '../new_experiments/experiment_9_5samples.txt/models/model_xval_iemocap_exp9_5samples.txt_run1_fold0'
if args.load_pretrained is not None:
    print ('Loading pretrained model: ' + args.load_pretrained)
    pretrained_dict = torch.load(args.load_pretrained)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    #model_dict.update(pretrained_dict)

    model.load_state_dict(pretrained_dict, strict=False)  #load best model

    #with torch.no_grad():
    #    model.fc1.weight.copy_(state_dict['fc1.weight'])
    #    model.fc1.bias.copy_(state_dict['fc1.bias'])


#define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.regularization_lambda)

#loss_function = nn.BCELoss()
loss_function = locals()[args.loss_function]

#init history
train_loss_hist = []
val_loss_hist = []

def evaluate(model, device, loss_function, dataloader, emo_weight):
    #compute loss without backprop
    model.eval()
    temp_loss = []
    with tqdm(total=len(dataloader)) as pbar, torch.no_grad():
        #validation data
        for i, (sounds, truth) in enumerate(dataloader):
            sounds = sounds.to(device)
            truth = truth.to(device)

            recon, pred = model(sounds)
            #recon = torch.unsqueeze(torch.sum(recon, axis=1), dim=1) / 4.

            #loss = loss_function(recon, sounds)
            loss = loss_function(recon, sounds, truth, pred, emo_weight)
            #loss = loss['total'].cpu().numpy()

            #temp_loss.append({'total':loss, 'emo':0, 'recon':0,
            #              'valence':0, 'arousal':0, 'dominance':0})
            temp_loss.append({'total':loss['total'].cpu().numpy(),
                                       'emo': loss['emo'],
                                       'recon':loss['recon'],
                                       'valence':loss['valence'], 'arousal':0, 'dominance':0})
            pbar.update(1)
    return temp_loss

def mean_batch_loss(batch_loss):
    #compute mean of each loss item
    d = {'total':[], 'emo':[], 'recon':[],
                  'valence':[], 'arousal':[], 'dominance':[]}
    for i in batch_loss:
        for j in i:
            name = j
            value = i[j]
            d[name].append(value)
    for i in d:
        d[i] = np.mean(d[i])
    return d

#training loop
for epoch in range(args.num_epochs):
    epoch_start = time.perf_counter()
    print ('\n')
    print ('Epoch: [' + str(epoch+1) + '/' + str(args.num_epochs) + '] ')
    train_batch_losses = []
    model.train()

    #emotional loss warm up and holes
    emo_weight = args.loss_beta
    #warm up
    if args.emo_loss_warmup_epochs is not None:
        if epoch < args.emo_loss_warmup_epochs:
            ramp = np.arange(args.emo_loss_warmup_epochs) / args.emo_loss_warmup_epochs
            w = ramp[epoch]
            emo_weight = emo_weight * w
    #holesramp
    if args.emo_loss_holes is not None:
        if epoch % args.emo_loss_holes == 0:
            emo_weight = 0

    #train data
    with tqdm(total=len(tr_data)) as pbar:
        for i, (sounds, truth) in enumerate(tr_data):
            optimizer.zero_grad()
            sounds = sounds.to(device)
            truth = truth.to(device)

            recon, pred = model(sounds)

            #recon = torch.unsqueeze(torch.sum(recon, axis=1), dim=1) / 4.
            #recon = torch.unsqueeze(torch.sqrt(torch.sum(recon**2, axis=1)), dim=1)
            #loss = loss_function(recon, sounds)
            loss = loss_function(recon, sounds, truth, pred, emo_weight)
            loss['total'].backward()
            optimizer.step()

            #loss = loss.detach().cpu().item()
            train_batch_losses.append({'total':loss['total'].detach().cpu().numpy(),
                                       'emo': loss['emo'],
                                       'recon':loss['recon'],
                                       'valence':loss['valence'], 'arousal':0, 'dominance':0})
            pbar.update(1)
            #del loss

    #validation data
    val_batch_losses = evaluate(model, device, loss_function, val_data, emo_weight)

    train_epoch_loss = mean_batch_loss(train_batch_losses)
    val_epoch_loss = mean_batch_loss(val_batch_losses)

    print ('\n EPOCH LOSSES:')
    print ('\n Training:')
    print (train_epoch_loss)
    print ('\n Validation:')
    print (val_epoch_loss)
    print ('Comments:')
    print (args.comment_1, args.comment_2)
    print ('EMO weight: ', emo_weight)

    train_loss_hist.append(train_epoch_loss)
    val_loss_hist.append(val_epoch_loss)

    #print ('\n  Train loss: ' + str(np.round(train_epoch_loss.item(), decimals=5)) + ' | Val loss: ' + str(np.round(val_epoch_loss.item(), decimals=5)))

    #compute epoch time
    epoch_time = float(time.perf_counter()) - float(epoch_start)
    print ('\n Epoch time: ' + str(np.round(float(epoch_time), decimals=1)) + ' seconds')

    #save best model (metrics = validation loss)
    if epoch == 0:
        torch.save(model.state_dict(), args.model_path)
        print ('\nModel saved')
        saved_epoch = epoch + 1
    else:
        if args.save_model_metric == 'total_loss':
            best_loss = min([i['total'] for i in val_loss_hist[:-1]])
            #best_loss = min(val_loss_hist['total'].item()[:-1])  #not looking at curr_loss
            curr_loss = val_loss_hist[-1]['total']
            if curr_loss < best_loss:
                torch.save(model.state_dict(), args.model_path)
                print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                saved_epoch = epoch + 1
        elif args.save_model_metric == 'epochs':
            if epoch % 100 == 0:
                torch.save(model.state_dict(), args.model_path)
                print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                saved_epoch = epoch + 1


        else:
            raise ValueError('Wrong metric selected')
    '''
    if args.num_experiment != 0:
        #print info on dataset, experiment and instance if performing a grid search
        utilstring = 'dataset: ' + str(args.dataset) + ', exp: ' + str(args.num_experiment) + ', run: ' + str(args.num_run) + ', fold: ' + str(args.num_fold)
        print ('')
        print (utilstring)
    '''

    if args.early_stopping and epoch >= args.patience+1:
        patience_vec = [i['total'] for i in val_loss_hist[-args.patience+1:]]
        #patience_vec = val_loss_hist[-args.patience+1:]
        best_l = np.argmin(patience_vec)
        if best_l == 0:
            print ('Training early-stopped')
            break


#COMPUTE METRICS WITH BEST SAVED MODEL
print ('\nComputing metrics with best saved model')

model.load_state_dict(torch.load(args.model_path), strict=False)  #load best model

train_batch_losses = evaluate(model, device, loss_function, tr_data, 1)
val_batch_losses = evaluate(model, device, loss_function, val_data, 1)
test_batch_losses = evaluate(model, device, loss_function, test_data, 1)

train_loss = mean_batch_loss(train_batch_losses)
val_loss = mean_batch_loss(val_batch_losses)
test_loss = mean_batch_loss(test_batch_losses)

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
temp_results['val_loss_hist'] = val_loss_hist
temp_results['parameters'] = vars(args)

np.save(args.results_path, temp_results)

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
