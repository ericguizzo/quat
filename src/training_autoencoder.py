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
parser.add_argument('--predictors_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy')
parser.add_argument('--target_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_target.npy')
parser.add_argument('--train_perc', type=float, default=0.7)
parser.add_argument('--val_perc', type=float, default=0.2)
parser.add_argument('--test_perc', type=float, default=0.1)
parser.add_argument('--normalize_predictors', type=str, default='True')
parser.add_argument('--time_dim', type=int, default=512)
parser.add_argument('--freq_dim', type=int, default=128)
parser.add_argument('--fast_test', type=str, default='True')
#training parameters
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.00001)
parser.add_argument('--regularization_lambda', type=float, default=0.)
parser.add_argument('--early_stopping', type=str, default='True')
parser.add_argument('--save_model_metric', type=str, default='total_loss')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--load_pretrained', type=str, default=None)
parser.add_argument('--num_folds', type=int, default=1)
parser.add_argument('--num_fold', type=int, default=0)
parser.add_argument('--fixed_seed', type=str, default='True')
#loss parameters
parser.add_argument('--loss_function', type=str, default='emo_loss')
parser.add_argument('--loss_beta', type=float, default=1.)
#model parameters
parser.add_argument('--model_name', type=str, default='r2he')
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

#eval string args
args = parser.parse_args()
#output filenames

args.fast_test = eval(args.fast_test)
args.normalize_predictors = eval(args.normalize_predictors)
args.use_cuda = eval(args.use_cuda)
args.early_stopping = eval(args.early_stopping)
args.fixed_seed = eval(args.fixed_seed)
args.model_verbose = eval(args.model_verbose)

if args.use_cuda:
    device = 'cuda:' + str(args.gpu_id)
else:
    device = 'cpu'

print ('\n Loading dataset')
loading_start = float(time.perf_counter())

#PREDICTORS_LOAD = os.path.join(args.dataset_path, 'iemocap_randsplit_spectrum_fast_predictors.npy')
#TARGET_LOAD = os.path.join(args.dataset_path, 'iemocap_randsplit_spectrum_fast_target.npy')
PREDICTORS_LOAD = args.predictors_path
TARGET_LOAD = args.target_path

dummy = np.load(TARGET_LOAD,allow_pickle=True)
dummy = dummy.item()
#create list of datapoints for current fold
foldable_list = list(dummy.keys())
fold_actors_list = uf.folds_generator(args.num_folds, foldable_list, [args.train_perc, args.val_perc, args.test_perc])
train_list = fold_actors_list[args.num_fold]['train']
val_list = fold_actors_list[args.num_fold]['val']
test_list = fold_actors_list[args.num_fold]['test']
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
if args.fixed_seed:
    seed = 1
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if args.fast_test:
    print ('FAST TEST: using unly 100 datapoints ')
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
    tr_max = np.max(training_predictors)
    #tr_max = 128
    training_predictors = np.divide(training_predictors, tr_max)
    validation_predictors = np.divide(validation_predictors, tr_max)
    test_predictors = np.divide(test_predictors, tr_max)

print ("Predictors range: ", np.min(training_predictors), np.max(training_predictors))

#reshaping for cnn
training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1],training_predictors.shape[2])
validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1], validation_predictors.shape[2])
test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1], test_predictors.shape[2])

#cut/pad dims
training_predictors = uf.pad_tensor_dims(training_predictors, args.time_dim, args.freq_dim)
validation_predictors = uf.pad_tensor_dims(validation_predictors, args.time_dim, args.freq_dim)
test_predictors = uf.pad_tensor_dims(test_predictors, args.time_dim, args.freq_dim)

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
print ('\nMoving model to device')
if args.model_name == 'r2he':
    model = locals()[args.model_name](latent_dim=args.model_latent_dim,
                                      in_channels=args.model_in_channels,
                                      architecture=args.model_architecture,
                                      classifier_dropout=args.model_classifier_dropout,
                                      flattened_dim=args.model_flattened_dim,
                                      verbose=args.model_verbose)

model = model.to(device)

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

def evaluate(model, device, loss_function, dataloader):
    #compute loss without backprop
    model.eval()
    temp_loss = []
    with tqdm(total=len(dataloader)) as pbar, torch.no_grad():
        #validation data
        for i, (sounds, truth) in enumerate(val_data):
            sounds = sounds.to(device)
            truth = truth.to(device)

            recon, v, a, d = model(sounds)
            loss = loss_function(sounds, recon, truth, v, a, d, args.loss_beta)
            loss['total'] = loss['total'].cpu().numpy()

            temp_loss.append(loss)
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
    #train data
    with tqdm(total=len(tr_data)) as pbar:
        for i, (sounds, truth) in enumerate(tr_data):
            optimizer.zero_grad()
            sounds = sounds.to(device)
            truth = truth.to(device)

            recon, v, a, d = model(sounds)
            loss = loss_function(sounds, recon, truth, v, a, d, args.loss_beta)
            loss['total'].backward(retain_graph=True)
            optimizer.step()

            loss['total'] = loss['total'].detach().cpu().item()
            train_batch_losses.append(loss)
            pbar.update(1)
            #del loss

    #validation data
    val_batch_losses = evaluate(model, device, loss_function, val_data)

    train_epoch_loss = mean_batch_loss(train_batch_losses)
    val_epoch_loss = mean_batch_loss(val_batch_losses)

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

train_batch_losses = evaluate(model, device, loss_function, tr_data)
val_batch_losses = evaluate(model, device, loss_function, val_data)
test_batch_losses = evaluate(model, device, loss_function, test_data)

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
temp_results['val_loss_hist'] = train_loss_hist
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
