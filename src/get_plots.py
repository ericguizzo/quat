import torch
from torch import nn
import librosa
import soundfile
import matplotlib.pyplot as plt
import numpy as np
from models import *
import argparse
import os
import utility_functions as uf

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../beta_exp/experiment_12_simple_autoencoder.txt/models/model_xval_iemocap_exp12_simple_autoencoder.txt_run1_fold0')
parser.add_argument('--model_name', type=str, default='rh2e')
parser.add_argument('--predictors_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy')
parser.add_argument('--target_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_target.npy')
parser.add_argument('--datapoints_list', type=str, default='[1,2,3,4,5]')
parser.add_argument('--output_path', type=str, default='../properties/NEW_experiments')
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--time_dim', type=int, default=512)
parser.add_argument('--freq_dim', type=int, default=128)
parser.add_argument('--use_set', type=str, default='training')
args = parser.parse_args()

args.use_cuda = eval(args.use_cuda)

args.datapoints_list = eval(args.datapoints_list)
def gen_plot(o, r, v, a, d, sound_id, curr_path, format='png'):
    print ('max: ', np.max(o), np.max(r),np.max(v),np.max(a),np.max(d))
    print ('mean: ', np.mean(o), np.mean(r),np.mean(v),np.mean(a),np.mean(d))
    exponent = 0.5/3
    r = (np.flip(r.T,-1)/np.max(r))**exponent
    v = (np.flip(v.T,-1)/np.max(v))**exponent
    a = (np.flip(a.T,-1)/np.max(a))**exponent
    d = (np.flip(d.T,-1)/np.max(d))**exponent

    plt.figure(1)
    plt.suptitle('AUTOENCODER OUTPUT MATRICES')
    plt.subplot(231)
    plt.title('Real')
    plt.pcolormesh(r)
    plt.subplot(232)
    plt.title('Valence')
    plt.pcolormesh(v)
    plt.subplot(233)
    plt.title('Arousal')
    plt.pcolormesh(a)
    plt.subplot(234)
    plt.title('Dominance')
    plt.pcolormesh(d)
    plt.subplot(235)
    plt.title('Original')
    plt.pcolormesh(np.flip(o.T,-1)/np.max(o))

    plt.tight_layout( rect=[0, 0.0, 0.95, 0.95])

    name = str(sound_id) + '_plot.' + format
    fig_name = os.path.join(curr_path, name)
    plt.savefig(fig_name, format = format, dpi=300)
    #plt.show()


if __name__ == '__main__':
    print ('Loading dataset')
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    PREDICTORS_LOAD = args.predictors_path
    TARGET_LOAD = args.target_path

    dummy = np.load(TARGET_LOAD,allow_pickle=True)
    dummy = dummy.item()
    #create list of datapoints for current fold
    foldable_list = list(dummy.keys())
    fold_actors_list = uf.folds_generator(1, foldable_list, [0.7, 0.2, 0.1])
    train_list = fold_actors_list[0]['train']
    val_list = fold_actors_list[0]['val']
    test_list = fold_actors_list[0]['test']
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

#load pretrained model if desired
model.load_state_dict(torch.load(args.model_path), strict=False)  #load best model


    for i in args.datapoints_list:

        #get autoencoder's outputs
        x = data[i].unsqueeze(0)
        with torch.no_grad():
            y = x.to(device)
            y, v,a,d = model(y)
            y = y.cpu().numpy()
            v = v.cpu().numpy()
            a = a.cpu().numpy()
            d = d.cpu().numpy()
        print ('aaaaaaaaaaaaaaaa',np.max(y), np.min(y))
        v = v.squeeze()
        a = a.squeeze()
        d = d.squeeze()
        p = {'truth': target[i].squeeze(),
            'prediction': [v,a,d]}
        print (p)

        original = x.squeeze().numpy()
        if y.shape[1] == 1:
            real = y.copy().squeeze()
            valence = y.copy().squeeze()
            arousal = y.copy().squeeze()
            dominance = y.copy().squeeze()
        else:
            real = y[:,0,:,:].squeeze()
            valence = y[:,1,:,:].squeeze()
            arousal = y[:,2,:,:].squeeze()
            dominance = y[:,3,:,:].squeeze()

        print ('shapes')
        curr_path = os.path.join(args.output_path, str(i))
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        print ('    Processing: ', str(i))

        gen_plot(original,real,valence,arousal,dominance,i, curr_path)
