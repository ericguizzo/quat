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
parser.add_argument('--model_path', type=str, default='./beta_exp/experiment_2_beta.txt/models/model_xval_iemocap_exp2_beta.txt_run1_fold0')
parser.add_argument('--predictors_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy')
parser.add_argument('--target_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_target.npy')
parser.add_argument('--datapoints_list', type=str, default='[1,2,3,4,5]')
parser.add_argument('--output_path', type=str, default='../properties/emo_ae_1')
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--time_dim', type=int, default=128)
parser.add_argument('--freq_dim', type=int, default=512)
parser.add_argument('--use_set', type=str, default='test')
args = parser.parse_args()

args.datapoints_list = eval(args.datapoints_list)

def gen_plot(r, v, a, d, sound_id, curr_path, format='png'):
    plt.figure(1)
    plt.suptitle('AUTOENCODER OUTPUT MATRICES')
    plt.subplot(221)
    plt.title('Real')
    plt.pcolormesh(r)
    plt.subplot(222)
    plt.title('Valence')
    plt.pcolormesh(v)
    plt.subplot(223)
    plt.title('Arousal')
    plt.pcolormesh(a)
    plt.subplot(224)
    plt.title('Dominance')
    plt.pcolormesh(d)
    plt.tight_layout( rect=[0, 0.0, 0.95, 0.95])

    name = str(sound_id) + '_plot.' + format
    fig_name = os.path.join(curr_path, name)
    plt.savefig(fig_name, format = format, dpi=300)
    plt.show()

def gen_sounds(r, v, a, d, sound_id,
               curr_path, sr=args.sample_rate, n_iter=100):
    pad = np.zeros(shape=[512,128])
    pad [:,:128] = r
    r = pad
    pad = np.zeros(shape=[512,128])
    pad [:,:128] = v
    v = pad
    pad = np.zeros(shape=[512,128])
    pad [:,:128] = a
    a = pad
    pad = np.zeros(shape=[512,128])
    pad [:,:128] = d
    d = pad

    real_wave = librosa.griffinlim(r, n_iter=n_iter)
    valence_wave = librosa.griffinlim(v, n_iter=n_iter)
    arousal_wave = librosa.griffinlim(a, n_iter=n_iter)
    dominance_wave = librosa.griffinlim(d, n_iter=n_iter)

    real_wave = (real_wave / np.max(real_wave)) * 0.9
    valence_wave = (valence_wave / np.max(valence_wave)) * 0.9
    arousal_wave = (arousal_wave / np.max(arousal_wave)) * 0.9
    dominance_wave = (dominance_wave / np.max(dominance_wave)) * 0.9

    name_r = name = str(sound_id) + '_real_wave.wav'
    name_v = name = str(sound_id) + '_valence_wave.wav'
    name_a = name = str(sound_id) + '_arousal_wave.wav'
    name_d = name = str(sound_id) + '_dominance_wave.wav'

    name_r = os.path.join(curr_path, name_r)
    name_v = os.path.join(curr_path, name_v)
    name_a = os.path.join(curr_path, name_a)
    name_d = os.path.join(curr_path, name_d)

    soundfile.write(name_r, real_wave, sr, 'PCM_16')
    soundfile.write(name_v, valence_wave, sr, 'PCM_16')
    soundfile.write(name_a, arousal_wave, sr, 'PCM_16')
    soundfile.write(name_d, dominance_wave, sr, 'PCM_16')




if __name__ == '__main__':
    print ('Loading dataset')
    PREDICTORS_LOAD = args.predictors_path
    TARGET_LOAD = args.target_path

    dummy = np.load(TARGET_LOAD,allow_pickle=True)
    dummy = dummy.item()
    #create list of datapoints for current fold
    foldable_list = list(dummy.keys())
    fold_actors_list = uf.folds_generator(1, foldable_list, [args.train_perc, args.val_perc, args.test_perc])
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

    if args.use_set == 'training':
        data = train_predictors
    elif args.use_set == 'validation':
        data = val_predictors
    elif args.use_set == 'test':
        data = test_predictors

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model = emo_ae()
    model.load_state_dict(torch.load(args.model_path), strict=False)  #load model

    for i in args.datapoints_list:

        #get autoencoder's outputs
        x = data[i]
        with torch.no_grad():
            x = model.autoencode(x).numpy()

        real = x[:,0,:,:].squeeze()
        valence = x[:,1,:,:].squeeze()
        arousal = x[:,2,:,:].squeeze()
        dominance = x[:,3,:,:].squeeze()

        curr_path = os.path.join(args.output_path, str(i))
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        print ('    Processing: ', str(i))
        gen_plot(real,valence,arousal,dominance,i, curr_path)
        gen_sounds(real,valence,arousal,dominance,i, curr_path)
