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
#parser.add_argument('--model_path', type=str, default='../beta_exp/experiment_2_beta.txt/models/model_xval_iemocap_exp2_beta.txt_run6_fold0')
parser.add_argument('--model_path', type=str, default='../beta_exp/experiment_3_beta_vgg.txt/results/results_iemocap_exp3_beta_vgg.txt_run2.npy')

parser.add_argument('--predictors_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy')
parser.add_argument('--target_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_target.npy')
parser.add_argument('--datapoints_list', type=str, default='[1,2,3,4,5]')
parser.add_argument('--output_path', type=str, default='../properties/emo_ae_1')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--time_dim', type=int, default=512)
parser.add_argument('--freq_dim', type=int, default=128)
parser.add_argument('--use_set', type=str, default='training')
args = parser.parse_args()

args.datapoints_list = eval(args.datapoints_list)
def gen_plot(o, r, v, a, d, sound_id, curr_path, format='png'):
    print ('max: ', np.max(o), np.max(r),np.max(v),np.max(a),np.max(d))
    print ('mean: ', np.mean(o), np.mean(r),np.mean(v),np.mean(a),np.mean(d))
    exponent = 0.01/3
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

def gen_sounds(o, r, v, a, d, sound_id,
               curr_path, sr=args.sample_rate, n_iter=100):
    pad = np.zeros(shape=[512,129])
    pad [:,:128] = o
    o = pad * pad.shape[-1]
    pad = np.zeros(shape=[512,129])
    pad [:,:128] = r
    r = pad * pad.shape[-1]
    pad = np.zeros(shape=[512,129])
    pad [:,:128] = v
    v = pad * pad.shape[-1]
    pad = np.zeros(shape=[512,129])
    pad [:,:128] = a
    a = pad * pad.shape[-1]
    pad = np.zeros(shape=[512,129])
    pad [:,:128] = d
    d = pad * pad.shape[-1]

    o = np.flip(o.T,-1)
    r = (np.flip(r.T,-1) - np.mean(r)) / np.std(r)
    v = np.flip(v.T,-1)
    a = np.flip(a.T,-1)
    d = np.flip(d.T,-1)

    original_wave = librosa.griffinlim(o, hop_length=128, n_iter=n_iter)
    real_wave = librosa.griffinlim(r, hop_length=128, n_iter=n_iter)
    valence_wave = librosa.griffinlim(v, hop_length=128, n_iter=n_iter)
    arousal_wave = librosa.griffinlim(a, hop_length=128, n_iter=n_iter)
    dominance_wave = librosa.griffinlim(d, hop_length=128, n_iter=n_iter)

    original_wave = (original_wave / np.max(original_wave)) * 0.9
    real_wave = (real_wave / np.max(real_wave)) * 0.9
    valence_wave = (valence_wave / np.max(valence_wave)) * 0.9
    arousal_wave = (arousal_wave / np.max(arousal_wave)) * 0.9
    dominance_wave = (dominance_wave / np.max(dominance_wave)) * 0.9

    name_o = name = str(sound_id) + '_original_wave.wav'
    name_r = name = str(sound_id) + '_real_wave.wav'
    name_v = name = str(sound_id) + '_valence_wave.wav'
    name_a = name = str(sound_id) + '_arousal_wave.wav'
    name_d = name = str(sound_id) + '_dominance_wave.wav'

    name_o = os.path.join(curr_path, name_o)
    name_r = os.path.join(curr_path, name_r)
    name_v = os.path.join(curr_path, name_v)
    name_a = os.path.join(curr_path, name_a)
    name_d = os.path.join(curr_path, name_d)

    soundfile.write(name_o, original_wave, sr, 'PCM_16')
    soundfile.write(name_r, real_wave, sr, 'PCM_16')
    soundfile.write(name_v, valence_wave, sr, 'PCM_16')
    soundfile.write(name_a, arousal_wave, sr, 'PCM_16')
    soundfile.write(name_d, dominance_wave, sr, 'PCM_16')




if __name__ == '__main__':
    print ('Loading dataset')
    seed=1
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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

    #normalize
    training_predictors = training_predictors / training_predictors.shape[-1]
    validation_predictors = validation_predictors / validation_predictors.shape[-1]
    test_predictors = test_predictors / test_predictors.shape[-1]

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
        target = training_target
    elif args.use_set == 'validation':
        data = val_predictors
        target = validation_target
    elif args.use_set == 'test':
        data = test_predictors
        target = test_target

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model = emo_ae_vgg()
    model.load_state_dict(torch.load(args.model_path), strict=False)  #load model
    model = model.to(device)
    for i in args.datapoints_list:

        #get autoencoder's outputs
        x = data[i].unsqueeze(0)
        with torch.no_grad():
            y = x.to(device)
            y, preds = model(y)
            y = y.cpu().numpy()
            preds = preds.cpu().numpy()

        preds = preds.squeeze()
        p = {'truth': target[i],
            'prediction': preds}
        print (p)
        original = x.squeeze().numpy()
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
        gen_sounds(original,real,valence,arousal,dominance,i, curr_path)
