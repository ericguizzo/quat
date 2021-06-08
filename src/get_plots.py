import torch
from torch import nn
import librosa
import soundfile
import matplotlib.pyplot as plt
import numpy as np
from models import *
import argparse
import os
import tqdm
import utility_functions as uf
parser = argparse.ArgumentParser()

parser.add_argument('--datapoints_list', type=str, default='[1,2,3,4,5]')
parser.add_argument('--output_path', type=str, default='../properties/NEW_experiments')
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--gpu_id', type=int, default=1)
parser.add_argument('--fixed_seed', type=str, default='True')

#dataset parameters
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--use_set', type=str, default='test')
parser.add_argument('--predictors_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy')
parser.add_argument('--target_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_target.npy')
parser.add_argument('--train_perc', type=float, default=0.7)
parser.add_argument('--val_perc', type=float, default=0.2)
parser.add_argument('--test_perc', type=float, default=0.1)
parser.add_argument('--normalize_predictors', type=str, default='True')
parser.add_argument('--time_dim', type=int, default=512)
parser.add_argument('--freq_dim', type=int, default=128)
parser.add_argument('--fast_test', type=str, default='True')
parser.add_argument('--num_folds', type=int, default=1)
parser.add_argument('--num_fold', type=int, default=0)
parser.add_argument('--sample_rate', type=int, default=16000)

#model parameters
parser.add_argument('--model_path', type=str, default='../new_experiments/experiment_1_beta0.txt/models/model_xval_iemocap_exp1_beta0.txt_run1_fold0')
parser.add_argument('--model_name', type=str, default='r2he')
parser.add_argument('--model_in_channels', type=int, default=1)
parser.add_argument('--model_flattened_dim', type=int, default=32768)
parser.add_argument('--model_latent_dim', type=int, default=1000)
parser.add_argument('--model_verbose', type=str, default='False')
parser.add_argument('--model_architecture', type=str, default='VGG16')
parser.add_argument('--model_classifier_dropout', type=float, default=0.5)

args = parser.parse_args()

args.use_cuda = eval(args.use_cuda)
args.datapoints_list = eval(args.datapoints_list)

#def gen_plot(o, r, v, a, d, sound_id, curr_path, format='png'):
def gen_plot(pred, truth, args):
    pred = pred.cpu().numpy()
    y, v,a,d = pred
    print ('AAAIFIEOEJFN', pred.shape)
    '''
    #print ('max: ', np.max(o), np.max(r),np.max(v),np.max(a),np.max(d))
    #print ('mean: ', np.mean(o), np.mean(r),np.mean(v),np.mean(a),np.mean(d))
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
    '''


if __name__ == '__main__':
    if args.use_cuda:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    #load data loaders
    tr_data, val_data, test_data = uf.load_datasets(args)

    #load model
    print ('\nMoving model to device')
    if args.model_name == 'r2he':
        model = locals()[args.model_name](latent_dim=args.model_latent_dim,
                                          in_channels=args.model_in_channels,
                                          architecture=args.model_architecture,
                                          classifier_dropout=args.model_classifier_dropout,
                                          flattened_dim=args.model_flattened_dim,
                                          verbose=args.model_verbose)
    else:
        raise ValueError('Invalid model name')

    model = model.to(device)

    #load pretrained model if desired
    model.load_state_dict(torch.load(args.model_path), strict=False)  #load best model

    #iterate batches
    model.eval()
    with tqdm(total=len(tr_data)) as pbar:
        for i, (sounds, truth) in enumerate(tr_data), torch.no_grad():
            if i in args.datapoints_list:
                print ('AJAJAJAJAJ', sounds.shape, truth.shape)
                #x = x.to(device)



        '''
        print ('shapes')
        curr_path = os.path.join(args.output_path, str(i))
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        print ('    Processing: ', str(i))

        gen_plot(pred, truth)
        '''
