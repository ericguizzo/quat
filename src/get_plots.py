import torch
from torch import nn
import librosa
import soundfile
import matplotlib.pyplot as plt
import numpy as np
from models import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='experiments_quat')
parser.add_argument('--predictors_path', type=str, default='../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy')
parser.add_argument('--datapoints_list', type=str, default='[1]')
parser.add_argument('--output_path', type=str, default='../properties')
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--n_fft', type=int, default=129)

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
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    x = torch.rand(1,1,512,128)
    model = emo_ae()

    with torch.no_grad():
        x = model.autoencode(x).numpy()

    real = x[:,0,:,:].squeeze()
    valence = x[:,1,:,:].squeeze()
    arousal = x[:,2,:,:].squeeze()
    dominance = x[:,3,:,:].squeeze()

    for i in [1]:
        curr_path = os.path.join(args.output_path, str(i))
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
        print ('Processing: ', str(i))
        gen_plot(real,valence,arousal,dominance,i, curr_path)
        gen_sounds(real,valence,arousal,dominance,i, curr_path)
