import torch
from torch import nn
from quaternion_layers import (QuaternionConv, QuaternionLinear,
                               QuaternionTransposeConv)



class emo_vae(nn.Module):
    def __init__(self,
                structure=[32, 64, 128, 256, 512],
                classifier_structure=[2000,1000,500,100],
                latent_dim=20,
                verbose=True,
                quat=True):
        super(emo_vae, self).__init__()

        self.quat = quat
        self.latent_dim =latent_dim
        self.verbose = verbose
        self.flattened_dim = 32768
        #build encoder *real-valued
        conv_layers = []
        in_chans = 1
        for curr_chans in structure:
            conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, out_channels=curr_chans,
                                kernel_size=3, stride=2, padding=[1, 1]),
                    nn.LeakyReLU())
                )
            in_chans = curr_chans

        self.encoder = nn.Sequential(*conv_layers)

        #latent dimension layers
        self.latent_real = nn.Linear(self.flattened_dim, latent_dim)
        self.latent_q =  QuaternionLinear(self.flattened_dim, latent_dim*4)

        #decoder input layers
        self.decoder_input_real = nn.Linear(latent_dim, self.flattened_dim)
        self.decoder_input_q = QuaternionLinear(latent_dim*4, self.flattened_dim)
        structure.reverse()

        #build decoder *real valued
        conv_layers = []
        for i in range(len(structure) - 1):
            conv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(structure[i], structure[i + 1],
                                       kernel_size=3, stride=2, padding=1,
                                       output_padding=1),
                    nn.LeakyReLU())
            )
        self.decoder_real = nn.Sequential(*conv_layers)

        #build decoder *quaternion-valued
        conv_layers = []
        for i in range(len(structure) - 1):
            conv_layers.append(
                nn.Sequential(
                    QuaternionTransposeConv(structure[i], structure[i + 1],
                                       kernel_size=3, stride=2, padding=1,
                                       output_padding=1),
                    nn.LeakyReLU())
            )
        self.decoder_q = nn.Sequential(*conv_layers)

        #final layers
        self.final_layer_decoder_real = nn.Sequential(
                            nn.ConvTranspose2d(structure[-1],
                                               structure[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.LeakyReLU(),
                            nn.Conv2d(structure[-1], out_channels=1,
                                      kernel_size=3, padding=1),
                            nn.Sigmoid())

        self.final_layer_decoder_q = nn.Sequential(
                            QuaternionTransposeConv(structure[-1],
                                                    structure[-1],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    output_padding=1),
                            nn.LeakyReLU(),
                            QuaternionConv(structure[-1], out_channels=4,
                                      kernel_size=3, stride=1, padding=1),
                            nn.Sigmoid())

        self.classifier_valence = nn.sequential(
                                nn.Linear()

        layers = []
        for curr_chans in classifier_structure:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_chans, curr_chans),
                    nn.LeakyReLU())
                )
            in_chans = curr_chans

        self.classifier_valence = nn.Sequential(*layers)
        self.classifier_arousal = nn.Sequential(*layers)
        self.classifier_dominance = nn.Sequential(*layers)

        self.final_layer_valence = nn.Sequential(
                nn.Dropout(),
                nn.Linear(layers[-1], 1),
                nn.LeackyReLU()
            )
        self.final_layer_arousal = nn.Sequential(
                nn.Dropout(),
                nn.Linear(layers[-1], 1),
                nn.LeackyReLU()
            )
        self.final_layer_dominance = nn.Sequential(
                nn.Dropout(),
                nn.Linear(layers[-1], 1),
                nn.LeackyReLU()
            )

    def forward(self, x):
        #encoder
        x = self.encoder(x)
        if self.verbose:
            print('encoder', x.shape)
        x = torch.flatten(x, start_dim=1)
        if self.verbose:
            print('flatten', x.shape)

        #decoder
        if self.quat:
            x = self.latent_q(x)
            if self.verbose:
                print ('latent', x.shape)
            x = self.decoder_input_q(x)
            x = x.view(-1, 512, 16, 4)
            if self.verbose:
                print('decoder_input', x.shape)
            x = self.decoder_q(x)
            if self.verbose:
                print('decoder', x.shape)
            x = self.final_layer_decoder_q(x)
            if self.verbose:
                print('final', x.shape)

        if not self.quat:
            x = self.latent_real(x)
            if self.verbose:
                print ('latent', x.shape)
            x = self.decoder_input_real(x)
            x = x.view(-1, 512, 16, 4)
            if self.verbose:
                print('decoder_input', x.shape)
            x = self.decoder_real(x)
            if self.verbose:
                print('decoder', x.shape)

            x = self.final_layer_decoder_real(x)
            if self.verbose:
                print('final', x.shape)

        #classifiers
        x_valence = x[:,1,:,:]
        x_arousal = x[:,2,:,:]
        x_dominance = x[:,3,:,:]

        x_valence = self.classifier_valence(x)
        x_arousal = self.classifier_arousal(x)
        x_dominance = self.classifier_dominance(x)

        x_valence = self.final_layer_dominance(x_valence)
        x_arousal = self.final_layer_dominance(x_arousal)
        x_dominance = self.final_layer_dominance(x_dominance)

        return x, x_valence, x_arousal, x_dominance
