import torch
from torch import nn
from quaternion_layers import (QuaternionConv, QuaternionLinear,
                               QuaternionTransposeConv)
from qbn import QuaternionBatchNorm2d

VGG_types = {
    "simple": [16,"M",32,"M",128,],
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M",],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M",],
    "VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M",],
    "VGG19": [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M",]
    }

class r2he(nn.Module):
    def __init__(self,
                latent_dim=4096,
                in_channels=1,
                architecture='VGG16',
                classifier_dropout=0.5,
                flattened_dim=32768,
                verbose=False,
                quat=True
                ):
        super(r2he, self).__init__()

        self.quat = quat
        self.in_channels = in_channels
        self.latent_dim =latent_dim
        self.verbose = verbose
        self.flattened_dim = flattened_dim
        self.last_dim = [i for i in VGG_types[architecture] if type(i) != str][-1]
        self.first_dim = [i for i in VGG_types[architecture] if type(i) != str][0]

        self.encoder = self.create_conv_layers_encoder(VGG_types[architecture])

        #self.latent =  QuaternionLinear(self.flattened_dim, latent_dim*4)
        self.latent_r =  nn.Linear(self.flattened_dim,  latent_dim)
        self.latent_v =  nn.Linear(self.flattened_dim,  latent_dim)
        self.latent_a =  nn.Linear(self.flattened_dim,  latent_dim)
        self.latent_d =  nn.Linear(self.flattened_dim,  latent_dim)

        if self.quat:
            self.decoder_input = QuaternionLinear(latent_dim*4, self.flattened_dim)
        else:
            self.decoder_input = nn.Linear(latent_dim*4, self.flattened_dim)

        self.decoder = self.create_conv_layers_decoder(VGG_types[architecture])

        if self.quat:
            self.decoder_output = nn.Sequential(QuaternionConv(self.first_dim,
                                                               out_channels=4,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1),
                                                nn.Sigmoid())
        else:
            self.decoder_output = nn.Sequential(nn.Conv2d(self.first_dim,
                                                               out_channels=1,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1),
                                                nn.Sigmoid())

        classifier_layers = [nn.Linear(self.latent_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 1),
                             nn.Sigmoid()
                             ]

        self.classifier_valence = nn.Sequential(*classifier_layers)
        self.classifier_arousal = nn.Sequential(*classifier_layers)
        self.classifier_dominance = nn.Sequential(*classifier_layers)


    def create_conv_layers_encoder(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    #nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

    def create_conv_layers_decoder(self, architecture):
        layers = []
        in_channels = self.last_dim  #last channels d
        out_channels = self.last_dim
        batchnorm_dim = self.last_dim
        architecture = architecture[::-1]  #reverse list

        if self.quat:
            for x in architecture:
                if type(x) == int:
                    out_channels = x
                    batchnorm_dim = x
                    layers += [
                        QuaternionTransposeConv(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                        #QuaternionBatchNorm2d(batchnorm_dim),
                        #nn.BatchNorm2d(batchnorm_dim),
                        nn.ReLU(),
                    ]
                    in_channels = x
                elif x == "M":
                    #out_channels = architecture[i+1]
                    layers += [QuaternionTransposeConv(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=(2, 2),
                                                  stride=(2, 2)
                                                  ),
                               #nn.BatchNorm2d(batchnorm_dim),
                               #QuaternionBatchNorm2d(batchnorm_dim),
                               nn.ReLU(),
                               ]
        else:
            for x in architecture:
                if type(x) == int:
                    out_channels = x
                    batchnorm_dim = x
                    layers += [
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                        #QuaternionBatchNorm2d(batchnorm_dim),
                        #nn.BatchNorm2d(batchnorm_dim),
                        nn.ReLU(),
                    ]
                    in_channels = x
                elif x == "M":
                    #out_channels = architecture[i+1]
                    layers += [nn.ConvTranspose2d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=(2, 2),
                                                  stride=(2, 2)
                                                  ),
                               #nn.BatchNorm2d(batchnorm_dim),
                               #QuaternionBatchNorm2d(batchnorm_dim),
                               nn.ReLU(),
                               ]


        return nn.Sequential(*layers)

    def autoencode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.latent(x)
        x = self.decoder_input(x)
        x = x.view(-1, 512, 16, 4)
        x = self.decoder(x)
        x = self.decoder_output(x)
        return x

    def forward(self, x):
        #encoder
        if self.verbose:
            print ('input: ', x.shape)

        x = self.encoder(x)
        encoder_output_shape = x.shape
        if self.verbose:
            print('encoder: ', x.shape)

        x = torch.flatten(x, start_dim=1)
        if self.verbose:
            print('flatten: ', x.shape)

        #latent VAD dimensions
        x_r = self.latent_r(x)
        if self.verbose:
            print ('latent_r: ', x_r.shape)

        x_v = self.latent_r(x)
        if self.verbose:
            print ('latent_v: ', x_v.shape)

        x_a = self.latent_r(x)
        if self.verbose:
            print ('latent_a: ', x_a.shape)

        x_d = self.latent_r(x)
        if self.verbose:
            print ('latent_a: ', x_d.shape)

        #decoder
        x = torch.cat((x_r, x_v, x_a, x_d), dim=-1)
        if self.verbose:
            print('latent cat: ', x.shape)
        x = self.decoder_input(x)
        if self.verbose:
            print('decoder_input: ', x.shape)
        x = x.view(-1, encoder_output_shape[-3], encoder_output_shape[-2], encoder_output_shape[-1])
        if self.verbose:
            print('decoder_input view: ', x.shape)

        x = self.decoder(x)
        if self.verbose:
            print('decoder: ', x.shape)

        x = self.decoder_output(x)
        if self.verbose:
            print('final: ', x.shape)

        #classifiers
        valence = self.classifier_valence(x_v)
        arousal = self.classifier_valence(x_a)
        dominance = self.classifier_valence(x_d)

        if self.verbose:
            print('output x: ', x.shape)
            print('output v: ', valence.shape)
            print('output a: ', arousal.shape)
            print('output d: ', dominance.shape)

        return x



class simple_autoencoder(nn.Module):
    def __init__(self,
                latent_dim=1000,
                time_dim = 128
                ):
        super(simple_autoencoder, self).__init__()
        self.time_dim = time_dim
        self.encoder = nn.Sequential(nn.Linear(time_dim*128, 4096),
                                     nn.LeakyReLU(),
                                     nn.Linear(4096, 2048),
                                     nn.LeakyReLU(),
                                     nn.Linear(2048, 1024),
                                     nn.LeakyReLU(),
                                     nn.Linear(1024, time_dim),
                                     nn.LeakyReLU(),
                                    )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, time_dim),
                                     nn.LeakyReLU(),
                                     nn.Linear(time_dim, 1024),
                                     nn.LeakyReLU(),
                                     nn.Linear(1024, 2048),
                                     nn.LeakyReLU(),
                                     nn.Linear(2048, 4096),
                                     nn.LeakyReLU(),
                                     nn.Linear(4096, time_dim*128)
                                    )

        self.latent = nn.Linear(time_dim, latent_dim)
'''

class r2he(nn.Module):
    def __init__(self,
                latent_dim=4096,
                in_channels=1,
                architecture='VGG16',
                classifier_dropout=0.5,
                flattened_dim=32768,
                verbose=False,
                quat=True
                ):
        super(r2he, self).__init__()

        self.quat = quat
        self.in_channels = in_channels
        self.latent_dim =latent_dim
        self.verbose = verbose
        self.flattened_dim = flattened_dim
        self.last_dim = [i for i in VGG_types[architecture] if type(i) != str][-1]
        self.first_dim = [i for i in VGG_types[architecture] if type(i) != str][0]

        self.encoder = self.create_conv_layers_encoder(VGG_types[architecture])

        #self.latent =  QuaternionLinear(self.flattened_dim, latent_dim*4)
        self.latent_r =  nn.Linear(self.flattened_dim,  latent_dim)
        self.latent_v =  nn.Linear(self.flattened_dim,  latent_dim)
        self.latent_a =  nn.Linear(self.flattened_dim,  latent_dim)
        self.latent_d =  nn.Linear(self.flattened_dim,  latent_dim)

        if self.quat:
            self.decoder_input = QuaternionLinear(latent_dim*4, self.flattened_dim)
        else:
            self.decoder_input = nn.Linear(latent_dim*4, self.flattened_dim)

        self.decoder = self.create_conv_layers_decoder(VGG_types[architecture])

        if self.quat:
            self.decoder_output = nn.Sequential(QuaternionConv(self.first_dim,
                                                               out_channels=4,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1),
                                                nn.Sigmoid())
        else:
            self.decoder_output = nn.Sequential(nn.Conv2d(self.first_dim,
                                                               out_channels=1,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1),
                                                nn.Sigmoid())

        classifier_layers = [nn.Linear(self.latent_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 1),
                             nn.Sigmoid()
                             ]

        self.classifier_valence = nn.Sequential(*classifier_layers)
        self.classifier_arousal = nn.Sequential(*classifier_layers)
        self.classifier_dominance = nn.Sequential(*classifier_layers)


    def create_conv_layers_encoder(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    #nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

    def create_conv_layers_decoder(self, architecture):
        layers = []
        in_channels = self.last_dim  #last channels d
        out_channels = self.last_dim
        batchnorm_dim = self.last_dim
        architecture = architecture[::-1]  #reverse list

        if self.quat:
            for x in architecture:
                if type(x) == int:
                    out_channels = x
                    batchnorm_dim = x
                    layers += [
                        QuaternionTransposeConv(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                        #QuaternionBatchNorm2d(batchnorm_dim),
                        #nn.BatchNorm2d(batchnorm_dim),
                        nn.ReLU(),
                    ]
                    in_channels = x
                elif x == "M":
                    #out_channels = architecture[i+1]
                    layers += [QuaternionTransposeConv(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=(2, 2),
                                                  stride=(2, 2)
                                                  ),
                               #nn.BatchNorm2d(batchnorm_dim),
                               #QuaternionBatchNorm2d(batchnorm_dim),
                               nn.ReLU(),
                               ]
        else:
            for x in architecture:
                if type(x) == int:
                    out_channels = x
                    batchnorm_dim = x
                    layers += [
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                        #QuaternionBatchNorm2d(batchnorm_dim),
                        #nn.BatchNorm2d(batchnorm_dim),
                        nn.ReLU(),
                    ]
                    in_channels = x
                elif x == "M":
                    #out_channels = architecture[i+1]
                    layers += [nn.ConvTranspose2d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=(2, 2),
                                                  stride=(2, 2)
                                                  ),
                               #nn.BatchNorm2d(batchnorm_dim),
                               #QuaternionBatchNorm2d(batchnorm_dim),
                               nn.ReLU(),
                               ]


        return nn.Sequential(*layers)

    def autoencode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.latent(x)
        x = self.decoder_input(x)
        x = x.view(-1, 512, 16, 4)
        x = self.decoder(x)
        x = self.decoder_output(x)
        return x

    def forward(self, x):
        #encoder
        if self.verbose:
            print ('input: ', x.shape)

        x = self.encoder(x)
        encoder_output_shape = x.shape
        if self.verbose:
            print('encoder: ', x.shape)

        x = torch.flatten(x, start_dim=1)
        if self.verbose:
            print('flatten: ', x.shape)

        #latent VAD dimensions
        x_r = self.latent_r(x)
        if self.verbose:
            print ('latent_r: ', x_r.shape)

        x_v = self.latent_r(x)
        if self.verbose:
            print ('latent_v: ', x_v.shape)

        x_a = self.latent_r(x)
        if self.verbose:
            print ('latent_a: ', x_a.shape)

        x_d = self.latent_r(x)
        if self.verbose:
            print ('latent_a: ', x_d.shape)

        #decoder
        x = torch.cat((x_r, x_v, x_a, x_d), dim=-1)
        if self.verbose:
            print('latent cat: ', x.shape)
        x = self.decoder_input(x)
        if self.verbose:
            print('decoder_input: ', x.shape)
        x = x.view(-1, encoder_output_shape[-3], encoder_output_shape[-2], encoder_output_shape[-1])
        if self.verbose:
            print('decoder_input view: ', x.shape)

        x = self.decoder(x)
        if self.verbose:
            print('decoder: ', x.shape)

        x = self.decoder_output(x)
        if self.verbose:
            print('final: ', x.shape)

        #classifiers
        valence = self.classifier_valence(x_v)
        arousal = self.classifier_valence(x_a)
        dominance = self.classifier_valence(x_d)

        if self.verbose:
            print('output x: ', x.shape)
            print('output v: ', valence.shape)
            print('output a: ', arousal.shape)
            print('output d: ', dominance.shape)

        return x, valence, arousal, dominance
'''


    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.latent(x)
        x = self.decoder(x)
        x = torch.sigmoid(x.view(-1, 1, self.time_dim, 128))

        #dummy = torch.tensor([0])
        return x
