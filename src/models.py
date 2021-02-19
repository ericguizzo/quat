import torch
from torch import nn
from quaternion_layers import (QuaternionConv, QuaternionLinear,
                               QuaternionTransposeConv)



class emo_ae(nn.Module):
    def __init__(self,
                structure=[32, 64, 128, 256, 512],
                classifier_structure=[2000,1000,500,100],
                latent_dim=1000,
                verbose=True,
                quat=True):
        super(emo_ae, self).__init__()

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
                            nn.Conv2d(structure[-1], out_channels=4,
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


        layers = []
        in_chans = self.flattened_dim * 2
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
                nn.Linear(classifier_structure[-1], 1),
                nn.LeakyReLU()
            )
        self.final_layer_arousal = nn.Sequential(
                nn.Dropout(),
                nn.Linear(classifier_structure[-1], 1),
                nn.LeakyReLU()
            )
        self.final_layer_dominance = nn.Sequential(
                nn.Dropout(),
                nn.Linear(classifier_structure[-1], 1),
                nn.LeakyReLU()
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
        x_valence = torch.flatten(x[:,1,:,:], start_dim=1)
        x_arousal = torch.flatten(x[:,2,:,:], start_dim=1)
        x_dominance = torch.flatten(x[:,3,:,:], start_dim=1)

        x_valence = self.classifier_valence(x_valence)
        x_arousal = self.classifier_arousal(x_arousal)
        x_dominance = self.classifier_dominance(x_dominance)

        x_valence = torch.sigmoid(self.final_layer_dominance(x_valence))
        x_arousal = torch.sigmoid(self.final_layer_dominance(x_arousal))
        x_dominance = torch.sigmoid(self.final_layer_dominance(x_dominance))

        return x, x_valence, x_arousal, x_dominance


VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M",],
    "VGG19": [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M",]
    }


class emo_ae_vgg(nn.Module):
    def __init__(self,
                latent_dim=1000,
                in_channels=1,
                verbose=True,
                batchnorm=True,
                architecture='VGG16',
                classifier_dropout=0.5,
                quat=True
                ):
        super(emo_ae_vgg, self).__init__()

        self.in_channels = in_channels
        self.latent_dim =latent_dim
        self.verbose = verbose
        self.flattened_dim = 32768
        self.last_dim = [i for i in VGG_types[architecture] if type(i) != str][-1]
        self.first_dim = [i for i in VGG_types[architecture] if type(i) != str][0]

        self.encoder = self.create_conv_layers_encoder(VGG_types[architecture])
        self.latent =  QuaternionLinear(self.flattened_dim, latent_dim*4)
        self.decoder_input = QuaternionLinear(latent_dim*4, self.flattened_dim)
        self.decoder = self.create_conv_layers_decoder(VGG_types[architecture])
        self.decoder_output = nn.Sequential(QuaternionConv(self.first_dim,
                                                           out_channels=4,
                                                           kernel_size=3,
                                                           stride=1,
                                                           padding=1),
                                            nn.Sigmoid())


        classifier_layers = [nn.Linear(self.flattened_dim*2, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 1),
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
                    nn.BatchNorm2d(x),
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
                    nn.BatchNorm2d(batchnorm_dim),
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
                           nn.BatchNorm2d(batchnorm_dim),
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
        x = self.encoder(x)
        if self.verbose:
            print('encoder', x.shape)

        x = torch.flatten(x, start_dim=1)
        if self.verbose:
            print('flatten', x.shape)

        x = self.latent(x)
        if self.verbose:
            print ('latent', x.shape)

        x = self.decoder_input(x)
        x = x.view(-1, 512, 16, 4)

        if self.verbose:
            print('decoder_input', x.shape)

        x = self.decoder(x)
        if self.verbose:
            print('decoder', x.shape)

        x = self.decoder_output(x)
        if self.verbose:
            print('final', x.shape)

        #classifiers
        #x_valence = torch.flatten(x[:,1,:,:], start_dim=1)
        #x_arousal = torch.flatten(x[:,2,:,:], start_dim=1)
        #x_dominance = torch.flatten(x[:,3,:,:], start_dim=1)

        x_valence = torch.sigmoid(self.classifier_valence(torch.flatten(x[:,1,:,:], start_dim=1)))
        x_arousal = torch.sigmoid(self.classifier_arousal(torch.flatten(x[:,2,:,:], start_dim=1)))
        x_dominance = torch.sigmoid(self.classifier_dominance(torch.flatten(x[:,3,:,:], start_dim=1)))


        #emo_preds = torch.cat((x_valence,x_arousal,x_dominance),1)

        return x, x_valence, x_arousal, x_dominance
