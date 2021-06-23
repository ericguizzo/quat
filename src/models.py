import torch
from torch import nn
import torch.nn.functional as F
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

class VGGNet(nn.Module):
    def __init__(self,
                architecture='VGG16',
                classifier_dropout=0.5,
                flatten_dim=32768,
                verbose=True,
                quat=False,
                num_classes = 5
                ):
        super(VGGNet, self).__init__()
        self.quat = quat
        if quat:
            self.in_channels = 4
        else:
            self.in_channels = 1
        self.verbose = verbose
        self.flatten_dim = flatten_dim
        self.last_dim = [i for i in VGG_types[architecture] if type(i) != str][-1]
        self.first_dim = [i for i in VGG_types[architecture] if type(i) != str][0]

        self.features = self.create_conv_layers(VGG_types[architecture])

        classifier_layers = [nn.Linear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)
                             ]
        classifier_layers_q = [QuaternionLinear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             QuaternionLinear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)
                             ]
        if quat:
            self.classifier = nn.Sequential(*classifier_layers_q)
        else:
            self.classifier = nn.Sequential(*classifier_layers)

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                if self.quat:
                    c = QuaternionConv(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1))
                else:
                    c = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1))
                layers += [c,
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.verbose:
            print ('input: ', x.shape)
        x = self.features(x)
        if self.verbose:
            print ('features: ', x.shape)
        x = torch.flatten(x, start_dim=1)
        if self.verbose:
            print('flatten: ', x.shape)
        x = self.classifier(x)
        if self.verbose:
            print('classification: ', x.shape)
        return x

class cazzo(nn.Module):
    def __init__(self,
                latent_dim=4096,
                in_channels=1,
                architecture='VGG16',
                classifier_dropout=0.5,
                flattened_dim=32768,
                verbose=False,
                quat=True,
                classifier_quat=True
                ):
        super(cazzo, self).__init__()

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
                                                               padding=1),)
                                                #nn.Sigmoid())
        else:
            self.decoder_output = nn.Sequential(nn.Conv2d(self.first_dim,
                                                               out_channels=1,
                                                               kernel_size=3,
                                                               stride=1,
                                                               padding=1),)
                                                #nn.Sigmoid())

        classifier_layers = [nn.Linear(self.latent_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 1)
                             ]



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
        #valence = self.classifier_valence(x_v)
        #arousal = self.classifier_valence(x_a)
        #dominance = self.classifier_valence(x_d)

        #if self.verbose:
        #    print('output x: ', x.shape)
        #    print('output v: ', valence.shape)
        #    print('output a: ', arousal.shape)
        #    print('output d: ', dominance.shape)


        x = torch.tensor(x, requires_grad=True)
        return x

class r2he(nn.Module):
    def __init__(self, quat=True, classifier_quat=True,
                 conv_structure=[16,32,64,128,256], classifier_structure=[4096,4096],
                 batch_normalization=True, time_dim=512, freq_dim=128,
                 classifier_dropout=0.5, num_classes=5,
                 embeddings_dim=[64,64], verbose=False):
        super(r2he, self).__init__()
        self.conv_structure = conv_structure
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.div = 2**len(conv_structure)
        flatten_dim = int((time_dim / self.div) * (freq_dim / self.div) * conv_structure[-1])
        self.encoder = self.build_encoder(conv_structure, batch_normalization)
        self.decoder = self.build_decoder(conv_structure, batch_normalization, quat)
        self.classifier = self.build_classifier(classifier_structure, classifier_dropout,
                                                classifier_quat, flatten_dim, num_classes)
        self.embeddings_dim = embeddings_dim
        self.verbose = verbose

    def build_encoder(self, conv_structure, batch_normalization):
        layers = []
        in_channels = 1
        for i, x in enumerate(conv_structure):
            out_channels = x
            c = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1))

            layers += [c,
                       nn.ReLU(),
                       nn.MaxPool2d(2, 2)]
            if batch_normalization and i != len(conv_structure)-1:
                layers += [nn.BatchNorm2d(x)]
            in_channels = x
        return nn.Sequential(*layers)

    def build_decoder(self, conv_structure, batch_normalization, quat):
        conv_structure = conv_structure[::-1]
        if quat:
            conv_structure += [4]
        else:
            conv_structure += [1]
        layers = []
        in_channels = conv_structure[0]
        for i, x in enumerate(conv_structure[:-1]):
            in_channels = x
            out_channels = conv_structure[i+1]

            if quat:
                c = QuaternionTransposeConv(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              output_padding=1)
                b = QuaternionBatchNorm2d(out_channels)
            else:
                c = nn.ConvTranspose2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              output_padding=1)
                b = nn.BatchNorm2d(out_channels)

            layers += [c]
            if batch_normalization and i != len(conv_structure)-2:
                layers += [nn.ReLU(), b]
            else:
                layers += [nn.Sigmoid()]
            #in_channels = x
        return nn.Sequential(*layers)

    def build_classifier(self, classifier_structure, classifier_dropout,
                         classifier_quat, flatten_dim, num_classes):
        layers = []
        in_neurons = flatten_dim
        for i, x in enumerate(classifier_structure):
            out_neurons = x
            if classifier_quat:
                l = QuaternionLinear(in_neurons, out_neurons)
            else:
                l = nn.Linear(in_neurons, out_neurons)

            layers += [l,
                       nn.ReLU(),
                       nn.Dropout(p=classifier_dropout)]
            in_neurons = x
        layers += [nn.Linear(classifier_structure[1], num_classes)]
        return nn.Sequential(*layers)

    def get_embeddings(self, x):
        x = self.encoder(x)
        x = x.view(-1, 4, self.embeddings_dim[0], self.embeddings_dim[1])
        return x

    def forward(self, x):
        if self.verbose:
            print ('input: ', x.shape)

        x = self.encoder(x)
        if self.verbose:
            print ('encoder: ', x.shape)

        x = torch.flatten(x, start_dim=1)
        if self.verbose:
            print ('flattening: ', x.shape)

        pred = self.classifier(x)
        if self.verbose:
            print ('prediction: ', pred.shape)

        x = x.view(-1, self.conv_structure[-1],
                   self.time_dim//self.div,
                   self.freq_dim//self.div)
        if self.verbose:
            print ('reshaping: ', x.shape)

        x = self.decoder(x)
        if self.verbose:
            print ('decoder: ', x.shape)

        return x, pred


class simple_autoencoder(nn.Module):
    def __init__(self, quat=True, classifier_quat=True, hidden_size=2048 ,flatten_dim=16384,
                 classifier_dropout=0.5, num_classes=5, batch_normalization=False,
                 reduced_batch_normalization=False):
        super(simple_autoencoder, self).__init__()
        self.flatten_dim = flatten_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_normalization = batch_normalization
        self.reduced_batch_normalization = reduced_batch_normalization

        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4_bn = nn.BatchNorm2d(128)

        ## decoder layers ##
        if quat:
            self.t_conv1 = QuaternionTransposeConv(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv2 = QuaternionTransposeConv(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv3 = QuaternionTransposeConv(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv4 = QuaternionTransposeConv(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv5 = QuaternionTransposeConv(16, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.tconv1_bn = QuaternionBatchNorm2d(128)
            self.tconv2_bn = QuaternionBatchNorm2d(64)
            self.tconv3_bn = QuaternionBatchNorm2d(32)
            self.tconv4_bn = QuaternionBatchNorm2d(16)
        else:
            self.t_conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,output_padding=1)
            self.t_conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,output_padding=1)
            self.t_conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,output_padding=1)
            self.t_conv4 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,output_padding=1)
            self.t_conv5 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1,output_padding=1)
            self.tconv1_bn =  nn.BatchNorm2d(128)
            self.tconv2_bn =  nn.BatchNorm2d(64)
            self.tconv3_bn =  nn.BatchNorm2d(32)
            self.tconv4_bn = QuaternionBatchNorm2d(16)

        classifier_layers = [nn.Linear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)]

        classifier_layers_quat = [QuaternionLinear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             QuaternionLinear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)]

        if classifier_quat:
            self.classifier = nn.Sequential(*classifier_layers_quat)
        else:
            self.classifier = nn.Sequential(*classifier_layers)

        i_l = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
        for m in self.modules():
            if isinstance(m, i_l):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        if self.batch_normalization or reduced_batch_normalization:
            x = self.conv1_bn(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if self.batch_normalization or reduced_batch_normalization:
            x = self.conv2_bn(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        if self.batch_normalization:
            x = self.conv3_bn(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        if self.batch_normalization:
            x = self.conv4_bn(x)

        x = F.relu(self.conv5(x))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        return x

    def get_embeddings(self, x):
        x = self.encode(x)
        x = x.view(-1, 4, 64, 64)

        return x

    def decode(self, x):
        x = x.view(-1, 256, 16, 4)

        x = F.relu(self.t_conv1(x))
        if self.batch_normalization:
            x = self.tconv1_bn(x)

        x = F.relu(self.t_conv2(x))
        if self.batch_normalization:
            x = self.tconv2_bn(x)

        x = F.relu(self.t_conv3(x))
        if self.batch_normalization:
            x = self.tconv3_bn(x)

        x = F.relu(self.t_conv4(x))
        if self.batch_normalization or reduced_batch_normalization:
            x = self.tconv4_bn(x)

        x = torch.sigmoid(self.t_conv5(x))

        return x

    def forward(self, x):
        x = self.encode(x)
        pred = self.classifier(x)
        x = self.decode(x)

        return x, pred

'''

class simple_autoencoder(nn.Module):
    def __init__(self, quat=True, classifier_quat=True, hidden_size=2048 ,flatten_dim=16384,
                 classifier_dropout=0.5, num_classes=5):
        super(simple_autoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.flatten_dim = flatten_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        #self.conv6 = nn.Conv2d(256, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.tconv2_bn = QuaternionBatchNorm2d(16)
        #self.hidden = nn.Linear(flatten_dim, hidden_size*4)
        #self.decoder_input = nn.Linear(hidden_size*4, flatten_dim)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        if quat:
            #self.t_conv0 = QuaternionTransposeConv(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv1 = QuaternionTransposeConv(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv2 = QuaternionTransposeConv(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv3 = QuaternionTransposeConv(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv4 = QuaternionTransposeConv(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv5 = QuaternionTransposeConv(16, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            #self.t_conv0 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1,output_padding=1)
            self.t_conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,output_padding=1)
            self.t_conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,output_padding=1)
            self.t_conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,output_padding=1)
            self.t_conv4 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,output_padding=1)
            self.t_conv5 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1,output_padding=1)

        classifier_layers = [nn.Linear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)]
        classifier_layers_quat = [QuaternionLinear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             QuaternionLinear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)]

        #self.classifier_valence = nn.Sequential(*classifier_layers)
        #self.classifier_arousal = nn.Sequential(*classifier_layers)
        #self.classifier_dominance = nn.Sequential(*classifier_layers)

        if classifier_quat:
            self.classifier = nn.Sequential(*classifier_layers_quat)
        else:
            self.classifier = nn.Sequential(*classifier_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv2_bn(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        #x = F.relu(self.conv6(x))
        #x = self.pool(x)


        #print ('CAZZOOOOOOOOOO', x.shape)
        #hidden dim
        x = torch.flatten(x, start_dim=1)
        #x = torch.sigmoid(self.hidden(x))
        #print (x.shape)

        return x

    def get_embeddings(self, x):
        x = self.encode(x)
        x = x.view(-1, 4, 64, 64)
        return x

    def decode(self, x):
        #x = F.relu(self.decoder_input(x))

        x = x.view(-1, 256, 16, 4)
        #x1 = F.relu(self.t_conv0(x1))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))

        x = F.relu(self.t_conv4(x))
        x = self.tconv2_bn(x)
        x = torch.sigmoid(self.t_conv5(x))

        return x

    def forward(self, x):
        #a = self.get_embeddings(x)
        x = self.encode(x)
        pred = self.classifier(x)
        x = self.decode(x)

        return x, pred
'''
