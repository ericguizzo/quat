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
                num_classes = 4
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

    def get_embeddings(self, x):
        x = self.encode(x)
        x = x.view(-1, 4, self.flatten_dim//4)
        #print ('h', x.shape)
        return x, 'dummy'


    def forward(self, x):
        #a = self.get_embeddings(x)
        x = self.encode(x)
        pred = self.classifier(x)
        x = self.decode(x)

        return x, pred
