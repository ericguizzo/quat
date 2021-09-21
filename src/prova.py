import torch
from torch import nn
import torch.nn.functional as F
from quaternion_layers import (QuaternionConv, QuaternionLinear,
                               QuaternionTransposeConv)
from qbn import QuaternionBatchNorm2d
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from models import *

a = simple_autoencoder_2(quat=True, classifier_quat=True)
#a = simple_autoencoder(quat=False)

x = torch.rand(1,1,512,128)
x, pred = a(x)
print (x.shape)
