import torch
from torch import nn
import torch.nn.functional as F
from quaternion_layers import (QuaternionConv, QuaternionLinear,
                               QuaternionTransposeConv)
from qbn import QuaternionBatchNorm2d
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from models import *

a = simple_autoencoder_2_vad(quat=True)

x = torch.rand(1,1,512,128)
x, p, v, a, d = a(x)
print (x.shape)
print(p, v, a, d)
