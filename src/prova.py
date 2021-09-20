import torch
from torch import nn
import torch.nn.functional as F
from quaternion_layers import (QuaternionConv, QuaternionLinear,
                               QuaternionTransposeConv)
from qbn import QuaternionBatchNorm2d
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from models import *

#a = dual_simple_autoencoder()
a = simple_autoencoder_2(quat=False)

x = torch.rand(1,1,512,256)
a(x)
