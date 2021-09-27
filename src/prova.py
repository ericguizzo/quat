import torch
from torch import nn
import torch.nn.functional as F

a = torch.tensor([[0.4243],
        [0.3788],
        [0.5170],
        [0.5040],
        [0.4850],
        [0.4560],
        [0.4811],
        [0.6360],
        [0.3033],
        [0.3625],
        [0.3628],
        [0.4103],
        [0.5194],
        [0.4134],
        [0.2764],
        [0.5408],
        [0.4633],
        [0.4931],
        [0.2561],
        [0.4481]])


b = torch.tensor([1., 1., 1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0.,
        0., 0.])

print ("COGLIONE: ", torch.sum(torch.round(a.squeeze()) == b) / a.shape[0])
