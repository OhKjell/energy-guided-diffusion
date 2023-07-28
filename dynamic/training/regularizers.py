import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def l1_regularization(model, l1=0):
    weights = model.conv1.weight
    bias = model.conv1.bias
    return torch.norm(weights, 1) + torch.norm(bias, 1)


def laplace1d():
    return np.array([-1, 4, -1]).astype(np.float32)[None, None, ...]


class Laplace1d(nn.Module):
    def __init__(self, padding):
        super(Laplace1d, self).__init__()
        filter = laplace1d()
        self.register_buffer("filter", torch.from_numpy(filter))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def forward(self, x):
        return F.conv1d(x, self.filter, bias=None, padding=self.padding_size)


class TimeLaplaceL23d(nn.Module):
    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace1d(padding=padding)

    def forward(self, x, avg=False):
        oc, ic, t = x.size()
        if avg:
            return torch.mean(self.laplace(x.view(oc * ic, 1, t)).pow(2)) / torch.mean(
                x.view(oc * ic, 1, t).pow(2)
            )
        else:
            return torch.sum(self.laplace(x.view(oc * ic, 1, t)).pow(2)) / torch.sum(
                x.view(oc * ic, 1, t).pow(2)
            )
