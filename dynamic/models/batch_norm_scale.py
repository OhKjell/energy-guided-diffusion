import torch

from torch import nn


class Bias3DLayer(nn.Module):
    def __init__(self, channels, initial=0, **kwargs):
        super(Bias3DLayer, self).__init__(**kwargs)

        self.bias = torch.nn.Parameter(
            torch.empty((1, channels, 1, 1, 1)).fill_(initial)
        )

    def forward(self, x):
        return x + self.bias


class Scale3DLayer(nn.Module):
    def __init__(self, channels, initial=1, **kwargs):
        super(Scale3DLayer, self).__init__(**kwargs)

        self.scale = torch.nn.Parameter(
            torch.empty((1, channels, 1, 1, 1)).fill_(initial)
        )

    def forward(self, x):
        return x * self.scale
