import torch


class PassiveRegularizer:
    def __init__(self, device):
        self.device = device

    def __call__(self, input, iteration):
        return torch.tensor([0]).to(self.device)
