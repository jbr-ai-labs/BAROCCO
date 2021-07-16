from torch import nn
from torch.nn import functional as F

from source.networks.noisy_layers import NoisyLinear
from source.networks.utils import ConstDiscrete


class ActionNet(nn.Module):
    def __init__(self, config, args, outDim=8, n_quant=1, add_features=0, noise=False):
        super().__init__()
        self.config, self.args, self.h = config, args, config.HIDDEN
        self.fc = NoisyLinear(self.h+add_features, self.h) if noise else nn.Linear(self.h+add_features, self.h)
        self.actionNet = ConstDiscrete(config, args, self.h, outDim, n_quant)

    def forward(self, s, eps=0, punish=None):
        x = F.relu(self.fc(s))
        outs, idx = self.actionNet(x, eps, punish)
        return outs, idx

    def reset_noise(self):
        self.fc.reset_noise()
