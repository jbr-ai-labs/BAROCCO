import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from source.networks.noisy_layers import NoisyLinear


def classify_Q(logits, eps=0.1, mode='boltzmann'):
    """
    https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf
    :param eps: epsilon for eps-greedy, temperature for boltzmann, dropout rate for bayes
    :param mode: eps-greedy, boltzmann, or bayes
    """
    if len(logits.shape) == 1:
        logits = logits.view(1, -1)
    if mode.lower().startswith('boltzman'):
        distribution = Categorical(1e-4 + F.softmax(logits/eps, dim=1))
        atn = distribution.sample()
    elif mode.lower().startswith('eps'):
        if np.random.random() > eps:
            atn = torch.argmax(logits).view(1, 1)
        else:
            atn = torch.randint(logits.shape[1], (1,)).view(1, 1)
    elif (mode.lower().startswith('bayes')) or (mode.lower() == 'noise') or (mode.lower().startswith('greed')):
        atn = torch.argmax(logits).view(1, 1)
    return atn


class ConstDiscrete(nn.Module):
    def __init__(self, config, args, h, n_attn, n_quant):
        super().__init__()
        self.fc1 = nn.Linear(h, n_attn * n_quant)
        self.n_attn = n_attn
        self.n_quant = n_quant
        self.args, self.config = args, config

    def forward(self, stim, eps, punish=None):
        x = self.fc1(stim).view(-1, self.n_attn, self.n_quant)
        if punish is None:
            xIdx = classify_Q(x.mean(2), eps, mode=self.config.EXPLORE_MODE)
        else:
            xIdx = classify_Q(x.mean(2) * (1 - self.config.PUNISHMENT) + punish, eps, mode=self.config.EXPLORE_MODE)
        return x, xIdx


class ValNet(nn.Module):
    def __init__(self, config, n_quant=1, add_features=0, noise=False):
        super().__init__()
        self.config = config
        self.h = config.HIDDEN
        self.fc = NoisyLinear(self.h+add_features, self.h) if noise else nn.Linear(self.h+add_features, self.h)
        self.valNet = nn.Linear(self.h, n_quant)
        self.n_quant = n_quant

    def forward(self, s):
        x = F.relu(self.fc(s))
        x = self.valNet(x).view(-1, 1, self.n_quant)
        return x

    def reset_noise(self):
        self.fc.reset_noise()


class EnvDummy(nn.Module):
    def __init__(self, config, noise, in_channels, out_channels, kernel_size, in_features, out_features, activation=True):
        super().__init__()
        self.config = config

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fc = NoisyLinear(in_features, out_features) if noise else nn.Linear(in_features, out_features)
        self.activation = nn.ReLU() if activation else lambda x: x

    def forward(self, env):
        x = F.relu(self.conv(env).view(env.shape[0], -1))
        x = self.activation(self.fc(x))
        return x

    def reset_noise(self):
        self.fc.reset_noise()


class Env(EnvDummy):
    def __init__(self, config, noise=False):
        super().__init__(config, noise, 3, 6, (3, 3), config.ENT_DIM, config.HIDDEN, True)


class EnvGlobal(EnvDummy):
    def __init__(self, config, out_features):
        super().__init__(config, False, 3, 6, (3, 3), config.ENT_DIM_GLOBAL, out_features, True)
