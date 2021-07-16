import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from source.networks.noisy_layers import NoisyLinear


def classify(logits):
    if len(logits.shape) == 1:
        logits = logits.view(1, -1)
    distribution = Categorical(1e-3 + F.softmax(logits, dim=1))
    atn = distribution.sample()
    return atn


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
    def __init__(self, args, config, n_quant):
        super().__init__()
        self.fc1 = nn.Linear(config.HIDDEN, config.N_OUTPUTS * n_quant)
        self.n_attn = config.N_OUTPUTS
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
    def __init__(self, config, n_quant=1, noise=False):
        super().__init__()
        self.config = config
        self.envNet = Env(config, noise)
        self.valNet = nn.Linear(config.HIDDEN, n_quant)
        self.n_quant = n_quant

    def forward(self, state):
        x = self.envNet(state)
        v = self.valNet(x).view(-1, 1, self.n_quant)
        return v

    def reset_noise(self):
        self.envNet.reset_noise()


class Env(nn.Module):
    def __init__(self, config, noise=False):
        super().__init__()
        h, entDim = config.HIDDEN, config.ENT_DIM
        self.fc = nn.Linear(entDim, h)
        self.fc2 = NoisyLinear(h, h) if noise else nn.Linear(h, h)

    def forward(self, state):
        x = F.relu(self.fc(state))
        x = F.relu(self.fc2(x))
        return x

    def reset_noise(self):
        self.fc2.reset_noise()


def checkTile(ent, idx, targets):
    targets = [t for t in targets if t.entID != ent.entID]
    if (idx > 0) and (len(targets) > 0):
        return targets[0], 1
    elif len(targets) > 0:
        return ent, 0
    else:
        return ent, None


def one_hot(labels, n_labels=10):
    one_hot = torch.zeros(labels.shape[0], n_labels).float()
    one_hot[torch.arange(one_hot.shape[0]), labels] = 1.
    return one_hot
