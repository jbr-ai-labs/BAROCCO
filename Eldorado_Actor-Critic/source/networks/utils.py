import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


def checkTile(ent, idx, targets):
    targets = [t for t in targets if t.entID != ent.entID]
    if 0 < idx and (len(targets) > 0):
        return targets[0], 1
    elif len(targets) > 0:
        return ent, 0
    else:
        return ent, None


def classify(logits):
    if len(logits.shape) == 1:
        logits = logits.view(1, -1)
    distribution = Categorical(F.softmax(logits, dim=1))
    atn = distribution.sample()
    return atn


class ConstDiscrete(nn.Module):
    def __init__(self, config, h, nattn):
        super().__init__()
        self.fc1 = nn.Linear(h, nattn)
        self.config = config

    def forward(self, stim):
        x = self.fc1(stim)
        xIdx = classify(x)
        return x, xIdx


class ValNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.HIDDEN, 1)
        self.envNet = Env(config, in_dim=config.ENT_DIM_GLOBAL)

    def forward(self, state):
        stim = self.envNet(state)
        x = self.fc(stim)
        x = x.view(-1, 1)
        return x


class QNet(nn.Module):
    def __init__(self, config, action_size):
        super().__init__()
        self.fc = torch.nn.Linear(config.HIDDEN, action_size)
        self.envNet = Env(config, in_dim=config.ENT_DIM_GLOBAL)

    def forward(self, state):
        stim = self.envNet(state)
        x = self.fc(stim)
        return x


class Env(nn.Module):
    def __init__(self, config, in_dim=None, out_dim=None, hidden_dim=None, last_fn=True):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else config.HIDDEN
        in_dim = in_dim if in_dim is not None else config.ENT_DIM
        out_dim = out_dim if out_dim is not None else h
        self.config = config
        self.last_fn = last_fn
        self.stateNet = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, out_dim))

    def forward(self, state):
        x = self.stateNet(state)
        if self.last_fn:
            x = F.relu(x)
        return x
