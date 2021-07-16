import torch
from torch import nn
from torch.nn import functional as F

from source.networks.utils import ConstDiscrete, Env


class ActionNet(nn.Module):
    def __init__(self, config, outDim=8, device='cpu', batch_size=1):
        super().__init__()
        self.config, self.h = config, config.HIDDEN
        self.outDim = outDim
        self.envNet = Env(config, config.LSTM, device=device, batch_size=batch_size)
        if config.INEQ: self.fc_ineq = nn.Linear(config.NPOP, self.h)
        self.fc = nn.Linear(self.h * (1 + int(config.INEQ)), self.h)
        self.actionNet = ConstDiscrete(config, self.h, self.outDim)

    def forward(self, s, ineq, done=False):
        s = self.envNet(s, is_done=done)
        if self.config.INEQ:
            ineq = F.relu(self.fc_ineq(ineq))
            s = torch.cat([s, ineq], dim=1)
        x = F.relu(self.fc(s))
        pi, actionIdx = self.actionNet(x)
        return pi, actionIdx
