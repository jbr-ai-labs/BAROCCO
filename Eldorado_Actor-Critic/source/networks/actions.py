from torch import nn
from torch.nn import functional as F
from source.networks.utils import ConstDiscrete, Env


class ActionNet(nn.Module):
    def __init__(self, config, args, entDim=11, outDim=2):
        super().__init__()
        self.config, self.args, self.h = config, args, config.HIDDEN
        self.entDim, self.outDim = entDim, outDim
        self.fc = nn.Linear(self.h, self.h)
        self.actionNet = ConstDiscrete(config, self.h, self.outDim)
        self.envNet = Env(config)

    def forward(self, state):
        stim = self.envNet(state)
        x = F.relu(self.fc(stim))
        pi, actionIdx = self.actionNet(x)
        return pi, actionIdx
