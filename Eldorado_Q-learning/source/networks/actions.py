from torch import nn

from source.networks.utils import ConstDiscrete, Env


class ActionNet(nn.Module):
    def __init__(self, args, config, n_quant=1, noise=False):
        super().__init__()
        self.config = config
        self.envNet = Env(config, noise)
        self.actionNet = ConstDiscrete(args, config, n_quant)

    def forward(self, state, eps=0, punish=None):
        x = self.envNet(state)
        outs, idx = self.actionNet(x, eps, punish)
        return outs, idx

    def reset_noise(self):
        self.envNet.reset_noise()
