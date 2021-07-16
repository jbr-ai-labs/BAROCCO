from torch import nn

from source.networks.actions import ActionNet
from source.networks.utils import ValNet


class ANN(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.config, self.args = config, args
        self.valNet = ValNet(config, n_quant=config.N_QUANT, noise=self.config.NOISE)
        self.actionNet = ActionNet(args, config, n_quant=config.N_QUANT, noise=self.config.NOISE)

    def forward(self, state, eps=0, punishmentLm=None):
        pi, actionIdx = self.actionNet(state, eps, punishmentLm)
        return pi.to('cpu'), actionIdx

    def getVal(self, state):
        return self.valNet(state).to('cpu')

    def reset_noise(self):
        self.valNet.reset_noise()
        self.actionNet.reset_noise()
