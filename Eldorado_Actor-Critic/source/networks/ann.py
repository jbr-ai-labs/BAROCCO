from torch import nn

from source.networks.actions import ActionNet
from source.networks.utils import ValNet


class ANN(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.valNet = ValNet(config)

        self.config, self.args = config, args
        self.actionNet = ActionNet(config, args, entDim=config.ENT_DIM, outDim=10)

    def forward(self, state, state_global=None, train=False):
        val = self.valNet(state_global) if train else 0
        pi, actionIdx = self.actionNet(state)

        return {'actions': actionIdx,
                'policy': pi,
                'val': val}
