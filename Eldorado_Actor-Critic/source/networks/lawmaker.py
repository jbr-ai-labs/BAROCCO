from torch import nn

from source.networks.utils import QNet


class COMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qNet = QNet(config, 10)

    def forward(self, state):
        Qs = self.qNet(state)
        return {'Qs': Qs}

    def get_advantage(self, outputs, actionsAgent, pi):
        Q_new = outputs['Qs'].gather(1, actionsAgent.view(-1, 1)) - (pi * outputs['Qs']).sum(1, keepdim=True)
        return {'Qs': Q_new}
