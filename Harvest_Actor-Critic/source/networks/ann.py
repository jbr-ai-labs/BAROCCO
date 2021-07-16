from torch import nn

from source.networks.actions import ActionNet
from source.networks.utils import QNet


class ANN(nn.Module):
    def __init__(self, config, device='cpu', batch_size=1):
        super().__init__()
        self.actionNet = ActionNet(config, device=device, batch_size=batch_size)
        self.valNet = QNet(config, outDim=1, device=device, batch_size=batch_size, use_actions=False)

    def forward(self, state, ineq, isDead=None):
        pi, actionIdx = self.actionNet(state, ineq, isDead)
        return {'outputs': (pi, actionIdx)}

    def getVal(self, state, global_state, actions, ineq):
        return self.valNet(state, global_state, actions, ineq)
