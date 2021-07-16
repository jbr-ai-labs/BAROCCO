from torch import nn

from source.networks.utils import QNet


class COMA(nn.Module):
    def __init__(self, config, device='cpu', batch_size=1):
        super().__init__()
        self.QNet = QNet(config, outDim=8, device=device, batch_size=batch_size, use_actions=True)

    def forward(self, state, global_state, actions, ineq):
        Qs = self.QNet(state, global_state, actions, ineq)
        return Qs

    def get_advantage(self, outputs, actionAgent, pi):
        Q = outputs.gather(1, actionAgent.view(-1, 1)) - (pi * outputs).sum(1, keepdim=True)
        return Q
