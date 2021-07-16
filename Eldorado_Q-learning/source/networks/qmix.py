import torch
from torch import nn
from torch.nn import functional as F

"""
Based on: https://github.com/ray-project/ray/pull/3548/files
"""


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        return torch.sum(agent_qs, dim=1, keepdim=True)


class QMixer(nn.Module):
    def __init__(self, config, value=True):
        super().__init__()

        self.n_agents = config.NPOP
        self.h = config.ENT_DIM
        self.embed_dim = 16
        self.VALUE = value

        self.hyper_w_1 = nn.Linear(self.h, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.h, self.embed_dim)
        self.hyper_b_1 = nn.Linear(self.h, self.embed_dim)
        if self.VALUE:
            self.V = nn.Sequential(nn.Linear(self.h, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, state):
        bs = agent_qs.shape[0]
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = torch.abs(self.hyper_w_1(state))
        b1 = self.hyper_b_1(state)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = torch.abs(self.hyper_w_final(state))
        w_final = w_final.view(-1, self.embed_dim, 1)

        # Compute final output
        y = torch.bmm(hidden, w_final)

        # State-dependent bias
        if self.VALUE:
            y += self.V(state).view(-1, 1, 1)

        # Reshape and return
        q_tot = y.view(bs, 1)
        return q_tot
