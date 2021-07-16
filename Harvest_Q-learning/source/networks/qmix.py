from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from source.networks.utils import EnvGlobal

"""
Based on: https://github.com/ray-project/ray/pull/3548/files
"""


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        return th.sum(agent_qs, dim=1, keepdim=True)


class QMixer(nn.Module):
    def __init__(self, config, value=True):
        super(QMixer, self).__init__()

        self.config = config
        self.n_agents = self.config.NPOP
        self.embed_dim = 16
        self.hid_dim = 32
        self.nQuant = self.config.N_QUANT_LM

        self.envNet = EnvGlobal(config, self.hid_dim)

        self.hyper_w_1 = nn.Linear(self.hid_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.hid_dim, self.embed_dim * self.nQuant)
        self.hyper_b_1 = nn.Linear(self.hid_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.VALUE = value
        if self.VALUE:
            self.V = nn.Sequential(nn.Linear(self.hid_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.nQuant))

    def forward(self, agent_qs, states):
        """Forward pass for the mixer.
        Arguments:
            agent_qs: Tensor of shape [B, n_agents]
            states: Tensor of shape [B, state_dim]
        """
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        states = self.envNet(states)

        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, self.nQuant)

        # Compute final output
        y = th.bmm(hidden, w_final)

        # State-dependent bias
        if self.VALUE:
            y += self.V(states).view(-1, 1, self.nQuant)

        # Reshape and return
        q_tot = y.view(bs, -1, self.nQuant)
        return q_tot
