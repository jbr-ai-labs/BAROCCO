import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn import functional as F


def classify(logits):
    if len(logits.shape) == 1:
        logits = logits.view(1, -1)
    distribution = Categorical(F.softmax(logits, dim=1))
    atn = distribution.sample()
    return atn


####### Network Modules
class ConstDiscrete(nn.Module):
    def __init__(self, config, h, nattn):
        super().__init__()
        self.fc1 = nn.Linear(h, nattn)
        self.config = config

    def forward(self, stim):
        x = self.fc1(stim)
        xIdx = classify(x)
        return x, xIdx


####### End network modules


class QNet(nn.Module):
    def __init__(self, config, action_size=8, outDim=8, device='cpu', batch_size=1, use_actions=False):
        super().__init__()
        self.config = config
        self.h = config.HIDDEN
        self.use_actions = use_actions
        self.envNet = Env(config, config.LSTM, device=device, batch_size=batch_size)
        self.envNetGlobal = Env(config, False, device=device, batch_size=batch_size, in_dim=config.ENT_DIM_GLOBAL)
        if self.use_actions: self.fc_actions = nn.Linear((config.NPOP - 1) * action_size, self.h)
        if config.INEQ: self.fc_ineq = nn.Linear(config.NPOP, self.h)
        self.fc = nn.Linear(self.h * (2 + int(use_actions) + int(config.INEQ)), self.h)
        self.QNet = torch.nn.Linear(self.h, outDim)

    def forward(self, state, global_state, actions, ineq):
        stim = self.envNet(state)
        global_stim = self.envNetGlobal(global_state)
        x = torch.cat([stim, global_stim], dim=1)
        if self.use_actions:
            actions = F.relu(self.fc_actions(actions))
            x = torch.cat([x, actions], dim=1)
        if self.config.INEQ:
            ineq = F.relu(self.fc_ineq(ineq))
            x = torch.cat([x, ineq], dim=1)
        x = F.relu(self.fc(x))
        x = self.QNet(x)
        return x


class Env(nn.Module):
    def __init__(self, config, isLstm=False, isLm=False, device='cpu', batch_size=1, in_dim=1014):
        super().__init__()
        h = config.HIDDEN
        self.config = config
        self.h = h
        self.batch_size = batch_size

        self.lstm = nn.LSTM(h, h, batch_first=True).to(device) if isLstm else None
        self.isLstm = isLstm
        self.isLm = isLm
        self.device = device
        if isLstm:
            if self.isLm:
                self.hiddens = [self.init_hidden(self.batch_size, h) for _ in range(config.NPOP)]
            else:
                self.hidden = self.init_hidden(self.batch_size, h)
        self.fc1 = nn.Linear(in_dim, h)
        self.conv = nn.Conv2d(3, 6, (3, 3))

    def init_hidden(self, batch_size, h):
        hidden = Variable(next(self.parameters()).data.new(1, batch_size, h), requires_grad=False)
        cell = Variable(next(self.parameters()).data.new(1, batch_size, h), requires_grad=False)
        return hidden.zero_().to(self.device), cell.zero_().to(self.device)

    def forward(self, s, is_done=False, agent_id=None):
        x = F.relu(self.conv(s.to(self.device)).view(s.shape[0], -1))
        x = F.relu(self.fc1(x))
        if self.isLstm:
            if self.batch_size != 1:
                x, _ = self.lstm(x.unsqueeze(0))
            else:
                x, hidden = self.lstm(x.unsqueeze(0), self.hidden) if not self.isLm else \
                            self.lstm(x.unsqueeze(0), self.hiddens[agent_id])
                if is_done:
                    if self.isLm:
                        self.hiddens[agent_id] = (self.init_hidden(1, self.h))
                    else:
                        self.hidden = self.init_hidden(1, self.h)
                else:
                    if self.isLm:
                        self.hiddens[agent_id] = hidden
                    else:
                        self.hidden = hidden
            x = F.relu(x.squeeze(0))
        return x

    def reset_noise(self):
        pass


class EnvDummy(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, in_features, out_features, activation=True):
        super().__init__()
        self.config = config

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU() if activation else lambda x: x

    def forward(self, env):
        x = F.relu(self.conv(env).view(env.shape[0], -1))
        x = self.activation(self.fc(x))
        return x


class EnvGlobal(EnvDummy):
    def __init__(self, config, out_features):
        super().__init__(config, 3, 6, (3, 3), config.ENT_DIM_GLOBAL, out_features, True)
