from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from source.env.lib.log import Quill
from source.networks import COMA
from source.networks.ann import ANN
from source.utils.torch import save, optim
from source.utils.torch.param import getParameters


def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0):
    for p in mod.parameters():
        if len(p.data.shape) >= 2:
            orthogonal_init(p.data, gain=scale)
        else:
            p.data.zero_()


class Model:
    def __init__(self, config):
        self.saver = save.Saver(config.NPOP, config.MODELDIR,
                                'models', 'bests', 'lawmaker', resetTol=256)
        self.config = config
        self.entropy = self.config.ENTROPY

        self.init()
        if self.config.LOAD or self.config.BEST:
            self.load(self.config.BEST)

    def init(self):
        print('Initializing new model...')
        self.unshared(self.config.NPOP)
        self.opt = [Adam(ann.parameters(), lr=self.config.LR, weight_decay=0.00001) for ann in self.anns]
        self.scheduler = [StepLR(opt, 1, gamma=self.config.LR_DECAY) for opt in self.opt]

        self.social = [COMA(self.config, device=self.config.DEVICE_OPTIMIZER,
                            batch_size=self.config.LSTM_PERIOD).to(self.config.DEVICE_OPTIMIZER) for _ in
                       range(self.config.NPOP)]
        [initialize_weights(ann) for ann in self.social]
        self.socLoss = Adam(sum([list(lm.parameters()) for lm in self.social], []),
                            lr=self.config.LR, weight_decay=0.00001)
        self.socScheduler = StepLR(self.socLoss, 1, gamma=self.config.LR_DECAY)

    def unshared(self, n):
        self.anns = [ANN(self.config, device=self.config.DEVICE_OPTIMIZER,
                         batch_size=self.config.LSTM_PERIOD).to(self.config.DEVICE_OPTIMIZER) for _ in range(n)]
        [initialize_weights(ann) for ann in self.anns]

    def annealEntropy(self):
        self.entropy = max(self.entropy * self.config.ENTROPY_ANNEALING, self.config.MIN_ENTROPY)

    def checkpoint(self, reward):
        self.saver.checkpoint(reward, self.anns, self.opt)
        self.saver.checkpointSocial(self.social, [self.socLoss])

    def load(self, best=False):
        print('Loading model...')
        self.saver.load(self.opt, self.anns, [self.socLoss], self.social, best)

    def model(self):
        return [getParameters(ann) for ann in self.anns]


class Learner:
    def __init__(self, config):
        self.config = config
        self.net = Model(config)
        self.quill = Quill(config.MODELDIR)

    def tenzorify(self, ndarray):
        return torch.from_numpy(ndarray).float().to(self.config.DEVICE_OPTIMIZER)

    def gatherTrajectory(self, states, rewards, policy, actions, global_states, actions_other, ineqs, annID):
        trajectory = defaultdict(list)

        for i in range(0, len(states), self.config.LSTM_PERIOD):
            stim = states[i: i + self.config.LSTM_PERIOD]
            ret = rewards[i: i + self.config.LSTM_PERIOD]
            oldPolicy = policy[i: i + self.config.LSTM_PERIOD]
            action = actions[i: i + self.config.LSTM_PERIOD].long()
            global_stim = global_states[i: i + self.config.LSTM_PERIOD]
            action_other = actions_other[i: i + self.config.LSTM_PERIOD]
            ineq = ineqs[i: i + self.config.LSTM_PERIOD]

            annReturns = self.net.anns[annID](stim, ineq, (i + 1) % self.config.LSTM_PERIOD == 0)
            val = self.net.anns[annID].getVal(stim, global_stim, action_other, ineq)
            outsSocial = self.net.social[annID](stim, global_stim, action_other, ineq)
            advSocial = self.net.social[annID].get_advantage(outsSocial, action, F.softmax(oldPolicy, dim=1))
            trajectory['Qs'].append(outsSocial.gather(1, action.view(-1, 1)))
            trajectory['As'].append(advSocial)
            trajectory['vals'].append(val)
            trajectory['returns'].append(ret)
            trajectory['oldPolicy'].append(F.softmax(oldPolicy, dim=1).gather(1, action.view(-1, 1)))
            trajectory['policy'].append(F.softmax(annReturns['outputs'][0], dim=1).gather(1, action.view(-1, 1)))
            trajectory['actions'].append(action)

        return trajectory

    def prepare_batch(self, batch):
        prepared = defaultdict(lambda: defaultdict(list))
        for annID, agentBatch in enumerate(batch):
            for key, value in agentBatch.items():
                prepared[annID][key] = self.tenzorify(value)
        return prepared

    def offPolicyTrain(self, batch):
        step = 500
        for i in range(0, self.config.HORIZON * self.config.EPOCHS, step):
            trajectories = []
            start = i
            for annID, agentBatch in batch.items():
                trajectory = self.gatherTrajectory(
                    agentBatch['state'][start:i + step],
                    agentBatch['reward'][start:i + step],
                    agentBatch['policy'][start:i + step],
                    agentBatch['action'][start:i + step],
                    agentBatch['global_state'][start:i + step],
                    agentBatch['action_other'][start:i + step],
                    agentBatch['ineq'][start:i + step],
                    annID)
                trajectories.append(trajectory)
            loss, outs = optim.backwardAgentOffPolicy(trajectories,
                                                      entWeight=self.net.entropy,
                                                      lambda_adv=self.config.LAMBDA_ADV_LM,
                                                      device=self.config.DEVICE_OPTIMIZER)
            for Qs in outs['Qs']:
                Qs = Qs.to(self.config.DEVICE_OPTIMIZER).float()
                if not self.config.VANILLA_COMA:
                    socLoss = optim.backwardSocial(outs['rets'], Qs,
                                                   device=self.config.DEVICE_OPTIMIZER, mode=self.config.LM_MODE)
                else:
                    socLoss = optim.backwardSocialVanillaComa(outs['rewards'], Qs,
                                                              device=self.config.DEVICE_OPTIMIZER,
                                                              mode=self.config.LM_MODE)
                socLoss.backward()
            [nn.utils.clip_grad_norm_(lm.parameters(), 0.5) for lm in self.net.social]
            self.net.socLoss.step()
            self.net.socScheduler.step()
            self.net.socLoss.zero_grad()
            loss.backward()
            [nn.utils.clip_grad_norm_(ann.parameters(), 0.5) for ann in self.net.anns]
            [opt.step() for opt in self.net.opt]
            self.net.annealEntropy()
            [scheduler.step() for scheduler in self.net.scheduler]
            [opt.zero_grad() for opt in self.net.opt]
        return

    def model(self):
        return self.net.model()

    def step(self, batch, logs):
        reward = self.quill.scrawl(logs)

        batch = self.prepare_batch(batch)
        for i in range(self.config.EPOCHS_PPO):
            self.offPolicyTrain(batch)

        self.net.checkpoint(reward)
        self.net.saver.print()

        return self.model()
