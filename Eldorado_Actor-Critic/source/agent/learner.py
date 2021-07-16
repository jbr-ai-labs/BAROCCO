from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from source.env.lib.log import Quill
from source.networks.ann import ANN
from source.networks.lawmaker import COMA
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
    def __init__(self, config, args):
        self.saver = save.Saver(config.NPOP, config.MODELDIR, 'models', 'bests', 'lawmaker', resetTol=256)
        self.config, self.args = config, args
        self.entropy = config.ENTROPY
        self.init()
        if self.config.LOAD or self.config.BEST:
            self.load(self.config.BEST)

    def init(self):
        print('Initializing new model...')
        self.unshared(self.config.NPOP)
        self.opt = [Adam(ann.parameters(), lr=self.config.LR, weight_decay=0.00001) for ann in self.anns]
        self.scheduler = [StepLR(opt, 1, gamma=0.999998) for opt in self.opt]

        self.social = [COMA(self.config).to(self.config.DEVICE_OPTIMIZER) for _ in range(self.config.NPOP)]
        [initialize_weights(s) for s in self.social]
        self.socOpt = Adam(sum([list(ann.parameters()) for ann in self.social], []), lr=self.config.LR,
                           weight_decay=0.00001)
        self.socScheduler = StepLR(self.socOpt, 1, gamma=0.999998)

    def unshared(self, n):
        self.anns = [ANN(self.config, self.args).to(self.config.DEVICE_OPTIMIZER) for _ in range(n)]
        [initialize_weights(ann) for ann in self.anns]

    def annealEntropy(self):
        self.entropy = max(self.entropy * self.config.ENTROPY_ANNEALING, self.config.MIN_ENTROPY)

    def checkpoint(self, reward):
        self.saver.checkpoint(reward, self.anns, self.opt)
        self.saver.checkpointSocial(self.social, [self.socOpt])

    def load(self, best=False):
        print('Loading model...')
        self.saver.load(self.opt, self.anns, [self.socOpt], self.social, best)

    def model(self):
        return [getParameters(ann) for ann in self.anns], [getParameters(ann) for ann in self.social]


class Learner:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.net = Model(config, args)
        self.quill = Quill(config.MODELDIR)

    def tenzorify(self, nparray):
        return torch.from_numpy(nparray).float().to(self.config.DEVICE_OPTIMIZER)

    def gatherTrajectory(self, states, states_global, returns, policy, actions, states_next, dead, rewards, annID):
        trajectory = {'vals': [],
                      'returns': [],
                      'Qs': [],
                      'As': [],
                      'retsTD': [],
                      'oldPolicy': [],
                      'policy': [],
                      'actions': []}
        dead = dead.bool()
        oldPolicy = policy.squeeze(1)
        action = actions.long().view(-1, 1)

        outsSocial = self.net.social[annID](states_global)
        annReturns = self.net.anns[annID](states, states_global, True)
        trajectory['Qs'].append(outsSocial['Qs'].gather(1, action.view(-1, 1)))
        outsSocial = self.net.social[annID].get_advantage(outsSocial, action, F.softmax(oldPolicy, dim=1))

        vals_next = self.net.anns[annID].valNet(states_next).view(-1)
        vals_next[dead] = 0
        rets = rewards + self.config.GAMMA * vals_next

        trajectory['As'].append(outsSocial['Qs'].detach())
        trajectory['vals'].append(annReturns['val'])
        trajectory['returns'].append(returns)
        trajectory['retsTD'].append(rets)
        trajectory['oldPolicy'].append(oldPolicy)
        trajectory['policy'].append(annReturns['policy'])
        trajectory['actions'].append(action)

        return trajectory

    def train(self, batch):
        step = len(batch[0]['states']) // 8 + 1
 
        inds = np.random.permutation(len(batch[0]['states']))
        for i in range(0, len(batch[0]['states']), step):
            trajectories = []
            start = i
            idx = inds[start:i + step]
            for annID, agentBatch in sorted(batch.items()):
                trajectory = self.gatherTrajectory(
                    agentBatch['states'][idx],
                    agentBatch['states_global'][idx],
                    agentBatch['return'][idx],
                    agentBatch['policy'][idx],
                    agentBatch['action'][idx],
                    agentBatch['states_next'][idx],
                    agentBatch['dead'][idx],
                    agentBatch['reward'][idx],
                    annID)
                trajectories.append(trajectory)
            loss, outs = optim.backwardAgent(trajectories, entWeight=self.net.entropy,
                                             device=self.config.DEVICE_OPTIMIZER,
                                             lambda_adv=self.config.LAMBDA_ADV)
            for Qs in outs['Qs']:
                Qs = Qs.to(self.config.DEVICE_OPTIMIZER).float()
                socLoss = optim.backwardSocial(outs['retsTD'], Qs, device=self.config.DEVICE_OPTIMIZER,  mode=self.config.SW)
                socLoss.backward()

            [nn.utils.clip_grad_norm_(ann.parameters(), 0.5) for ann in self.net.social]
            self.net.socOpt.step()
            self.net.socScheduler.step()
            self.net.socOpt.zero_grad()

            loss.backward()
            [nn.utils.clip_grad_norm_(ann.parameters(), 0.5) for ann in self.net.anns]
            [opt.step() for opt in self.net.opt]
            self.net.annealEntropy()
            [scheduler.step() for scheduler in self.net.scheduler]
            [opt.zero_grad() for opt in self.net.opt]
        return

    def prepare_batch(self, batch):
        prepared = defaultdict(dict)
        for annID, agentBatch in batch.items():
            for key, value in agentBatch.items():
                prepared[annID][key] = self.tenzorify(value)
        return prepared

    def model(self):
        return self.net.model()

    def step(self, batch, logs):
        # Write logs
        lifetime = self.quill.scrawl(logs)

        batch = self.prepare_batch(batch)
        for i in range(self.config.EPOCHS_PPO):
            self.train(batch)

        self.net.checkpoint(lifetime)
        self.net.saver.print()

        return self.model()
