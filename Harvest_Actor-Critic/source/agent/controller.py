from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch

from source.networks.ann import ANN
from source.utils.rollouts import Rollout
from source.utils.torch.param import setParameters, zeroGrads


class Controller:
    def __init__(self, config):
        self.config = config
        self.nANN = config.NPOP
        self.anns = [ANN(config) for _ in range(self.nANN)]

        self.updates, self.rollouts = defaultdict(lambda: Rollout()), {}
        self.initBuffer()

    def backward(self):
        self.blobs = [r.feather.blob for r in self.rollouts.values()]
        self.rollouts = {}

    def sendLogUpdate(self):
        blobs = self.blobs
        self.blobs = []
        return blobs

    def recvUpdate(self, update):
        for idx, paramVec in enumerate(update):
            setParameters(self.anns[idx], paramVec)
            zeroGrads(self.anns[idx])

    def collectStep(self, entID, atnArgs, val, reward, stim=None):
        if self.config.TEST:
            return
        self.updates[entID].step(atnArgs, val, reward, stim)

    def collectRollout(self, entID, ent, tick, epoch):
        assert entID not in self.rollouts
        rollout = self.updates[entID]
        rollout.finish()
        rollout.feather.blob.tick = tick
        annID = ent.annID
        self.buffer[annID]['return'][(epoch + 1) * self.config.HORIZON - 1] = self.buffer[annID]['reward'][
            (epoch + 1) * self.config.HORIZON - 1]
        for i in reversed(range(epoch * self.config.HORIZON, (epoch + 1) * self.config.HORIZON - 1)):
            self.buffer[annID]['return'][i] = self.buffer[annID]['reward'][i] + \
                                              self.config.GAMMA * self.buffer[annID]['return'][i + 1]
        self.rollouts[entID] = rollout
        del self.updates[entID]

    def initBuffer(self):
        batchSize = self.config.HORIZON * self.config.EPOCHS
        self.buffer = [{'state': np.ndarray((batchSize, 3, 15, 15), dtype=float),
                        'policy': np.ndarray((batchSize, 8), dtype=float),
                        'action': np.ndarray((batchSize, 1), dtype=int),
                        'reward': np.ndarray((batchSize, 1), dtype=float),
                        'return': np.ndarray((batchSize, 1), dtype=float),
                        'global_state': np.ndarray((batchSize, 3, 16, 38), dtype=float),
                        'ineq': np.ndarray((batchSize, self.config.NPOP), dtype=float),
                        'action_other': np.ndarray((batchSize, 8 * (self.config.NPOP - 1)), dtype=int)}
                       for i in range(self.nANN)]

    def dispatchBuffer(self):
        buffer = deepcopy(self.buffer)
        self.initBuffer()
        return buffer

    @torch.no_grad()
    def decide(self, ent, stim, reward, isDead, step, epoch, global_state, apples, ineq):
        entID, annID = ent.agent_id + str(epoch), ent.annID

        stim = self.prepare_input(stim)
        stim_tensor = torch.from_numpy(stim).unsqueeze(0).float()
        global_state = self.prepare_input(global_state)
        ineq_tensor = torch.from_numpy(ineq).unsqueeze(0).float() if self.config.INEQ else None
        annReturns = self.anns[annID](stim_tensor, ineq_tensor, isDead)

        self.buffer[annID]['state'][step] = stim
        self.buffer[annID]['global_state'][step] = global_state
        if self.config.INEQ: self.buffer[annID]['ineq'][step] = ineq
        self.buffer[annID]['policy'][step] = annReturns['outputs'][0].numpy()
        self.buffer[annID]['action'][step] = annReturns['outputs'][1]
        action = int(annReturns['outputs'][1])
        self.updates[entID].feather.scrawl(apples, ent, reward)
        return action

    def prepare_input(self, stim):
        stim = np.transpose(stim, (2, 0, 1)).copy()
        return stim
