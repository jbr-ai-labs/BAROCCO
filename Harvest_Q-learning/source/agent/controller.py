from collections import defaultdict
import numpy as np
import torch

from source.networks.ann import ANN
from source.networks.qmix import QMixer, VDNMixer
from source.networks.social import SocialANN
from source.utils.rollouts import Rollout
from source.utils.torch.forward import ForwardSocial, Forward
from source.utils.torch.replay import ReplayMemory, ReplayMemoryLm


class Controller:
    def __init__(self, config, args, idx):
        self.config, self.args = config, args
        self.nANN, self.h = config.NPOP, config.HIDDEN
        self.anns = [ANN(config, args) for _ in range(self.nANN)]

        self.init, self.nRollouts = True, 32
        self.updates = defaultdict(lambda: Rollout())
        self.blobs = []
        self.idx = idx
        self.ReplayMemory = [ReplayMemory(self.config) for _ in range(self.nANN)]
        self.ReplayMemoryLm = ReplayMemoryLm(self.config)
        self.forward, self.forwardSocial = Forward(config, args), ForwardSocial(config, args)

        self.social = [SocialANN(args, config) for _ in range(self.nANN)]
        self.mixer = QMixer(self.config) if self.config.QMIX else VDNMixer()
        self.tick = 0

    def sendBufferUpdate(self, evaluate=False):
        if evaluate:
            return None, None
        buffer = [replay.send_buffer() for replay in self.ReplayMemory]
        priorities = [self.forward.get_priorities_from_samples(buf, ann, ann, lm) for buf, ann, lm in
                      zip(buffer, self.anns, self.social)] if self.config.REPLAY_PRIO else [None] * len(self.anns)
        bufferLm = self.ReplayMemoryLm.send_buffer()
        prioritiesLm = self.forwardSocial.get_priorities_from_samples(bufferLm, self.anns, self.social, self.mixer) \
            if self.config.REPLAY_PRIO_LM else None
        return (buffer, priorities), (bufferLm, prioritiesLm)

    def sendLogUpdate(self):
        blobs = self.blobs
        self.blobs = []
        return blobs

    def sendUpdate(self, evaluate=False):
        recvs, recvs_lm = self.sendBufferUpdate(evaluate=evaluate)
        logs = self.sendLogUpdate()
        return recvs, recvs_lm, logs

    def recvUpdate(self, update):
        update, update_lm = update
        if update is not None:
            self.loadAnnsFrom(update)
        if update_lm is not None:
            self.loadLmFrom(update_lm)

    def collectStep(self, entID, annID, s, action, reward, dead):
        if self.config.TEST:
            return
        self.ReplayMemory[annID].append(entID, s, action, reward, dead)
        self.ReplayMemoryLm.append(entID, annID, s, action, reward, dead)

    def collectRollout(self, entID):
        rollout = self.updates[entID]
        rollout.feather.blob.tick = self.tick
        rollout.finish()
        self.blobs.append(rollout.feather.blob)
        del self.updates[entID]

    def decide(self, entID, annID, stim, reward, reward_stats, apples, isDead, evaluate=False):
        stim_tensor = self.prepareInput(stim)
        outsLm, punishLm = self.social[annID](stim_tensor)
        eps = self.config.EPS_CUR * int(not evaluate)
        atnArgs, val = self.anns[annID](stim_tensor, eps, punishLm)
        action = int(atnArgs[1])

        if not evaluate:
            self.collectStep(entID, annID, stim_tensor, atnArgs[1], reward, isDead)
        if not self.config.TEST:
            self.updates[entID].feather.scrawl(apples, annID, reward_stats)
        return action

    def prepareInput(self, stim):
        stim = np.transpose(stim, (2, 0, 1)).copy()
        stim_tensor = torch.from_numpy(stim).unsqueeze(0).float()
        return stim_tensor

    def reset_noise(self):
        nets = self.anns + self.social
        for net in nets:
            net.reset_noise()

    def loadAnnsFrom(self, states):
        [ann.load_state_dict(state) for ann, state in zip(self.anns, states)]

    def loadLmFrom(self, states):
        [lm.load_state_dict(state) for lm, state in zip(self.social, states[0])]
        self.mixer.load_state_dict(states[1])
