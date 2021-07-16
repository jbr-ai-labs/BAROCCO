from torch.nn.utils import clip_grad_value_
from random import shuffle
import numpy as np
import torch

from source.agent import Learner
from source.agent.learner import Model
from source.utils.torch.forward import ForwardSocial
from source.utils.torch.optim import backwardAgent


class ModelSocial(Model):
    def __init__(self, config, args):
        super(ModelSocial, self).__init__(config, args)

    def backward(self, batch):
        self.lmOpt.zero_grad()
        backwardAgent(batch, device=self.config.device, n_quant=self.config.N_QUANT_LM)
        [clip_grad_value_(ann.parameters(), 10) for ann in self.social]
        clip_grad_value_(self.mixer.parameters(), 10)
        self.lmOpt.step()
        self.lmScheduler.step()

    def checkpoint(self, reward=None):
        if self.config.TEST:
            return
        self.saver.checkpointSocial(self.social, self.lmOpt)


class LearnerSocial(Learner):
    def __init__(self, config, args):
        super(LearnerSocial, self).__init__(config, args)
        self.net = ModelSocial(config, args)
        self.forward = ForwardSocial(config, args)

    def step(self, sample, weights, n_epochs=1):
        if self.config.NOISE:
            self.net.reset_noise()

        if not self.config.REPLAY_LM:
            n_updates = int(np.ceil(len(sample) / self.config.BATCH_SIZE_LM))
            batch_size = int(np.ceil(len(sample) / n_updates))

            for epoch in range(n_epochs):
                shuffle(sample)
                for i in range(n_updates):
                    if self.config.NOISE:
                        self.net.reset_noise()
                    sample_cur = sample[i * batch_size: min((i + 1) * batch_size, len(sample))]
                    batch, priorities = self.forward.forward(sample_cur, weights, self.net.targetAnns, self.net.social,
                                                             self.net.targetSocial, self.net.mixer, self.net.targetMixer,
                                                             device=self.config.device)
                    self.net.backward(batch)
                    self.tick += 1

        else:
            batch, priorities = self.forward.forward(sample, weights, self.net.targetAnns, self.net.social,
                                                     self.net.targetSocial, self.net.mixer, self.net.targetMixer,
                                                     device=self.config.device)
            self.net.backward(batch)
            self.tick += 1

        if not self.config.TEST and (self.tick + 1) % 256 == 0:
            self.net.checkpoint()

        if (self.tick + 1) % self.config.TARGET_PERIOD_LM == 0:
            self.net.updateTargetAnns()

        return self.net.sendLm(), priorities
