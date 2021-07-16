from torch.nn.utils import clip_grad_value_
from random import shuffle
import numpy as np

from source.agent.learner import Model, Learner
from source.utils.torch.forward import ForwardSocial
from source.utils.torch.optim import backwardAgent


class ModelSocial(Model):
    def __init__(self, config, args):
        super(ModelSocial, self).__init__(config, args)

    def backward(self, batch):
        self.lmOpt.zero_grad()
        backwardAgent(batch, device=self.config.device)
        [clip_grad_value_(lm.parameters(), 10) for lm in self.social]
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

    def step(self, sample, n_epochs=1):
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
                    batch = self.forward.forward(sample_cur, self.net.targetAnns, self.net.social, self.net.targetSocial,
                                                 self.net.mixer, self.net.targetMixer, device=self.config.device)
                    self.net.backward(batch)

            if not self.config.TEST and (self.tick + 1) % 16 == 0:
                self.net.checkpoint()

        else:
            batch = self.forward.forward(sample, self.net.targetAnns, self.net.social, self.net.targetSocial,
                                         self.net.mixer, self.net.targetMixer, device=self.config.device)
            self.net.backward(batch)

            if not self.config.TEST and (self.tick + 1) % 256 == 0:
                self.net.checkpoint()

        if (self.tick + 1) % self.config.TARGET_PERIOD == 0:
            self.net.updateTargetAnns()

        self.tick += 1

        return self.net.sendLm()
