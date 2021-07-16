import time

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_value_

from source.networks.ann import ANN
from source.networks.qmix import QMixer, VDNMixer
from source.networks.social import SocialANN
from source.utils.torch import save
from source.utils.torch.forward import Forward
from source.utils.torch.optim import backwardAgent


class Model:
    def __init__(self, config, args):
        self.saver = save.Saver(config.NPOP, config.MODELDIR,
                                'models', 'bests', 'social', resetTol=256)
        self.config, self.args = config, args
        self.nANN = config.NPOP
        self.envNets = []

        self.init()
        if self.config.LOAD or self.config.BEST:
            self.load(self.config.BEST)

    def init(self):
        print('Initializing new model...')
        self.anns = [ANN(self.config, self.args).to(self.config.device) for _ in range(self.nANN)]
        self.targetAnns = [ANN(self.config, self.args).to(self.config.device) for _ in range(self.nANN)]
        self.annsOpts = [Adam(ann.parameters(), lr=self.config.LR, weight_decay=0.00001) for ann in self.anns]
        self.schedulers = [StepLR(opt, 1, gamma=0.999995) for opt in self.annsOpts]

        self.social = [SocialANN(self.args, self.config).to(self.config.device) for _ in range(self.nANN)]
        self.targetSocial = [SocialANN(self.args, self.config).to(self.config.device) for _ in range(self.nANN)]
        self.mixer = QMixer(self.config).to(self.config.device) if self.config.QMIX else VDNMixer().to(
            self.config.device)
        self.targetMixer = QMixer(self.config).to(self.config.device) if self.config.QMIX else VDNMixer().to(
            self.config.device)
        self.lmOpt = Adam(sum([list(lm.parameters()) for lm in self.social], []) + list(self.mixer.parameters()),
                          lr=self.config.LR, weight_decay=0.00001)
        self.lmScheduler = StepLR(self.lmOpt, 1, gamma=0.999995)

        self.updateTargetAnns()

    def backward(self, batches):
        [opt.zero_grad() for opt in self.annsOpts]
        [backwardAgent(batch, device=self.config.device, n_quant=self.config.N_QUANT) for batch in batches]
        [clip_grad_value_(ann.parameters(), 10) for ann in self.anns]
        [opt.step() for opt in self.annsOpts]
        [sc.step() for sc in self.schedulers]

    def checkpoint(self, reward):
        self.saver.checkpoint(reward, self.anns, self.annsOpts)

    def load(self, best=False):
        print('Loading model...')
        self.saver.load(self.annsOpts, self.anns, best, self.lmOpt, self.social)

    def loadAnnsFrom(self, states):
        states = [self.convertStateDict(state) for state in states]
        [ann.load_state_dict(state) for ann, state in zip(self.anns, states)]

    def loadLmFrom(self, states):
        states = [self.convertStateDict(state) for state in states]
        [lm.load_state_dict(state) for lm, state in zip(self.social, states)]

    def sendAnns(self):
        states = [ann.state_dict() for ann in self.anns]
        states = [self.convertStateDict(state, device='cpu') for state in states]
        return states

    def sendLm(self):
        states = [lm.state_dict() for lm in self.social]
        states = [self.convertStateDict(state, device='cpu') for state in states]
        return states

    def reset_noise(self):
        nets = self.anns + self.targetAnns + self.social + self.targetSocial
        for net in nets:
            net.reset_noise()

    def updateTargetAnns(self):
        [t.load_state_dict(a.state_dict()) for a, t in zip(self.anns, self.targetAnns)]
        [t.load_state_dict(l.state_dict()) for l, t in zip(self.social, self.targetSocial)]
        self.targetMixer.load_state_dict(self.mixer.state_dict())

    def convertStateDict(self, state, device=None):
        if device is None:
            device = self.config.device
        for k, v in state.items():
            state[k] = v.to(device)
        return state


class Learner:
    def __init__(self, config, args):
        self.start, self.tick, self.nANN = time.time(), 0, config.NPOP
        self.config, self.args = config, args
        self.net = Model(config, args)
        self.forward = Forward(config, args)

        self.period = 1

    def step(self, samples, weights, lifetime=0):
        if self.config.NOISE:
            self.net.reset_noise()
        batches, priorities = self.forward.forward_multi(samples, weights, self.net.anns, self.net.targetAnns,
                                                         self.net.targetSocial, device=self.config.device)
        self.net.backward(batches)

        self.tick += 1
        if not self.config.TEST:
            if (self.tick + 1) % 256 == 0:
                self.net.checkpoint(lifetime)
                self.net.saver.print()

            if (self.tick + 1) % self.config.TARGET_PERIOD == 0:
                self.net.updateTargetAnns()

        return self.net.sendAnns(), priorities
