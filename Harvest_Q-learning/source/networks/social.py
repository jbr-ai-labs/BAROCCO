from torch import nn

from source.networks.actions import ActionNet
from source.networks.utils import Env


class SocialANN(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.config = config

        self.actionNet = ActionNet(config, args, outDim=8, noise=self.config.NOISE)
        self.envNet = Env(config, self.config.NOISE)

    def forward(self, env):
        s = self.envNet(env)
        outputs = self.actionNet(s, 0)[0].to('cpu')
        punishments = outputs.mean(2).detach() * self.config.PUNISHMENT
        return outputs, punishments

    def reset_noise(self):
        self.envNet.reset_noise()
        self.actionNet.reset_noise()
