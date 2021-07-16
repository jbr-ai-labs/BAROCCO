from torch import nn

from source.networks.actions import ActionNet
from source.networks.utils import ValNet, Env


class ANN(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.config, self.args = config, args
        self.valNet = ValNet(config, n_quant=config.N_QUANT, noise=self.config.NOISE)
        self.actionNet = ActionNet(config, args,  outDim=8, n_quant=config.N_QUANT, noise=self.config.NOISE)
        self.envNet = Env(config, self.config.NOISE)

    def forward(self, env, eps=0, punishmentsLm=None):
        s = self.envNet(env)
        val = self.valNet(s)

        pi, actionIdx = self.actionNet(s, eps, punishmentsLm)
        outputs = (pi.to('cpu'), actionIdx)

        return outputs, val.to('cpu')

    def reset_noise(self):
        self.envNet.reset_noise()
        self.valNet.reset_noise()
        self.actionNet.reset_noise()
