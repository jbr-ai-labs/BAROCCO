from torch import nn

from source.networks.actions import ActionNet


class SocialANN(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.config = config

        self.actionNet = ActionNet(args, config, noise=self.config.NOISE)

    def forward(self, state):
        outputs = self.actionNet(state)[0].to('cpu')
        punishments = outputs.mean(2).detach() * self.config.PUNISHMENT
        return outputs, punishments

    def reset_noise(self):
        self.actionNet.reset_noise()
