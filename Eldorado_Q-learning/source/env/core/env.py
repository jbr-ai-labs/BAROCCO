import numpy as np

from source.env.core import Map


class Env:
    def __init__(self, config, idx):
        # Load the world file
        self.env = Map(config, idx)
        self.shape = self.env.shape
        self.spawn = config.SPAWN
        self.config = config

        self.stimSize = 3
        self.worldDim = 2 * self.stimSize + 1
        self.tick = 0

    def stim(self, pos):
        return self.env.getPadded(self.env.tiles, pos, self.stimSize, key=lambda e: e.index).astype(np.int8)
