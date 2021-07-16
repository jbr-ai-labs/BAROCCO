import numpy as np

from source.env.lib.enums import Palette


class Spawner:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.nEnt, self.nANN = config.NENT, config.NPOP
        self.popSz = self.nEnt // self.nANN
        self.popCounts = np.zeros(self.nANN)
        self.Timeout = np.zeros(self.nANN)
        self.palette = Palette(config.NPOP)
        self.entID = 0

    # Returns IDs for spawning
    def spawn(self):
        self.Timeout -= np.min(self.Timeout)
        spawned_lst = []
        for annID in range(self.nANN):
            for pop in range(self.popSz):
                if (self.Timeout[annID] > 0) or (self.popCounts[annID] == self.popSz):
                    continue
                entID = str(self.entID)
                self.entID += 1

                self.popCounts[annID] += 1
                color = self.palette.color(annID)

                spawned_lst.append((entID, (annID, color)))
        return spawned_lst

    def cull(self, annID):
        self.popCounts[annID] -= 1
        self.Timeout[annID] = self.config.TIMEOUT
        assert self.popCounts[annID] >= 0

    def decrementTimeout(self):
        self.Timeout = np.max((self.Timeout - 1, np.zeros(self.Timeout.shape)), axis=0)
