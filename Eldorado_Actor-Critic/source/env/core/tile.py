import numpy as np


class Tile:
    def __init__(self, mat, r, c, nCounts, tex):
        self.r, self.c = r, c
        self.mat = mat()
        self.ents = {}
        self.state = mat()
        self.capacity = self.mat.capacity
        self.counts = np.zeros(nCounts)
        self.tex = tex

    @property
    def nEnts(self):
        return len(self.ents)

    def addEnt(self, entID, ent):
        assert entID not in self.ents
        self.ents[entID] = ent

    def delEnt(self, entID):
        assert entID in self.ents
        del self.ents[entID]

    def step(self):
        if not self.static:
            if self.mat.curPeriod <= 0:
                self.capacity += 1
                self.mat.curPeriod = self.mat.respawnPeriod
            else:
                self.mat.curPeriod -= 1
        # Try inserting a pass
        if self.static:
            self.state = self.mat

    @property
    def static(self):
        assert self.capacity <= self.mat.capacity
        return self.capacity == self.mat.capacity

    def harvest(self):
        if self.capacity == 0:
            return False
        elif self.capacity <= 1:
            self.state = self.mat.degen()
        self.capacity -= 1
        return True
