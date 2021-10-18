import numpy as np

from source.env.core.config import Config


class Experiment(Config):
    def defaults(self):
        super().defaults()
        self.MODELDIR = 'resource/logs/'
        self.HIDDEN = 128
        self.LOAD = False
        self.BEST = False
        self.EPOCHS = 1
        self.EPOCHS_PPO = 10
        self.LAMBDA_ADV = 1.
        self.ATTACK = True
        self.SHARE = False
        self.STEPREWARD = 0.1
        self.DEADREWARD = -1
        self.NPOP = 2
        self.ENTROPY = 0.05
        self.ENTROPY_ANNEALING = 0.99998
        self.MIN_ENTROPY = 0.
        self.VAMPYR = 1
        self.RNG = 1
        self.DMG = 1
        self.DOUBLE_DMG_PROB = 1/50
        self.HORIZON = 1000
        self.MOVEFEAT = 5
        self.SHAREFEAT = 2 * int(self.SHARE)
        self.ATTACKFEAT = 1 * int(self.ATTACK)
        self.ENT_DIM = 16 + 2 * (self.MOVEFEAT + self.ATTACKFEAT + self.SHAREFEAT)
        self.ENT_DIM_GLOBAL = self.ENT_DIM + (self.MOVEFEAT + self.ATTACKFEAT + self.SHAREFEAT)
        self.TIMEOUT = 0
        self.DEVICE_OPTIMIZER = 'cpu'
        self.MAP = '_eldorado'
        self.stepsPerEpoch = 4000
        self.LR = 0.0005
        self.SW = 'sum'  # sum, min
        self.GAMMA = 0.99
        self.COMMON = False
        self.n_step_ret = 10_000


class SimpleMap(Experiment):
    def defaults(self):
        super().defaults()
        self.MELEERANGE = self.RNG
        self.RANGERANGE = 0
        self.MAGERANGE = 0

    def vamp(self, ent, targ, dmg):
        n_food = min(targ.food.val, dmg)
        n_water = min(targ.water.val, dmg)
        targ.food.decrement(amt=n_food)
        targ.water.decrement(amt=n_water)
        ent.food.increment(amt=n_food)
        ent.water.increment(amt=n_water)

    def MELEEDAMAGE(self, ent, targ):
        dmg = self.DMG
        if np.random.random() < self.DOUBLE_DMG_PROB:
            dmg *= 2
        targ.applyDamage(dmg)
        self.vamp(ent, targ, self.VAMPYR)
        return dmg

    def RANGEDAMAGE(self, ent, targ):
        return 0

    def MAGEDAMAGE(self, ent, targ):
        return 0

    def SHAREWATER(self, ent, targ):
        ent.giveResources(targ, ent.shareWater, 0)
        return ent.shareWater

    def SHAREFOOD(self, ent, targ):
        ent.giveResources(targ, 0, ent.shareFood)
        return ent.shareFood
