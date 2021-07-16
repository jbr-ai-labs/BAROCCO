import numpy as np

from source.env.core.config import Config


class Experiment(Config):
    def defaults(self):
        super().defaults()
        self.MODELDIR = 'resource/logs/'
        self.HIDDEN = 64
        self.TEST = False
        self.LOAD = False
        self.BEST = False
        self.NPOP = 2
        self.NENT = 2
        self.EPS_CUR = 1
        self.EPS_MIN = 0.
        self.EPS_STEP = 0.99999
        self.EXPLORE_MODE = 'eps'
        self.VAMPYR = 1
        self.RNG = 1  # currently the agents can interact across the whole map
        self.DMG = 1
        self.DOUBLE_DMG_PROB = 1/50
        self.HORIZON = 1000
        self.MOVEFEAT = 5
        self.SHAREFEAT = 2
        self.ATTACKFEAT = 1
        self.TIMEOUT = 0
        self.STEPREWARD = 0.1
        self.DEADREWARD = -10.
        self.PUNISHMENT = 1
        self.MAP = '_eldorado'
        self.ATTACK = True
        self.SHARE = False
        self.ATTACKFEAT = self.ATTACKFEAT * int(self.ATTACK)
        self.SHAREFEAT = self.SHAREFEAT * int(self.SHARE)
        self.N_OUTPUTS = self.MOVEFEAT * (2 ** self.ATTACKFEAT) * (2 ** self.SHAREFEAT)
        self.ENT_DIM = 17 + 2 * (self.MOVEFEAT + self.ATTACKFEAT + self.SHAREFEAT)
        self.LR = 0.0005
        self.NOISE = True
        self.LM_FUNCTION = 'sum'  # 'sum', 'min', 'max'
        self.QMIX = True
        self.REPLAY_LM = True
        self.BATCH_SIZE = 2**7
        self.BATCH_SIZE_LM = 2**5
        self.TARGET_PERIOD = 2**11
        self.BUFFER_SIZE = 2**19
        self.BUFFER_SIZE_LM = 2**12 if not self.REPLAY_LM else 2**18
        self.MIN_BUFFER = 2**13
        self.GAMMA = 0.99
        self.N_QUANT = 10
        self.REPLAY_PRIO = True
        self.REPLAY_NSTEP = 5
        self.REPLAY_NSTEP_LM = 1
        self.VANILLA_QMIX = False
        self.device = 'cpu'  # 'cpu' or 'cuda'
        self.COMMON = False


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
