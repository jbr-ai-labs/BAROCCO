import numpy as np
from source.env.systems import ai
from source.env.lib.enums import Material


class Stat:
    def __init__(self, val, maxVal):
        self._val = val
        self._max = maxVal

    def increment(self, amt=1):
        self._val = min(self.max, self.val + amt)

    def decrement(self, amt=1):
        self._val = max(0, self.val - amt)

    def center(self):
        return (self.val - self.max / 2.0) / self.max

    @property
    def val(self):
        return self._val

    @property
    def max(self):
        return self._max

    def packet(self):
        return {'val': self.val, 'max': self.max}


class Player:
    public = set(
        'pos lastPos R C food water health entID annID name colorInd color timeAlive kill attackMap damage freeze immune'.split())

    def __init__(self, entID, color, config):
        self._config = config
        self._annID, self._color = color
        self._colorInd = self._annID

        self._R, self._C = config.R, config.C
        self._pos = config.SPAWN()
        self._lastPos = self.pos

        self._food = Stat(config.FOOD, config.FOOD)
        self._water = Stat(config.WATER, config.WATER)
        self._health = Stat(config.HEALTH, config.HEALTH)

        self._entID = entID
        self._name = 'Alice' if self._colorInd == 0 else 'Bob'  # ''Neural_' + str(self._entID)
        self._timeAlive = 0

        self._damage = None
        self._freeze = 0
        self._immune = True
        self._kill = False

        self._index = 1
        self._immuneTicks = 1

        self.move = None
        self.attack = None
        self.shareWater, self.shareFood = 0, 0

        self._attackMap = np.zeros((7, 7, 3)).tolist()

        self.moveDec = 0
        self.attackDec = 0
        self.shareWaterDec = 0
        self.shareFoodDec = 0

    def __getattribute__(self, name):
        if name in Player.public:
            return getattr(self, '_' + name)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in Player.public:
            raise AttributeError('Property \"' + name + '\" is read only: agents cannot modify their server-side data')
        return super().__setattr__(name, value)

    def forage(self, world):
        r, c = self._pos
        isForest = type(world.env.tiles[r, c].mat) in [Material.FOREST.value]
        if isForest and world.env.harvest(r, c):
            self.food.increment(6 // len(world.env.tiles[r, c].ents))

        isWater = Material.WATER.value in ai.adjacentMats(world.env, self._pos)
        if isWater:
            self.water.increment(6)

    def lavaKill(self, world):
        r, c = self._pos
        if type(world.env.tiles[r, c].mat) == Material.LAVA.value:
            self._kill = True
        return self._kill

    def updateStats(self):
        if (self._food.val > self._food.max // 2 and
                self._water.val > self._water.max // 2):
            self._health.increment()

        self._water.decrement()
        self._food.decrement()

        if self._food.val <= 0:
            self._health.decrement()
        if self._water.val <= 0:
            self._health.decrement()

    def updateCounts(self, world):
        r, c = self._pos
        world.env.tiles[r, c].counts[self._colorInd] += 1

    def step(self, world):
        if not self.alive: return
        self._freeze = max(0, self._freeze - 1)
        self.updateCounts(world)

        if self.lavaKill(world): return
        self.forage(world)
        self.updateStats()

        self._damage = None
        self._timeAlive += 1
        if self._timeAlive > self._config.HORIZON:
            self._kill = True
        self.updateImmune()

    def act(self, world, actions, arguments):
        self.mapAttack()
        self._lastPos = self._pos
        for action, args in zip(actions, arguments):
            action.call(world, self, *args)

    @property
    def alive(self):
        return self._health.val > 0

    def getLifetime(self):
        return self._timeAlive

    def updateImmune(self):
        if self._timeAlive >= self._immuneTicks:
            self._immune = False

    # Note: does not stack damage, but still applies to health
    def applyDamage(self, damage):
        if self.immune:
            return
        self._damage = damage
        self._health.decrement(damage)

    def mapAttack(self):
        if self.attack is not None:
            attack = self.attack
            name = attack.action.__name__
            if name == 'Melee':
                attackInd = 0
            elif name == 'Range':
                attackInd = 1
            elif name == 'Mage':
                attackInd = 2
            rt, ct = attack.args.pos
            rs, cs = self._pos
            dr = rt - rs
            dc = ct - cs
            if abs(dr) <= 3 and abs(dc) <= 3:
                self._attackMap[3 + dr][3 + dc][attackInd] += 1

    def giveResources(self, ent, n_water=0, n_food=0):
        n_water = min(n_water, self._water._val, ent.water.max - ent.water.val)
        n_food = min(n_food, self._food._val, ent.food.max - ent.food.val)

        self._water.decrement(n_water)
        ent.water.increment(n_water)
        self._food.decrement(n_food)
        ent.food.increment(n_food)
