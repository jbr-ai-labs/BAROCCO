from source.env.action import action
from source.env.lib import utils, enums


class Arg:
    def __init__(self, val, discrete=True, set=False, min=-1, max=1):
        self.val = val
        self.discrete = discrete
        self.continuous = not discrete
        self.min = min
        self.max = max
        self.n = self.max - self.min + 1


class ActionV2:
    def edges(self):
        return [Move, Attack]


class Pass(action.Pass):
    priority = 0

    @staticmethod
    def call(world, entity):
        return

    def args(stim, entity, config):
        return [()]

    @property
    def nArgs():
        return 1


class Move(action.Move):
    priority = 1

    def call(world, entity, rDelta, cDelta):
        r, c = entity.pos
        rNew, cNew = r + rDelta, c + cDelta
        if world.env.tiles[rNew, cNew].state.index in enums.IMPASSIBLE:
            return
        if not utils.inBounds(rNew, cNew, world.shape):
            return
        if entity.freeze > 0:
            return

        entity._pos = rNew, cNew
        entID = entity.entID

        r, c = entity.lastPos
        world.env.tiles[r, c].delEnt(entID)

        r, c = entity.pos
        world.env.tiles[r, c].addEnt(entID, entity)

    def args(stim, entity, config):
        rets = []
        for delta in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)):
            r, c = delta
            # r, c = Arg(r), Arg(c)
            rets.append((r, c))
        return rets

    @property
    def nArgs():
        return len(Move.args(None, None))


class Attack(action.Attack):
    def inRange(entity, stim, N, pos, border):
        R, C = pos
        # R -= border
        # C -= border
        rets = []
        for r in range(8):#R - N, R + N + 1):
            for c in range(8):#C - N, C + N + 1):
                for e in stim[r, c].ents.values():
                    rets.append(e)
        return rets

    def l1(pos, cent):
        r, c = pos
        rCent, cCent = cent
        return abs(r - rCent) + abs(c - cCent)

    def call(world, entity, targ, damageF, freeze=False, share=False):
        if entity.entID == targ.entID:
            if not share:
                entity._attack = None
            return
        # entity.targPos = targ.pos
        # entity.attkPos = entity.lastPos
        # entity.targ = targ
        damage = damageF(entity, targ)
        assert type(damage) == int
        if freeze and damage > 0:
            targ._freeze = 3
        return
        # return damage

    def args(stim, entity, config):
        return {Melee.name: Melee,
                Range.name: Range,
                Mage.name: Mage,
                ShareWater.name: ShareWater,
                ShareFood.name: ShareFood}


class Melee(action.Melee):
    name = 'attack'
    priority = 2

    def call(world, entity, targ):
        damageF = world.config.MELEEDAMAGE
        Attack.call(world, entity, targ, damageF)

    def args(stim, entity, config):
        return Attack.inRange(entity, stim, config.MELEERANGE, entity.pos, config.BORDER)


class Range(action.Range):
    name = 'range'
    priority = 2

    def call(world, entity, targ):
        damageF = world.config.RANGEDAMAGE
        Attack.call(world, entity, targ, damageF)

    def args(stim, entity, config):
        return Attack.inRange(entity, stim, config.RANGERANGE, entity.pos, config.BORDER)


class Mage(action.Mage):
    name = 'mage'
    priority = 2

    def call(world, entity, targ):
        damageF = world.config.MAGEDAMAGE
        dmg = Attack.call(world, entity, targ, damageF, freeze=True)

    def args(stim, entity, config):
        return Attack.inRange(entity, stim, config.MAGERANGE, entity.pos, config.BORDER)


class ShareWater():
    name = 'shareWater'
    priority = 2

    def call(world, entity, targ):
        damageF = world.config.SHAREWATER
        dmg = Attack.call(world, entity, targ, damageF, share=True)

    def args(stim, entity, config):
        return Attack.inRange(entity, stim, config.RNG, entity.pos, config.BORDER)


class ShareFood():
    name = 'shareFood'
    priority = 2

    def call(world, entity, targ):
        damageF = world.config.SHAREFOOD
        Attack.call(world, entity, targ, damageF, share=True)

    def args(stim, entity, config):
        return Attack.inRange(entity, stim, config.RNG, entity.pos, config.BORDER)
