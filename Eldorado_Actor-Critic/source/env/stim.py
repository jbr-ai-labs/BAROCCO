import numpy as np


def center(x, mmax):
    return (x - mmax / 2.0) / mmax


def stats(ent, other, env, config):
    ret = stats_agent(ent, config) + stats_agent(other, config)

    rSelf, cSelf = ent.pos
    rOther, cOther = other.pos
    rDelta, cDelta = rOther - rSelf, cOther - cSelf
    ret += [rDelta, cDelta]

    forests = env.getForestStatus()
    forests = [center(f, 5) for f in forests]
    ret += forests

    ret = np.array(ret)
    return ret


def stats_agent(ent, config):
    health = ent.health.center()
    food = ent.food.center()
    water = ent.water.center()
    lifetime = center(ent.timeAlive, config.HORIZON)

    rSelf, cSelf = ent.pos
    r, c = center(rSelf, ent.R), center(cSelf, ent.C)
    ret = [lifetime, health, food, water, r, c]

    ret += actions_agent(ent, config)

    return ret


def actions_agent(ent, config):
    ret = list(one_hot(ent.moveDec, 5))
    if config.ATTACK:
        ret += [float(ent.attackDec)]
    if config.SHARE:
        ret += [float(ent.shareWaterDec), float(ent.shareFoodDec)]
    return ret


def one_hot(v, n):
    ary = np.zeros(n)
    ary[v] = 1.
    return ary
