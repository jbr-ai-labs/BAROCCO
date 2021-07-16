from source.env.lib import utils


# Adjacency functions
def adjacentDeltas():
    return [(-1, 0), (1, 0), (0, 1), (0, -1)]


def adjacentPos(pos):
    return [posSum(pos, delta) for delta in adjacentDeltas()]


def adjacentMats(env, pos):
    return [type(env.tiles[p].mat) for p in adjacentPos(pos) if utils.inBounds(*p, env.shape)]


def posSum(pos1, pos2):
    return pos1[0] + pos2[0], pos1[1] + pos2[1]
