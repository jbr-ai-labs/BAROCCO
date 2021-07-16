import ray
import torch

from source.env import core, entity
from source.env.core.spawn import Spawner


class ActionArgs:
    def __init__(self, action, args):
        self.action = action
        self.args = args


class Realm:
    def __init__(self, config, args, idx):
        self.world, self.desciples = core.Env(config, idx), {}
        self.config, self.args = config, args
        self.npop = config.NPOP

        self.env = self.world.env
        self.values = None
        self.idx = idx

    def clientData(self):
        if self.values is None and hasattr(self, 'sword'):
            self.values = self.sword.anns[0].visVals()

        ret = {
            'environment': self.world.env,
            'entities': dict((k, v.packet()) for k, v in self.desciples.items()),
            'values': self.values
        }
        return ret

    def spawn(self):
        if len(self.desciples) >= self.config.NENT:
            return

        spawned_lst = self.spawner.spawn()
        for entID, color in spawned_lst:
            ent = entity.Player(entID, color, self.config)
            self.desciples[ent.entID] = ent

            r, c = ent.pos
            self.world.env.tiles[r, c].addEnt(entID, ent)
            self.world.env.tiles[r, c].counts[ent.colorInd] += 1

    def cullDead(self, dead):
        self.spawner.decrementTimeout()
        for ent in dead:
            entID = ent.entID
            r, c = ent.pos
            self.world.env.tiles[r, c].delEnt(entID)
            self.spawner.cull(ent.annID)
            del self.desciples[entID]

    def stepEnv(self):
        self.world.env.step()
        self.env = self.world.env.np()

    def stepEnt(self, ent, action, arguments):
        if self.config.ATTACK:
            if self.config.SHARE:
                move, attack, _, _ = action
                moveArgs, attackArgs, _, _ = arguments
            else:
                move, attack = action
                moveArgs, attackArgs = arguments
            ent.move = ActionArgs(move, moveArgs)
            ent.attack = ActionArgs(attack, attackArgs[0])
        else:
            ent.move = ActionArgs(action[0], arguments[0])

    def getStim(self):
        return self.world.env.stim()


@ray.remote
class NativeRealm(Realm):
    def __init__(self, agent, config, args, idx):
        super().__init__(config, args, idx)
        self.spawner = Spawner(config, args)
        self.controller = agent.Controller(config, args, idx)
        self.controller.anns[0].world = self.world
        self.logs = []
        self.stepCount = 0

    def stepEnts(self):
        returns = []
        desciples = self.desciples.values()

        for ent in desciples:
            ent.step(self.world)

        dead = self.funeral(desciples)
        n_dead = len(dead) if self.config.COMMON else 0
        self.cullDead(dead)
        self.spawn()

        desciples = list(self.desciples.values())
        state = self.controller.prepareInput(desciples[0], desciples[1], self.world.env) if desciples[0].annID == 0 else \
            self.controller.prepareInput(desciples[1], desciples[0], self.world.env)
        stim = self.getStim()

        for ent in desciples:
            action, arguments = self.controller.decide(ent, state, stim, n_dead)
            returns.append((action, arguments))

        for i, ent in enumerate(desciples):
            action, arguments = returns[i]
            ent.act(self.world, action, arguments)
            self.stepEnt(ent, action, arguments)

        [memory.push() for memory in self.controller.ReplayMemory]
        self.controller.ReplayMemoryLm.push()

        self.controller.config.EPS_CUR = max(self.controller.config.EPS_MIN, self.controller.config.EPS_CUR * self.controller.config.EPS_STEP)
        if self.config.NOISE:
            [self.controller.reset_noise() for _ in range(self.idx + 1)]

        for ent in dead:
            self.controller.collectRollout(ent.entID, self.controller.tick)

    def funeral(self, desciples):
        dead = []
        for i, ent in enumerate(desciples):
            if not ent.alive or ent.kill:
                dead.append(ent)
        n_dead = len(dead) if self.config.COMMON else 1
        for ent in dead:
            reward = self.controller.getReward(ent.timeAlive, n_dead)
            self.controller.collectStep(ent.entID, ent.annID, torch.zeros((1, self.config.ENT_DIM)).float(),
                                        torch.zeros((1, 1), dtype=torch.int64), reward, True)
        return dead

    def step(self):
        self.stepEnv()
        self.stepEnts()
        self.controller.tick += 1

    def run(self, swordUpdate=None):
        self.recvControllerUpdate(swordUpdate)

        updates, updates_lm, logs = None, None, None
        self.stepCount = 0
        while updates is None:
            self.stepCount += 1
            self.step()
            if self.config.TEST:
                updates, updates_lm, logs = (None, None), (None, None), None
            else:
                updates, updates_lm, logs = self.controller.sendUpdate()
        return self.idx, updates, updates_lm, logs

    def recvControllerUpdate(self, update):
        if update is None:
            return
        self.controller.recvUpdate(update)