import ray
import torch

from source.env import entity, core
from source.env.core.spawn import Spawner


class ActionArgs:
    def __init__(self, action, args):
        self.action = action
        self.args = args


class Env:
    def __init__(self, config, args, idx):
        self.world, self.desciples = core.Env(config, idx), {}
        self.config, self.args, self.tick = config, args, 0
        self.npop = config.NPOP

        self.env = self.world.env
        self.values = None
        self.idx = idx

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
            ent = self.desciples[entID]
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
class NativeRealm(Env):
    def __init__(self, agent, config, args, idx):
        super().__init__(config, args, idx)
        self.spawner = Spawner(config, args)
        self.controller = agent.Controller(config, args)
        self.stepCount = 0

    def stepEnts(self):
        desciples = self.desciples.values()
        returns = []

        for ent in desciples:
            ent.step(self.world)

        dead = self.funeral(desciples)
        n_dead = len(dead) if self.config.COMMON else 0
        self.cullDead(dead)
        self.spawn()

        desciples = list(self.desciples.values())
        desciples = sorted(desciples, key=lambda ent: ent.annID)
        state = self.controller.prepareInput(desciples[0], desciples[1], self.world.env) if desciples[0].annID == 0 else \
                self.controller.prepareInput(desciples[1], desciples[0], self.world.env)
        stim = self.getStim()

        for ent in desciples:
            playerActions, playerTargets = self.controller.decide(ent, state, stim, n_dead)
            returns.append((playerActions, playerTargets))

        for i, ent in enumerate(desciples):
            ent.act(self.world, returns[i][0], returns[i][1])
            self.stepEnt(ent, returns[i][0], returns[i][1])

        for i in range(len(desciples)):
            ent, other = desciples[i], desciples[1 - i]
            global_state = self.controller.prepareGlobalInput(state.numpy(), other)
            self.controller.collectStepGlobal(ent.entID, global_state)

        for ent in dead:
            self.controller.collectRollout(ent.entID, ent, self.tick)

    def funeral(self, desciples):
        dead = []
        for i, ent in enumerate(desciples):
            if not ent.alive or ent.kill:
                dead.append(ent)

        n_dead = len(dead) if self.config.COMMON else 1
        for ent in dead:
            stim = self.getStim()
            self.controller.decide(ent, torch.zeros(self.config.ENT_DIM).float(), stim, n_dead)
        return dead

    def step(self):
        self.spawn()
        self.stepEnv()
        self.stepEnts()
        self.tick += 1

    def run(self, update=None):
        self.recvControllerUpdate(update)

        buffer = None
        while buffer is None:
            self.step()
            buffer, logs = self.controller.sendUpdate()
        return self.idx, buffer, logs

    def recvControllerUpdate(self, update):
        if update is None:
            return
        self.controller.recvUpdate(update)
