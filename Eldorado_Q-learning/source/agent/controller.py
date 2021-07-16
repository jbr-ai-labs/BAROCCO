from collections import defaultdict
import torch

from source.env.action.tree import ActionTree
from source.env.action.v2 import ActionV2
from source.env.stim import stats
from source.networks.ann import ANN
from source.networks.social import SocialANN
from source.utils.rollouts import Rollout
from source.utils.torch.replay import ReplayMemory, ReplayMemoryLm
from source.utils.torch.forward import Forward
from source.networks.utils import checkTile


class Controller:
    def __init__(self, config, args, idx):
        self.config, self.args = config, args
        self.nANN, self.h = config.NPOP, config.HIDDEN
        self.anns = [ANN(config, args) for _ in range(self.nANN)]

        self.init, self.nRollouts = True, 32
        self.updates = defaultdict(lambda: Rollout(config))
        self.blobs = []
        self.idx = idx
        self.ReplayMemory = [ReplayMemory(self.config) for _ in range(self.nANN)]
        self.ReplayMemoryLm = ReplayMemoryLm(self.config)
        self.forward = Forward(config, args)
        self.buffer_size_to_send = 2 ** 7

        self.social = [SocialANN(args, config) for _ in range(self.nANN)]
        self.tick = 0
        self.logTick = 0

    def sendBufferUpdate(self):
        for replay in self.ReplayMemory:
            if len(replay) < self.buffer_size_to_send:
                return None, None
        buffer = [replay.send_buffer() for replay in self.ReplayMemory]
        priorities = [self.forward.get_priorities_from_samples(buf, ann, ann, lm) for buf, ann, lm in
                      zip(buffer, self.anns, self.social)] if self.config.REPLAY_PRIO else [None] * len(self.anns)
        bufferLm = self.ReplayMemoryLm.send_buffer()
        return (buffer, priorities), bufferLm

    def sendLogUpdate(self):
        self.logTick += 1
        if ((self.logTick + 1) % 2 ** 6) == 0:
            blobs = self.blobs
            self.blobs = []
            return blobs
        return None

    def sendUpdate(self):
        recvs, recvs_lm = self.sendBufferUpdate()
        logs = self.sendLogUpdate() if recvs is not None else None
        return recvs, recvs_lm, logs

    def recvUpdate(self, update):
        update, update_lm = update
        if update is not None:
            self.loadAnnsFrom(update)
        if update_lm is not None:
            self.loadLmFrom(update_lm)

    def collectStep(self, entID, annID, s, actions, reward, dead):
        self.ReplayMemory[annID].append(entID, s, actions, reward, dead)
        self.ReplayMemoryLm.append(entID, annID, s, actions, reward, dead)

    def collectRollout(self, entID, tick):
        rollout = self.updates[entID]
        rollout.feather.blob.tick = tick
        rollout.finish()
        self.blobs.append(rollout.feather.blob)
        del self.updates[entID]

    def decide(self, ent, state, stim, n_dead=0):
        entID, annID = ent.entID, ent.annID
        reward = self.getReward(ent.timeAlive, n_dead)

        outputsLm, punishmentsLm = self.social[annID](state)
        atnArgs = self.anns[annID](state, self.config.EPS_CUR, punishmentsLm)
        action, arguments, decs = self.actionTree(ent, stim, atnArgs)

        attack = decs.get('attack', None)
        shareFood = decs.get('shareFood', None)
        shareWater = decs.get('shareWater', None)
        contact = int(attack is not None)

        ent.moveDec = self.getMoveAction(atnArgs)
        if contact:
            ent.shareFoodDec = shareFood
            ent.shareWaterDec = shareWater
            ent.attackDec = attack

        self.collectStep(entID, annID, state, atnArgs[1], reward, False)
        avgPunishmentLm = calcAvgPunishment(atnArgs, outputsLm)
        self.updates[entID].feather.scrawl(ent, avgPunishmentLm, reward, attack, contact)
        return action, arguments

    def prepareInput(self, ent, other, env):
        state = stats(ent, other, env, self.config)
        state = torch.from_numpy(state).float().view(-1).unsqueeze(0)
        return state

    def getReward(self, timeAlive, n_dead):
        return 1 if timeAlive > self.config.HORIZON else self.config.STEPREWARD + self.config.DEADREWARD * n_dead

    def actionTree(self, ent, env, outputs):
        actions = ActionTree(env, ent, ActionV2).actions()
        move, attkShare = actions

        playerActions = [move]
        actionTargets = [move.args(env, ent, self.config)[self.getMoveAction(outputs)]]

        actionDecisions = {}
        for tpl in [('attack', self.config.ATTACK, self.getAttackAction),
                    ('shareWater', self.config.SHARE, self.getShareWaterAction),
                    ('shareFood', self.config.SHARE, self.getShareFoodAction)]:
            if tpl[1]:
                action = attkShare.args(env, ent, self.config)[tpl[0]]
                targets = action.args(env, ent, self.config)
                target, decision = checkTile(ent, tpl[2](outputs), targets)
                playerActions.append(action), actionTargets.append([target])
                actionDecisions[tpl[0]] = decision
        return playerActions, actionTargets, actionDecisions

    def getMoveAction(self, outputs):
        return int(outputs[1] % self.config.MOVEFEAT)

    def getAttackAction(self, outputs):
        if self.config.ATTACK:
            return int(int(outputs[1] // self.config.MOVEFEAT) % (2 ** self.config.ATTACKFEAT))
        return None

    def getShareWaterAction(self, outputs):
        if self.config.SHARE:
            return int(int(outputs[1] // (self.config.MOVEFEAT * (2 ** self.config.ATTACKFEAT))) %
                       (2 ** int(self.config.SHAREFEAT / 2)))
        return None

    def getShareFoodAction(self, outputs):
        if self.config.SHARE:
            return int(int(outputs[1] // (self.config.MOVEFEAT * (2 ** self.config.ATTACKFEAT) *
                                          (2 ** int(self.config.SHAREFEAT / 2)))) %
                       (2 ** int(self.config.SHAREFEAT / 2)))
        return None

    def reset_noise(self):
        nets = self.anns + self.social
        for net in nets:
            net.reset_noise()

    def loadAnnsFrom(self, states):
        [ann.load_state_dict(state) for ann, state in zip(self.anns, states)]

    def loadLmFrom(self, states):
        [lm.load_state_dict(state) for lm, state in zip(self.social, states)]


def calcAvgPunishment(atnArgs, punishmentsLm):
    return punishmentsLm.mean(2).view((-1,))[int(atnArgs[1])].detach().numpy() - punishmentsLm.mean().detach().numpy()
