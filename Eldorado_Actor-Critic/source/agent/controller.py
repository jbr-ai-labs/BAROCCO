from collections import defaultdict, deque

import numpy as np
import torch

from source.env.action.tree import ActionTree
from source.env.action.v2 import ActionV2
from source.networks import COMA
from source.networks.ann import ANN
from source.networks.utils import checkTile
from source.utils.rollouts import Rollout
from source.utils.torch.param import setParameters, zeroGrads
from source.env.stim import stats, actions_agent


class Controller:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.nANN, self.h = config.NPOP, config.HIDDEN
        self.anns = [ANN(config, args) for _ in range(self.nANN)]

        self.updates, self.rollouts = defaultdict(lambda: Rollout(config)), {}
        self.rets = defaultdict(deque)
        self.states = defaultdict(deque)
        self.states_global = defaultdict(deque)
        self.policy = defaultdict(deque)
        self.actions = defaultdict(deque)
        self.rewards = defaultdict(deque)
        self.states_next = defaultdict(deque)
        self.dead = defaultdict(deque)
        self.buffer = None

        self.social = [COMA(config) for _ in range(config.NPOP)]

    def collect_buffer(self):
        self.blobs = [r.feather.blob for r in self.rollouts.values()]
        self.rollouts = {}
        length = min([len(self.rets[i]) for i in range(self.nANN)])
        if length == 0:
            return
        self.initBuffer()
        buffer = defaultdict(lambda: {'policy': [], 'action': [], 'states': [], 'states_global': [], 'return': [],
                                      'reward': [], 'states_next': [], 'dead': []})
        for _ in range(length):
            for i in range(self.nANN):
                buffer[i]['states'].append(self.states[i].popleft())
                buffer[i]['states_global'].append(self.states_global[i].popleft())
                buffer[i]['return'].append(self.rets[i].popleft())
                buffer[i]['policy'].append(self.policy[i].popleft().detach().numpy())
                buffer[i]['action'].append(self.actions[i].popleft().detach().numpy())
                buffer[i]['reward'].append(self.rewards[i].popleft())
                buffer[i]['dead'].append(self.dead[i].popleft())
                buffer[i]['states_next'].append(self.states_next[i].popleft())
        for i in range(self.nANN):
            self.buffer[i]['states'] = np.asarray(buffer[i]['states'], dtype=np.float32)
            self.buffer[i]['states_global'] = np.asarray(buffer[i]['states_global'], dtype=np.float32)
            self.buffer[i]['return'] = np.asarray(buffer[i]['return'], dtype=np.float32)
            self.buffer[i]['policy'] = np.asarray(buffer[i]['policy'], dtype=np.float32)
            self.buffer[i]['action'] = np.asarray(buffer[i]['action'], dtype=np.float32)
            self.buffer[i]['reward'] = np.asarray(buffer[i]['reward'], dtype=np.float32)
            self.buffer[i]['dead'] = np.asarray(buffer[i]['dead'], dtype=np.float32)
            self.buffer[i]['states_next'] = np.asarray(buffer[i]['states_next'], dtype=np.float32)

    def sendLogUpdate(self):
        blobs = self.blobs
        self.blobs = []
        return blobs

    def sendUpdate(self):
        if self.buffer is None:
            return None, None
        buffer = self.dispatchBuffer()
        return buffer, self.sendLogUpdate()

    def recvUpdate(self, update):
        update, update_lm = update
        for idx, paramVec in enumerate(update):
            setParameters(self.anns[idx], paramVec)
            zeroGrads(self.anns[idx])
        for idx, paramVec in enumerate(update_lm):
            setParameters(self.social[idx], paramVec)
            zeroGrads(self.social[idx])

    def collectStep(self, entID, action, policy, state, reward, contact, val):
        self.updates[entID].step(action, policy, state, reward, contact, val)

    def collectStepGlobal(self, entID, state_global):
        self.updates[entID].stepGlobal(state_global)

    def collectRollout(self, entID, ent, tick):
        assert entID not in self.rollouts
        rollout = self.updates[entID]
        rollout.finish()
        self.rets[ent.annID] += rollout.rets
        self.states[ent.annID] += rollout.states[:-1]
        self.states_global[ent.annID] += rollout.states_global
        self.policy[ent.annID] += rollout.policy[:-1]
        self.actions[ent.annID] += rollout.actions[:-1]
        self.rewards[ent.annID] += rollout.rewards[1:]
        self.states_next[ent.annID] += rollout.states_global[1:] + [np.zeros_like(rollout.states_global[-1])]
        self.dead[ent.annID] += [False for _ in range(len(rollout.states) - 2)] + [True]
        rollout.feather.blob.tick = tick
        self.rollouts[entID] = rollout
        del self.updates[entID]

        if min([len(self.rets[i]) for i in range(self.nANN)]) >= self.config.stepsPerEpoch // 2:
            self.collect_buffer()

    def initBuffer(self):
        self.buffer = defaultdict(dict)

    def dispatchBuffer(self):
        buffer = self.buffer
        self.buffer = None
        return buffer

    def getActionArguments(self, annReturns, stim, ent):
        actions = ActionTree(stim, ent, ActionV2).actions()
        move, attkShare = actions
        playerActions = [move]
        actionDecisions = {}
        moveAction = int(annReturns['actions'])
        attack = moveAction > 4
        if attack:
            moveAction -= 5
        actionTargets = [move.args(stim, ent, self.config)[moveAction]]

        action = attkShare.args(stim, ent, self.config)['attack']
        targets = action.args(stim, ent, self.config)
        target, decision = checkTile(ent, int(attack), targets)
        playerActions.append(action), actionTargets.append([target])
        actionDecisions['attack'] = decision

        return playerActions, actionTargets, actionDecisions

    def decide(self, ent, state, stim, n_dead=0):
        entID, annID = ent.entID, ent.annID
        reward = 1 if ent.timeAlive > self.config.HORIZON else self.config.STEPREWARD + self.config.DEADREWARD * n_dead

        annReturns = self.anns[annID](state.unsqueeze(0), train=True)

        playerActions, actionTargets, actionDecisions = self.getActionArguments(annReturns, stim, ent)

        moveAction = int(annReturns['actions'])
        attack = actionDecisions.get('attack', None)
        if moveAction > 4:
            moveAction -= 5
        ent.moveDec = moveAction
        contact = int(attack is not None)

        self.collectStep(entID, annReturns['actions'], annReturns['policy'], state.numpy(), reward, contact,
                         float(annReturns['val']))
        self.updates[entID].feather.scrawl(ent, float(annReturns['val']), reward, attack, contact)
        return playerActions, actionTargets

    def prepareInput(self, ent, other, env):
        state = stats(ent, other, env, self.config)
        state = torch.from_numpy(state).float().view(-1)
        return state

    def prepareGlobalInput(self, state, ent):
        actions = np.array(actions_agent(ent, self.config)).reshape(-1)
        state = np.append(state, actions, 0)
        return state
