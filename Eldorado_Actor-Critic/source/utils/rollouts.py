import numpy as np

from source.env.lib.log import Blob


def discountRewardsTD(rewards, vals, gamma=0.99, nSteps=1):
    N = len(rewards) - 1
    rets = np.zeros(N)
    rets[-1] = rewards[-1]
    for idx in reversed(range(N - 1)):
        nStepsCur = min(nSteps, N - 1 - idx)
        ret = [rewards[idx + i + 1] * gamma ** i for i in range(nStepsCur)]
        rets[idx] = sum(ret) + gamma ** nStepsCur * vals[idx + nStepsCur]
    return list(rets)


class Rollout:
    def __init__(self, config):
        self.config = config
        self.actions = []
        self.policy = []
        self.states = []
        self.states_global = []
        self.rewards = []
        self.rets = []
        self.vals = []
        self.states = []
        self.contacts = []
        self.feather = Feather()

    def step(self, action, policy, state=None, reward=None, contact=None, val=None):
        self.actions.append(action)
        self.policy.append(policy)
        self.states.append(state)
        self.rewards.append(reward)
        self.vals.append(val)
        self.contacts.append(contact)

    def stepGlobal(self, state_global):
        self.states_global.append(state_global)

    def finish(self):
        self.lifespan = len(self.rewards)
        self.rets = discountRewardsTD(self.rewards, self.vals, self.config.GAMMA)
        self.feather.finish()


# Rollout logger
class Feather:
    def __init__(self):
        self.blob = Blob()

    def scrawl(self, ent, val, reward, attack, contact):
        self.blob.annID = ent.annID
        self.stats(val, reward, attack, contact)

    def stats(self, value, reward, attack, contact):
        self.blob.reward.append(reward)
        self.blob.value.append(float(value))
        self.blob.contact.append(float(contact))
        if attack is not None:
            self.blob.attack.append(float(attack))

    def finish(self):
        self.blob.finish()
