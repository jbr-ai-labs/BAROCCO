import numpy as np

from source.env.lib.log import Blob


def discountRewards(rewards, gamma=0.9999):
    rets, N = [], len(rewards)
    discounts = np.array([gamma ** i for i in range(N)])
    rewards = np.array(rewards)
    for idx in range(N): rets.append(sum(rewards[idx:] * discounts[:N - idx]))
    return rets


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
        self.rets = discountRewards(self.rewards, self.config.GAMMA)
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
