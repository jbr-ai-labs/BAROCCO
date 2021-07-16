import numpy as np


# Untested function
from source.env.lib.log import Blob


def discountRewards(rewards, gamma=0.99):
    rets, N = [], len(rewards)
    discounts = np.array([gamma ** i for i in range(N)])
    rewards = np.array(rewards)
    for idx in range(N): rets.append(sum(rewards[idx:] * discounts[:N - idx]))
    return rets


class Rollout:
    def __init__(self):
        self.atnArgs = []
        self.vals = []
        self.rewards = []
        self.states = []
        self.feather = Feather()

    def step(self, atnArgs, val, reward, stim=None):
        self.atnArgs.append(atnArgs)
        self.vals.append(val)
        self.states.append(stim)
        self.rewards.append(reward)

    def finish(self):
        self.lifespan = len(self.rewards)
        self.feather.finish()


# Rollout logger
class Feather:
    def __init__(self):
        self.blob = Blob()

    def scrawl(self, apple, annID, reward):
        self.blob.annID = annID
        self.stats(reward, apple)

    def stats(self, reward, apple):
        self.blob.reward.append(reward)
        self.blob.apples.append(apple)

    def finish(self):
        self.blob.finish()
