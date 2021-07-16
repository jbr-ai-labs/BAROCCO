import os
import pickle
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np

from source.env.lib.enums import Material


# Static blob analytics

# Untested function
def discountRewards(rewards, gamma=0.99):
    rets, N = [], len(rewards)
    discounts = np.array([gamma ** i for i in range(N)])
    rewards = np.array(rewards)
    for idx in range(N): rets.append(sum(rewards[idx:] * discounts[:N - idx]))
    return rets


class InkWell:

    def lifetime(blobs):
        return {'lifetime': [blob.lifetime for blob in blobs]}

    def reward(blobs):
        return {'reward': [blob.reward for blob in blobs]}

    def value(blobs):
        return {'value': [blob.value for blob in blobs]}

    def attack(blobs):
        return {'attack': [blob.attack for blob in blobs]}

    def tick(blobs):
        return {'tick': [blob.tick for blob in blobs]}

    def contact(blobs):
        return {'contact': [blob.contact for blob in blobs]}


# Agent logger
class Blob:
    def __init__(self):
        self.reward = []
        self.value = []
        self.attack, self.contact = [], []

    def finish(self):
        self.lifetime = len(self.reward)
        self.reward = discountRewards(self.reward[1:])[0] if self.lifetime > 0 else 0
        self.contact = np.mean(self.contact)
        self.attack = 0 if len(self.attack) == 0 else np.mean(self.attack)
        self.value = np.mean(self.value)


class Quill:
    def __init__(self, modeldir):
        self.time = time.time()
        self.dir = modeldir
        self.index = 0
        try:
            os.remove(modeldir + 'logs.p')
        except:
            pass

    def timestamp(self):
        cur = time.time()
        ret = cur - self.time
        self.time = cur
        return str(ret)

    def print(self):
        print(
            'Time: ', self.timestamp(),
            ', Iter: ', str(self.index))

    def scrawl(self, logs):
        # Collect log update
        self.index += 1
        rewards, blobs = [], logs
        returns = 0
        for blob in logs:
            returns += float(blob.reward)
            rewards.append(float(blob.lifetime))

        returns /= len(logs)
        self.lifetime = np.mean(rewards)
        blobRet = []
        for e in blobs:
            if np.random.uniform() < 0.1:
                blobRet.append(e)
        self.save(blobRet)
        return returns

    def latest(self):
        return self.lifetime

    def save(self, blobs, name='logs.p'):
        with open(self.dir + name, 'ab') as f:
            pickle.dump(blobs, f)
