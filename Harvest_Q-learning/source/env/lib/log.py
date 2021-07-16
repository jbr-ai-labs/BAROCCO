import os
import pickle
import time

import numpy as np


# Static blob analytics


class InkWell:
    def lifetime(blobs):
        return {'lifetime': [blob.lifetime for blob in blobs]}

    def reward(blobs):
        return {'reward': [blob.reward for blob in blobs]}

    def value(blobs):
        return {'value': [blob.value for blob in blobs]}

    def apples(blobs):
        return {'apples': [blob.apples for blob in blobs]}

    def lmValue(blobs):
        return {'lmValue': [blob.lmValue for blob in blobs]}

    def lmPunishment(blobs):
        return {'lmPunishment': [blob.lmPunishment for blob in blobs]}

    def attack(blobs):
        return {'attack': [blob.attack * 1000 for blob in blobs]}

    def shareFood(blobs):
        return {'shareFood': [blob.shareFood for blob in blobs]}

    def shareWater(blobs):
        return {'shareWater': [blob.shareWater for blob in blobs]}

    def contact(blobs):
        return {'contact': [blob.contact for blob in blobs]}

    def tick(blobs):
        return {'tick': [blob.tick for blob in blobs]}


# Agent logger
class Blob:
    def __init__(self):
        self.reward, self.ret = [], []
        self.value = []
        self.lmPunishment = []
        self.apples = []

    def finish(self):
        self.lifetime = len(self.reward)
        self.reward = np.sum(self.reward)
        self.apples = 0 if len(self.apples) == 0 else np.sum(self.apples)
        self.value = np.mean(self.value)
        self.lmPunishment = np.mean(self.lmPunishment)

class BlobLight:
    def __init__(self):
        self.reward, self.ret = [], []
        self.value = []
        self.lmPunishment = []
        self.apples = []


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
        rewards, blobs = [], []
        for blobList in logs:
            blobs += blobList
            for blob in blobList:
                rewards.append(float(blob.lifetime))

        self.lifetime = np.mean(rewards)
        blobRet = []
        for e in blobs:
            blobRet.append(e)
        self.save(blobRet)

    def latest(self):
        return self.lifetime

    def save(self, blobs, name='logs.p'):
        with open(self.dir + name, 'ab') as f:
            pickle.dump(blobs, f)

    def scratch(self):
        pass
