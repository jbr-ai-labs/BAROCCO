import os
import os.path as osp
import pickle
import sys
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

import experiments
import logs as loglib
from source.env.lib.enums import Neon
from source.env.lib.log import InkWell
import seaborn as sns
sns.set()


def plot(x, idxs, label, idx, path):
    colors = Neon.color12()
    c = colors[idx % 12]
    loglib.plot(x, inds=idxs, label=str(idx), c=c.norm)
    loglib.godsword()
    loglib.save(path + label + '.png')
    plt.close()


def plots(ticks, x, label, path, split):
    colors = Neon.color12()
    for idx, item in enumerate(x.items()):
        annID, val = item
        c = colors[idx % 12]
        idxs, val = compress(val, split)
        _, tcks = compress(ticks[idx], split)
        loglib.plot(val, inds=idxs, label=str(annID), c=c.norm)
    loglib.godsword()
    loglib.save(path + label + '.png')
    plt.close()


def compress(x, split):
    rets, idxs = [], []
    if split == 'train':
        n = 1 + len(x) // 25
    else:
        n = 1 + len(x) // 25
    for idx in range(0, len(x) - n, n):
        rets.append(np.mean(x[idx:(idx + n)]))
        idxs.append(idx)
    return 10 * np.array(idxs), rets


def popPlots(popLogs, path, split):
    idx = 0
    print(path)

    ticks = popLogs.pop('tick')
    for key, val in popLogs.items():
        print(key)
        plots(ticks, val, str(key), path, split)
        idx += 1


def flip(popLogs):
    ret = defaultdict(dict)
    for annID, logs in popLogs.items():
        for key, log in logs.items():
            if annID not in ret[key]:
                ret[key][annID] = []
            if type(log) != list:
                ret[key][annID].append(log)
            else:
                ret[key][annID] += log
    return ret


def group(blobs, idmaps):
    rets = defaultdict(list)
    for blob in blobs:
        groupID = idmaps[blob.annID]
        rets[groupID].append(blob)
    return rets


def mergePops(blobs, idMap):
    blobs = group(blobs, idMap)
    pops = defaultdict(list)
    for groupID, blobList in blobs.items():
        pops[groupID] += list(blobList)
    return pops


def individual(blobs, logDir, name, accum, split):
    savedir = logDir + name + '/' + split + '/'
    if not osp.exists(savedir):
        os.makedirs(savedir)

    blobs = mergePops(blobs, accum)
    minLength = min([len(v) for v in blobs.values()])
    blobs = {k: v[:minLength] for k, v in blobs.items()}
    popLogs = {}
    for annID, blobList in blobs.items():
        logs, blobList = {}, list(blobList)
        logs = {**logs, **InkWell.lifetime(blobList)}
        logs = {**logs, **InkWell.reward(blobList)}
        logs = {**logs, **InkWell.value(blobList)}
        logs = {**logs, **InkWell.tick(blobList)}
        logs = {**logs, **InkWell.contact(blobList)}
        logs = {**logs, **InkWell.attack(blobList)}
        popLogs[annID] = logs
    popLogs = flip(popLogs)
    popLogs = prepare_avg(popLogs, 'reward')
    popPlots(popLogs, savedir, split)


def prepare_avg(dct, key):
    dct[key + '_avg'] = {}
    lst = list(dct[key].values())
    length = min([len(lst[i]) for i in range(len(lst))])
    for i in range(len(lst)):
        lst[i] = lst[i][:length]
    dct[key + '_avg'][0] = np.mean(lst, axis=0)
    return dct


def makeAccum(config, form='single'):
    assert form in 'pops single split'.split()
    if form == 'pops':
        return dict((idx, idx) for idx in range(config.NPOP))
    elif form == 'single':
        return dict((idx, 0) for idx in range(config.NPOP))
    elif form == 'split':
        pop1 = dict((idx, 0) for idx in range(config.NPOP1))
        pop2 = dict((idx, 0) for idx in range(config.NPOP2))
        return {**pop1, **pop2}


if __name__ == '__main__':
    logDir = 'resource/exps/'
    logName = '/model/logs.p'
    for name, config in experiments.exps.items():
        try:
            with open(logDir + name + logName, 'rb') as f:
                dat = []
                idx = 0
                while True:
                    idx += 1
                    try:
                        dat += pickle.load(f)
                    except EOFError as e:
                        break
                print('Blob length: ', idx)
                split = 'test' if config.TEST else 'train'
                accum = makeAccum(config, 'pops')
                individual(dat, logDir, name, accum, split)
                print('Log success: ', name)
        except Exception as err:
            print(str(err))
