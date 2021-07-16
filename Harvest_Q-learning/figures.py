import os
import os.path as osp
import pickle
import sys
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

import experiments
import logs as loglib
import seaborn as sns

from source.env.lib.enums import Neon
from source.env.lib.log import InkWell

sns.set()


def plot(x, idxs, label, idx, path):
    colors = Neon.color12()
    # loglib.dark()
    c = colors[idx % 12]
    loglib.plot(x, inds=idxs, label=str(idx), c=c.norm)
    loglib.godsword()
    loglib.save(path + label + '.png')
    plt.close()


def plots(m, x, label, ticks, color, linestyle, dashes):
    colors = Neon.color12()
    # loglib.dark()

    print(len(ticks))
    val = [xi[0] for xi in x]
    n_exps = len(val)
    c = colors[color]
    idxs, val = compress(val, m)
    tcks = compress_ticks(ticks[0][:min([len(xi[0]) for xi in x])], n_exps)
    print(len(val))
    print(len(tcks))
    loglib.plot(val, inds=tcks, label=label, c=c.norm, linestyle=linestyle, dashes=dashes)
    loglib.godsword()


def meanfilter(x, n=1):
    ret = []
    for idx in range(len(x) - n):
        val = np.mean(x[idx:(idx + n)])
        ret.append(val)
    return ret


def compress(x, m):
    rets, idxs = [], []
    length = min([len(xi) for xi in x])
    n = 1 + length // 50
    rng = np.arange(length)
    for idx in range(0, length, n):
        rets += [np.mean(xi[idx:(idx + n)]) for xi in x]
        idxs += len(rng[idx:(idx + n)]) * [np.mean(rng[idx:(idx + n)]) / 5]
    return m * 200 * np.array(idxs), rets


def compress_ticks(ticks, n_exps):
    rets = []
    n = 1 + len(ticks) // 50
    for idx in range(0, len(ticks), n):
        rets += n_exps * [np.mean(ticks[idx: idx + n]) / 5]
    return rets


def popPlots(popLogs, path, split):
    idx = 0
    print(path)

    mult = [1, 1, 1, 1, 1, 1, 1]
    ticks = [{k: v * mult[i] for k, v in popLog.pop('tick_avg')[0].items()} for i, popLog in enumerate(popLogs)]
    line_style = ['-', '--', '-.', ':', '--', '-.', ':']
    #labels = [r'selfish', r'BAROCCO, $\lambda=0.1$', r'BAROCCO, $\lambda=0.3$',
    #          r'BAROCCO, $\lambda=0.5$',
    #          r'BAROCCO, $\lambda=0.7$', r'BAROCCO, $\lambda=0.9$', r'BAROCCO, $\lambda=1$']
    labels = ['selfish', 'CRS, sum', 'CRS, min', 'BAROCCO, sum', 'BAROCCO, min', 'Vanilla COMA, sum',
              'Vanilla COMA, min']
    # labels = ['selfish', 'CRS, sum', 'BAROCCO, sum']
    #   labels = ['selfish', 'CRS, sum', 'BAROCCO, sum', 'BAROCCO, min', 'Vanilla COMA, sum']
    dashes = [None, None, None, None, (2, 4), (2, 5), (2, 6)]
    # mult = [1, 2, 9 / 6, 8 / 6, 1]
    trunc = [0, 0, 0, 0, 0, 0, 0, 0]
    skip = [2, 6]

    for key, val in popLogs[0].items():
        print(key)
        for i, popLog in enumerate(popLogs):
            if i in skip:
                continue
            plots(mult[i],
                  popLog[key],
                  labels[i], ticks[i], i, linestyle=line_style[i], dashes=dashes[i])

        #  plots(mult[i], {k: [va for j, va in enumerate(v) if j * mult[i] < minLength] for k, v in popLog[key].items()}, labels[i], idx, path, split, i, linestyle=line_style[i], dashes=dashes[i])
        idx += 1
        loglib.save(path + str(key) + '.png')
        plt.close()


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


def calcAvg(dict, key):
    popDict = prepare_avg(dict, key)
    popDict.pop(key)
    return popDict


def calcDiff(dict, key):
    popDict = prepare_diff(dict, key)
    return popDict


def individual(blobsMulti, logDir, name, accum, split):
    savedir = logDir + name + '/' + split + '/'
    if not osp.exists(savedir):
        os.makedirs(savedir)
    apples = True

    popLogs = []

    for blobMulti in blobsMulti:
        expLog = defaultdict(list)
        for blobs in blobMulti:
            blobs = mergePops(blobs, accum)
            minLength = min([len(v) for v in blobs.values()])
            blobs = {k: v[:minLength] for k, v in blobs.items()}
            popDict = {}
            for annID, blobList in blobs.items():
                logs, blobList = {}, list(blobList)
                logs = {**logs, **InkWell.apples(blobList)} if apples else {**logs, **InkWell.lifetime(blobList)}
                logs = {**logs, **InkWell.tick(blobList)}
                popDict[annID] = logs
            popDict = flip(popDict)
            popDict = calcDiff(popDict, 'apples') if apples else calcDiff(popDict, 'lifetime')
            for k in logs.keys():
                popDict = calcAvg(popDict, k)
            for k in popDict.keys():
                expLog[k].append(popDict[k])
        popLogs.append(expLog)
    popPlots(popLogs, savedir, split)


def prepare_avg(dct, key):
    dct[key + '_avg'] = {}
    lst = list(dct[key].values())
    length = min([len(lst[i]) for i in range(len(lst))])
    for i in range(len(lst)):
        lst[i] = lst[i][:length]
    dct[key + '_avg'][0] = np.sum(lst, axis=0)
    return dct


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.sort(array, axis=0)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1).reshape(-1, 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return (np.sum((2 * index - n - 1) * array, axis=0)) / (n * np.sum(array, axis=0)) * 2  # Gini coefficient


def prepare_diff(dct, key):
    dct[key + '_diff'] = {}
    lst = list(dct[key].values())
    length = min([len(lst[i]) for i in range(len(lst))])
    for i in range(len(lst)):
        lst[i] = lst[i][:length]

    lst = np.array(lst)
    rets, idxs = [], []
    n = 1 + length // 25
    for idx in range(0, length, n):
        # print(np.amax(lst, axis=0)[idx:(idx + n)])
        # print(np.sort(np.amax(lst, axis=0)[idx:(idx + n)]))
        mc = []
        mc_np = np.zeros((lst.shape[0], len(lst[i][idx:(idx + n)])))
        for i in range(lst.shape[0]):
            mc.append(lst[i][idx:(idx + n)])
        for i in range(lst.shape[0]):
            mc_np[i] = np.array([np.random.choice(mc[i], 10).mean() for _ in range(len(mc[i]))])
        # maxes_s = np.array([np.random.choice(maxes, 100).mean() for i in range(len(maxes))])
        # mins_s = np.array([np.random.choice(mins, 100).mean() for i in range(len(maxes))])
        buf = np.minimum(gini(mc_np), 1.0)
        #    buf = gini(lst)
        rets += list(buf)
    dct[key + '_diff'][0] = rets
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
    arg = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]

    logDir = 'resource/exps/'
    # logName = '/baseline/logs.p'
    # logNames = [f'/model/logs.p' for i in range(1)]
    expNames = [f'/model/{i}/' for i in range(7)]
    logNames = [f'logs{i}.p' for i in range(3)]
    for name, config in experiments.exps.items():
        try:
            exps = []
            for expName in expNames:
                dats = []
                for logName in logNames:
                    with open(logDir + name + expName + logName, 'rb') as f:
                        dat = []
                        idx = 0
                        while True:
                            idx += 1
                            try:
                                dat += pickle.load(f)
                            except EOFError as e:
                                break
                        print('Blob length: ', idx)
                    dats.append(dat)
                exps.append(dats)
            split = 'test' if config.TEST else 'train'
            accum = makeAccum(config, 'pops')
            individual(exps, logDir, name, accum, split)
            print('Log success: ', name)
        except Exception as err:
            print(str(err))
