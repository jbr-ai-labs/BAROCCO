import numpy as np
import torch


def zeroGrads(ann):
    ind = 0
    for e in ann.parameters():
        if e.grad is None:
            continue
        shape = e.size()
        nParams = np.prod(shape)
        e.grad.data *= 0
        ind += nParams


def setParameters(ann, meanVec):
    ind = 0
    for e in ann.parameters():
        shape = e.size()
        nParams = np.prod(shape)
        e.data = torch.Tensor(np.array(meanVec[ind:ind + nParams]).reshape(*shape))
        ind += nParams


def getParameters(ann):
    ret = []
    for name, e in ann.named_parameters():
        ret += e.data.cpu().view(-1).numpy().tolist()
    return ret
