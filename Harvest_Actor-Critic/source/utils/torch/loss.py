import torch


def advantage(A):
    std = 1e-4 + A.std() if len(A) > 0 else 1
    adv = (A - A.mean()) / std
    adv = adv.detach()
    adv[adv != adv] = 0
    return adv


def valueLoss(v, returns):
    return (0.5 * (v - returns) ** 2).mean()


def entropyLoss(prob, logProb):
    return (prob * logProb).sum(1).mean()


def ppo_loss(A, rho, eps=0.2):
    return -torch.min(rho * A, rho.clamp(1 - eps, 1 + eps) * A).mean()


def PG(pi, rho, A):
    polLoss = ppo_loss(A, rho)
    entLoss = entropyLoss(pi, pi.log())
    return polLoss, entLoss
