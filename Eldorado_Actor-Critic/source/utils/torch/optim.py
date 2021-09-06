import torch
from torch.nn import functional as F

from source.utils.torch import loss


def mergeTrajectories(trajectories):
    outs = {'vals': [], 'rewards': [], 'rets': [],
            'Qs': [], 'As': [],
            'policy': [],
            'rho': [],
            'retsTD': [],
            'action': []}
    for trajectory in trajectories:
        vals = torch.cat(trajectory['vals']).view(-1, 1)
        rets = torch.cat(trajectory['returns']).view(-1, 1)
        retsTD = torch.cat(trajectory['retsTD']).view(-1, 1)
        outs['rets'].append(rets)
        outs['vals'].append(vals)
        outs['retsTD'].append(retsTD)
        outs['Qs'].append(torch.cat(trajectory['Qs']).view(-1, 1))
        outs['As'].append(torch.cat(trajectory['As']).view(-1, 1))
        actions = torch.cat(trajectory['actions'])
        policy = torch.cat(trajectory['policy'])
        outs['policy'].append(policy)
        oldPolicy = torch.cat(trajectory['oldPolicy'])
        outs['rho'].append((F.log_softmax(policy, dim=1).gather(1, actions.view(-1, 1)) -
                            F.log_softmax(oldPolicy, dim=1).gather(1, actions.view(-1, 1))).exp())
    return outs


def backwardAgent(trajectories, valWeight=0.5, entWeight=0.01, device='cuda', lambda_adv=0.5):
    outs = mergeTrajectories(trajectories)
    vals = torch.stack(outs['vals']).to(device).view(-1, 1).float()
    rets = torch.stack(outs['rets']).to(device).view(-1, 1).float()
    rho = torch.stack(outs['rho']).to(device).view(-1, 1).float()
    As = torch.stack(outs['As']).to(device).view(-1, 1).float().detach()
    policy = torch.stack(outs['policy']).to(device).view(-1, 10).float()
    pg, entropy = loss.PG(policy, rho, loss.advantage(lambda_adv * As + (1 - lambda_adv) * (rets - vals)))
    valLoss = loss.valueLoss(rets, vals)
    return (pg + valWeight * valLoss + entWeight * entropy), outs


def welfareFunction(vals, mode):
    if mode == 'min':
        return torch.stack(vals).view(2, -1).min(0)[0]
    elif mode == 'softmin':
        return -torch.logsumexp(-torch.stack(vals).view(2, -1), dim=0)
    else:
        return torch.stack(vals).view(2, -1).sum(0)


def backwardSocial(rets, Q_tot, device='cpu', mode='sum'):
    rets = welfareFunction(rets, mode).to(device).reshape(-1, 1).float()
    totLoss = loss.valueLoss(Q_tot, rets.detach())
    return totLoss
