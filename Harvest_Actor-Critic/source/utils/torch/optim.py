from collections import defaultdict
import torch

from source.utils.torch import loss


def mergeTrajectories(trajectories, device):
    outs = defaultdict(list)
    for trajectory in trajectories:
        vals = torch.cat(trajectory['vals']).view(-1, 1)
        rets = torch.cat(trajectory['returns']).view(-1, 1)
        outs['rewards'].append(rets)
        outs['rets'].append(rets + 0.99 * torch.cat((vals[1:].detach(), torch.zeros(1, 1).to(device))))
        outs['vals'].append(vals)
        policy = torch.cat(trajectory['policy'])
        oldPolicy = torch.cat(trajectory['oldPolicy'])
        outs['policy'].append(policy)
        outs['Qs'].append(torch.cat(trajectory['Qs']).view(-1, 1))
        outs['As'].append(torch.cat(trajectory['As']).view(-1, 1))
        outs['rho'].append(policy / (1e-5 + oldPolicy))
    return outs


def backwardAgentOffPolicy(trajectories, valWeight=0.5, entWeight=0.01, lambda_adv=0.5, device='cuda'):
    outs = mergeTrajectories(trajectories, device)
    vals = torch.stack(outs['vals']).to(device).view(-1, 1).float()
    rets = torch.stack(outs['rets']).to(device).view(-1, 1).float()
    As = torch.stack(outs['As']).to(device).view(-1, 1).float()
    rho = torch.stack(outs['rho']).to(device).view(-1, 1).float()
    policy = torch.cat(outs['policy']).to(device).float()

    pg, entropy = loss.PG(policy, rho, loss.advantage(lambda_adv * As + (1 - lambda_adv) * (rets - vals)))
    valLoss = loss.valueLoss(rets, vals)
    return pg + valWeight * valLoss + entWeight * entropy, outs


def welfareFunction(vals, mode):
    n_agents = len(vals)
    if mode == 'min':
        return torch.stack(vals).view(n_agents, -1).min(0)[0]
    elif mode.startswith('bot'):
        k = int(mode[3:])
        return torch.stack(vals).view(n_agents, -1).topk(k, 0, largest=False)[0].sum(0)
    else:
        return torch.stack(vals).view(n_agents, -1).sum(0)


def backwardSocial(rets, Qs, device='cpu', mode='min'):
    rets = welfareFunction(rets, mode).to(device).reshape(-1, 1).float()
    totLoss = loss.valueLoss(Qs, rets.detach())
    return totLoss


def backwardSocialVanillaComa(rewards, Qs, device='cpu', mode='min'):
    rets = welfareFunction(rewards, mode).to(device).reshape(-1, 1).float()
    rets += 0.99 * torch.cat((Qs[1:], torch.zeros(1, 1).to(device)))
    totLoss = loss.valueLoss(Qs, rets.detach())
    return totLoss
