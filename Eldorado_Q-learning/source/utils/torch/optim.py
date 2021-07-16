import torch

from source.utils.torch import loss


def backwardAgent(batches, device='cpu', n_quant=1):
    returns = torch.stack(batches['returns']).to(device).float().detach()
    Qs = torch.stack(batches['Qs']).to(device).float()
    weights = batches.get('weights', None)
    if weights is not None:
        weights = torch.tensor(weights).to(device).view(-1, 1).float()
    if n_quant > 1:
        qmse = 0
        taus = torch.linspace(0.0, 1.0 - 1./n_quant, n_quant).to(device) + 0.5 / n_quant
        for i in range(n_quant):
            qmse += loss.Q_huber(Qs[:, i].view(-1, 1), returns, weights, tau=taus[i])
    else:
        qmse = loss.Q_MSE(Qs.view(-1, 1), returns.view(-1, 1), weights)
    qmse.backward()
