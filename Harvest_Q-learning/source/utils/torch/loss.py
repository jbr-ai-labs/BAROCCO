def Q_MSE(Q, returns, weights=None):
    if weights is None:
        return (0.5 * (Q - returns) ** 2).mean()
    else:
        return (0.5 * weights * (Q - returns) ** 2).mean()


def huber(u, eps):
    cond = (u.abs() < eps).float().detach()
    return 0.5 * u.pow(2) * cond + eps * (u.abs() - 0.5 * eps) * (1.0 - cond)


def Q_huber(Q, returns, weights=None, eps=1, tau=0.5):
    u = returns - Q
    loss = huber(u, eps) / eps
    if weights is not None:
        loss = weights * loss
    loss = loss * (tau - (u < 0).float()).abs()
    return loss.mean()
