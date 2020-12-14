"""
Thanks to http://www.turingfinance.com/computational-investing-with-python-week-one/ for many of the metrics.
"""

import torch


def sharpe(x):
    return x.mean() / x.std()


def skew(x):
    mx = x.mean()
    m2 = ((x-mx)**2).mean()
    m3 = ((x-mx)**3).mean()
    return m3 / (m2**1.5)


def kurtosis(x):
    mx = x.mean()
    m2 = ((x-mx)**2).mean()
    m4 = ((x-mx)**4).mean()
    return (m4 / (m2**2)) - 3


def adj_sharpe(x):
    return sharpe(x) * (1 + ((skew(x) / 6) * sharpe(x)) - ((kurtosis(x) / 24) * (sharpe(x)**2)))


def lpm(r, t, o):
    rt = r - t
    return torch.abs(torch.minimum(torch.tensor(1e-7), rt) ** o).mean()


def hpm(r, t, o):
    rt = r - t
    return torch.abs(torch.maximum(torch.tensor(1e-7), rt) ** o).mean()


def sortino(x, t=0.010415154):
    xt = x - t
    return xt.mean() / torch.sqrt(lpm(x, t, 2))


def kappa_three(x, t=1e-7):
    l = lpm(x, t, 3)
    return x.mean() / torch.sign(l) * (torch.pow(torch.abs(l), float(1/3)))


def gain_loss_ratio(x, t=1e-7):
    return hpm(x, t, 1) / lpm(x, t, 1)


def upside_potential_ratio(x, t=1e-7):
    return hpm(x, t, 1) / torch.sqrt(lpm(x, t, 2))


# TODO: max_dd doesn't return the same result as Numerai's max drawdown.
#  This is due to how pandas does rolling windows.
#  To be resolved!
def max_dd(x, w=20):
    r = torch.max(torch.cumprod(x+1, dim=0).unfold(0, w, 1))
    d = torch.cumprod(x+1, dim=0)
    return -torch.max(r - d)


def calmar(x):
    return x.mean() / max_dd(x)