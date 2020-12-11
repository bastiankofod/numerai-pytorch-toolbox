import torch

def sharpe(x):
    return x.mean() / x.std()

def numerai_sharpe(x):
    return (x.mean() - 0.010415154) / x.std()

def ar1(x):
    return 

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

def adj_sharpe(x, sharpe_f):
    return sharpe_f(x) * (1 + ((skew(x) / 6) * sharpe_f(x)) - ((kurtosis(x) / 24) * (sharpe_f(x)**2)))

def sortino(x, t=0.010415154):
    xt = x - t
    return xt.mean() / (torch.sum(torch.minimum(torch.tensor(1e-7), xt)**2)/(len(xt)-1))**0.5
