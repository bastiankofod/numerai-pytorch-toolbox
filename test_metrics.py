from metrics import *

x = torch.FloatTensor([0.0308,  0.0240,  0.0360,  0.0314,  0.0033,  0.0097,  0.0221,  0.0263,
                       -0.0083,  0.0329,  0.0182,  0.0251,  0.0272,  0.0447,  0.0099, -0.0234,
                       -0.0007,  0.0217,  0.0123,  0.0175,  0.0303,  0.0011,  0.0175,  0.0377,
                       0.0149, -0.0320,  0.0150,  0.0232])

print('##### TESTING METRICS! #####')
print(f'Sharpe ratio is: {sharpe(x)}')
print(f'Skew is: {skew(x)}')
print(f'Kurtosis is: {kurtosis(x)}')
print(f'Adjusted sharpe ratio is: {adj_sharpe(x)}')
print(f'Omega ratio is: {omega(x)}')
print(f'Sortino ratio is: {sortino(x)}')
print(f'Kappa three ratio is: {kappa_three(x)}')
print(f'Gain/Loss ratio is: {gain_loss_ratio(x)}')
print(f'Upside potential ratio is: {upside_potential_ratio(x)}')
print(f'Max drawdown is: {max_dd(x, 5)}')
print(f'Calmar ratio is: {calmar(x)}')
