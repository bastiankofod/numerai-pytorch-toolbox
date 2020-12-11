import torch
from metrics import *

x = torch.randn(100)

print('##### TESTING METRICS! #####')
print(f'Sharpe ratio is: {sharpe(x)}')
print(f'Numerai sharpe ratio is: {numerai_sharpe(x)}')
print(f'Skew is: {skew(x)}')
print(f'Kurtosis is: {kurtosis(x)}')
print(f'Adjusted sharpe ratio is: {adj_sharpe(x, sharpe)}')
print(f'Adjusted numerai sharpe ratio is: {adj_sharpe(x, numerai_sharpe)}')
print(f'Sortino ratio is: {sortino(x)}')

