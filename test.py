import numpy as np
import scipy.sparse as sps
import torch
import cupy as cp
# row = np.array([0, 0, 1, 3, 1, 0, 0])
# col = np.array([0, 2, 1, 3, 1, 0, 0])
# data = np.array([1, 1, 1, 1, 1, 1, 1])
# matrix = sps.coo_matrix((data, (row, col)), shape=[4, 4]).tocsr()
a = torch.arange(10)
n = torch.exp(a)
print(type(n))
print(n)
