# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

from costs import *
from grid_search import *
def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    a = np.matmul(np.transpose(tx),tx)
    b = np.matmul(np.transpose(tx),y)
    w = np.linalg.solve(a,b)
    MSE = compute_loss(y, tx, w)
    return MSE, w
    # ***************************************************
    raise NotImplementedError