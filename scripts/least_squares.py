# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    a = np.matmul(np.transpose(tx),tx)
    b = np.matmul(np.transpose(tx),y)
    w = np.linalg.solve(a,b)
    MSE = compute_mse(y, tx, w)
    return MSE, w
    # ***************************************************
    raise NotImplementedError
