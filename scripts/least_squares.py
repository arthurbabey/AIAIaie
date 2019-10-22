# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    a = np.matmul(np.transpose(tx),tx)
    b = np.matmul(np.transpose(tx),y)
    w = np.linalg.solve(a,b)
    MSE = compute_loss(y, tx, w)
    return MSE, w
    # ***************************************************
    raise NotImplementedError
