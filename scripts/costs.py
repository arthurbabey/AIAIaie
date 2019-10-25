import numpy as np
# -*- coding: utf-8 -*-
"""A function to compute the cost."""


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - np.matmul(tx,w)
    mse = np.matmul(e,e) / (2 * len(e))
    return mse
