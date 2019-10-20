# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    return 1/(2*y.shape[0])*np.sum((y-np.matmul(tx, w))**2)
    # ***************************************************
    raise NotImplementedError