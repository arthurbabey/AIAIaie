# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    a = np.matmul(np.transpose(tx),tx)+(2*len(y)*lambda_*np.eye(tx.shape[1]))
    b = np.matmul(np.transpose(tx),y)
    w = np.linalg.solve(a,b)
    
    return w
    # ***************************************************