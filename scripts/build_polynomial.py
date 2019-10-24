# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    phi = np.ones((x.shape[0],1))
    for i in range(1,degree+1):
        phi = np.concatenate((phi, x**i),1)
    return phi
    # ***************************************************
    raise NotImplementedError
