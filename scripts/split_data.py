# -*- coding: utf-8 -*-


import numpy as np

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    np.random.seed(seed)
    n = len(x)

    n_train = int(ratio * n)
    train_ind = np.random.choice(n, n_train, replace=False)

    ind = np.arange(n)

    mask = np.in1d(ind, train_ind)

    test_ind = np.random.permutation(ind[~mask])

    x_train = x[train_ind]
    y_train = y[train_ind]

    x_test = x[test_ind]
    y_test = y[test_ind]

    return x_train, y_train, x_test, y_test
