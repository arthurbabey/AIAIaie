# -*- coding: utf-8 -*-
"""Gradient Descent"""
import numpy as np
from costs import compute_mse
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    error = y-np.matmul(tx,w)
    return -1/len(y)*np.matmul(np.transpose(tx),error)
    # ***************************************************
    raise NotImplementedError


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        # ***************************************************
        #raise NotImplementedError
        # ***************************************************
        w = w-gamma*gradient
        # ***************************************************
        #raise NotImplementedError
        # store w and loss
        ws.append(w)
        losses.append(loss)
        if n_iter%10 == 0:
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
