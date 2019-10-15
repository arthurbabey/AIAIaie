# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    error = y-np.matmul(tx,w)
    return -1/len(y)*np.matmul(np.transpose(tx),error)
    # ***************************************************
    raise NotImplementedError


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        gradient=0
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient += compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        gradient = gradient/batch_size
        loss = compute_loss(y, tx, w)
        # ***************************************************
        #raise NotImplementedError
        # ***************************************************
        w = w-gamma*gradient
        # ***************************************************
        #raise NotImplementedError
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws