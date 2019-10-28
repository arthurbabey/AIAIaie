import numpy as np
from functions_for_implementations import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # ***************************************************
    # Define parameter loss
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        #compute gradient and loss
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        
        #update weights
        w = w-gamma*gradient
        # store loss
        losses.append(loss)

        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        #converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break
    return w, losses[-1]
    # ***************************************************

    
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    batch_size = 1
    w = initial_w
    # Define parameter to store loss
    losses = []
    
    for n_iter in range(max_iters):
        #calculate gradient using minibatch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        gradient = gradient/batch_size
        loss = compute_mse(y, tx, w)

        w = w-gamma*gradient

        # store w and loss
        losses.append(loss)
        
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        #converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break
            
    return w, losses[-1]
    # ***************************************************
    
    

def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    #compute optimal weights
    a = np.matmul(np.transpose(tx),tx)
    b = np.matmul(np.transpose(tx),y)
    w = np.linalg.solve(a,b)
    
    #compute loss
    loss = compute_mse(y, tx, w)
    return w, loss
    # ***************************************************
    
    
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    #compute optimal weights
    a = np.matmul(np.transpose(tx),tx)+(2*len(y)*lambda_*np.eye(tx.shape[1]))
    b = np.matmul(np.transpose(tx),y)
    w = np.linalg.solve(a,b)
    
    #compute loss
    loss = compute_mse(y, tx, w)
    return w, loss
    # ***************************************************
    

    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    run gradient descent, using logistic regression
    Return the loss and final weights.
    """
    # ***************************************************
    batch_size = 1
    losses = []
    w = initial_w

    # start gradient descent
    for n_iter in range(max_iters):
        # get loss and update w using logistic regression.
        loss, w = learning_by_log_regression(y, tx, w, batch_size, gamma)
        
        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break
            
    return w, losses[-1] 
    # ***************************************************    
    
    
    
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """
    run gradient descent, using logistic regression
    Return the loss and final weights.
    """
    # ***************************************************
    batch_size = 1
    losses = []
    w = initial_w

    # start gradient descent
    for n_iter in range(max_iters):
        # get loss and update w using regularized logistic regression.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_, batch_size)

        if n_iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break
            
    return w, losses[-1] 
    # *************************************************** 
    