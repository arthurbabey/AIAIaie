import numpy as np
from helpers import batch_iter
import time


def sigmoid(t):
    """apply sigmoid function on t."""
    # ***************************************************
    return 1 / (1 + np.exp(-t))
    # ***************************************************
    
    
def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    # ***************************************************    
    return np.sum(np.log(1 + np.exp(np.matmul(tx, w))) - np.matmul(tx, w)*y)
    # ***************************************************

    
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    # ***************************************************
    return np.matmul(np.transpose(tx), sigmoid(np.matmul(tx, w))-y)
    # ***************************************************

    
    
def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    # ***************************************************
    loss = calculate_losslog(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    
    w = w - gamma*gradient
    
    return loss, w
    # ***************************************************


def learning_by_log_regression(y, tx, w, batch_size, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        gradient = calculate_gradient(minibatch_y, minibatch_tx, w)
        loss = calculate_loss(minibatch_y, minibatch_tx, w)
    gradient = gradient/batch_size
    loss = loss/batch_size
    
    w = w-gamma*gradient

    return loss, w
    # ***************************************************




def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    S = np.zeros((len(y), len(y)))
    
    for i in range(len(y)):
        a = sigmoid(np.matmul(tx[i,:], w))
        S[i,i] = a*(1 - a)
    
    return np.matmul(np.matmul(np.transpose(tx), S), tx)
    # ***************************************************
    
    
    
def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    
    return loss, gradient, hessian
    # ***************************************************

    
    
    
def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    # ***************************************************
    loss, gradient, hessian = logistic_regression(y, tx, w)
    
    w = w - np.matmul(gamma*np.linalg.inv(hessian),gradient)

    return loss, w
    # ***************************************************
    


def penalized_logistic_regression(y, tx, w, lambda_,batch_size):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        gradient = calculate_gradient(minibatch_y, minibatch_tx, w)+lambda_*w
        loss = calculate_loss(minibatch_y, minibatch_tx, w)
    gradient = gradient/batch_size
    loss = loss/batch_size
    return loss, gradient
    # ***************************************************


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_,batch_size):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_,batch_size)
    w = w - gamma*gradient

    return loss, w
    # ***************************************************


    
def running_gradient(y, tx, lambda_, method='log'):
    """
    run gradient descent, using logistic regression, 
    penalized log regression or newton method.
    Return the loss and final weights.
    """
    # ***************************************************
    batch_size = 32
    max_iter = 1000
    gamma = 0.01
    threshold = 1e-8
    losses = []
    w = np.zeros((tx.shape[1], 1))

    # start gradient descent
    for iter in range(max_iter):
        # get loss and update w.
        if method == 'log':
            loss, w = learning_by_log_regression(y, tx, w, batch_size, gamma)
        if method == 'penalized':
            loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_, batch_size)
        if method == 'newton':
            loss, w = learning_by_newton_method(y, tx, w, gamma)
        # log info

        if iter % 100 == 0:
            #print(w)
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return loss, w
    # ***************************************************




