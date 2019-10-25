import numpy as np
from helpers import batch_iter


def sigmoid(t):
    """apply sigmoid function on t."""
    # ***************************************************
   
    return np.exp(t) / (1 + np.exp(t))
    
    # ***************************************************
    
    
def calculate_lossreg(y, tx, w):
    """compute the cost by negative log likelihood."""
    # ***************************************************    
    
    return np.sum(np.log(1 + np.exp(np.matmul(tx, w))) - np.matmul(tx, w)*y)/len(y)
    
    # ***************************************************

    
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    # ***************************************************
    
    return np.matmul(np.transpose(tx), sigmoid(np.matmul(tx, w))-y)/len(y)

    # ***************************************************

    
    
def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # compute the cost: TODO
    loss = calculate_lossreg(y, tx, w)

    # ***************************************************
    
    gradient = calculate_gradient(y, tx, w)
    # compute the gradient: TODO

    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    
    w = w - gamma*gradient
    
    # ***************************************************

    return loss, w



def logistic_regression_SGD(y, tx, initial_w, batch_size, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    w = initial_w
    gradient=0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        gradient += calculate_gradient(minibatch_y, minibatch_tx, w)
    gradient = gradient/batch_size
    loss = calculate_lossreg(y, tx, w)

    # ***************************************************
    
    w = w-gamma*gradient
    
    # ***************************************************
    
    return loss, w





def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate hessian: TODO
    
    S = np.zeros((len(y), len(y)))
    
    for i in range(len(y)):
        a = sigmoid(np.matmul(tx[i,:], w))
        S[i,i] = a*(1 - a)
    
    return np.matmul(np.matmul(np.transpose(tx), S), tx)
    #return np.matmul(np.matmul(np.transpose(tx), S), tx)
    
    # ***************************************************
    
    
    
def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and hessian: TODO
    
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    
    return loss, gradient, hessian
    # ***************************************************

    
    
    
def learning_by_newton_method(y, tx, w):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient and hessian: TODO
    
    loss, gradient, hessian = logistic_regression(y, tx, w)

    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    
    w = w - np.matmul(np.linalg.inv(hessian),gradient)
    # ***************************************************
    return loss, w



def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and hessian: TODO
    
    gradient = calculate_gradient(y, tx, w)+lambda_*w
    #hessian = calculate_hessian(y, tx, w)
    loss = calculate_loss(y, tx, w)
    
    return loss, gradient#, hessian
    
    # ***************************************************


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient: TODO
    
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    # ***************************************************
    #raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    
    w = w - gamma*gradient
    
    # ***************************************************
    return loss, w
