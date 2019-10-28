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

    #print(y.shape), print(tx.shape), print(w.shape)
    #y = np.array(y, ndmin=2)
    loss = calculate_loss(y, tx, w)/len(y)
    gradient = calculate_gradient(y, tx, w)/len(y)

    w = w - gamma*gradient

    return loss, w
    # ***************************************************


def learning_by_log_regression(y, tx, w, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************

    ind = np.random.choice(len(y))
    stochy = np.array(y[ind],ndmin=2)
    stochx = np.array(tx[ind],ndmin=2)
    gradient = calculate_gradient(stochy, stochx, w)
    loss = calculate_loss(stochy, stochx, w)

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



def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    ind = np.random.choice(len(y))
    stochy = np.array(y[ind],ndmin=2)
    stochx = np.array(tx[ind],ndmin=2)
    gradient = calculate_gradient(stochy, stochx, w)+lambda_*w
    loss = calculate_loss(stochy, stochx, w)
    return loss, gradient
    # ***************************************************


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*gradient

    return loss, w
    # ***************************************************



def running_gradient(y, tx, w, lambda_, method='log'):
    """
    run gradient descent, using logistic regression,
    penalized log regression or newton method.
    Return the loss and final weights.
    """
    # ***************************************************
    max_iter = 8000
    gamma = 0.01
    threshold = 1e-8
    losses = []

    #w = np.zeros((tx.shape[1], 1))

    # start gradient descent
    for n_iter in range(max_iter):
        # get loss and update w.
        if method == 'log':
            loss, w = learning_by_log_regression(y, tx, w, gamma)
        if method == 'penalized':
            loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if method == 'newton':
            loss, w = learning_by_newton_method(y, tx, w, gamma)
        if method == 'gradient':
            loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info

        if n_iter % 10 == 0:
            #print(w)
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
        # converge criterion
        #if len(losses) == 1000:
         #   break
        losses.append(loss)
        #if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
        #    break

    return w
    # ***************************************************
