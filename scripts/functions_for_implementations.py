import numpy as np


def compute_mse(y, tx, w):
    """Calculate the loss with MSE.
    """
    # ***************************************************
    return 1/(2*y.shape[0])*np.sum((y-np.matmul(tx, w))**2)
    # ***************************************************
    
    
    
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    error = y-np.matmul(tx,w)
    return -1/len(y)*np.matmul(np.transpose(tx),error)
    # ***************************************************
    

    
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    error = y-np.matmul(tx,w)
    return -1/len(y)*np.matmul(np.transpose(tx),error)      
    # ***************************************************
    
    
    
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    reset_num = 0
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = (batch_num-reset_num) * batch_size
        end_index = min((batch_num-reset_num + 1) * batch_size, data_size)
        if min((batch_num-reset_num + 1) * batch_size, data_size) == data_size:
            reset_num = batch_num+1
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            


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
    

    
def learning_by_log_regression(y, tx, w, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    #calculate gradient using minibatch
    gradient = calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w)
    gradient = gradient/len(y)
    loss = loss/len(y)
    
    w = w-gamma*gradient

    return loss, w
    # ***************************************************
    


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    # ***************************************************
    #calculate gradient 
    gradient = calculate_gradient(y, tx, w)+lambda_*w
    loss = calculate_loss(y, tx, w)
    gradient = gradient/len(y)
    loss = loss/len(y)
    
    return loss, gradient
    # ***************************************************

    

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_,batch_size):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    #calculate loss and gradient
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    
    #update the weights
    w = w - gamma*gradient

    return loss, w
    # ***************************************************    
    
    
    
    
    
    
    
            