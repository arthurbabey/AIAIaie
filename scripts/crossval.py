from functions_for_implementations import compute_mse
from ridge_regression import ridge_regression
from build_polynomial import build_poly
import numpy as np


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # get k'th subgroup in test, others in train
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]
    
    ind = np.linspace(0,k_indices.shape[0]-1,k_indices.shape[0], dtype = np.int64)
    ind = np.delete(ind,k)
    
    new_ind = np.ravel(k_indices[ind])
    x_train = x[new_ind]
    y_train = y[new_ind]
    

    #form data with polynomial degree
    x_test_poly = build_poly(x_test,degree).T
    x_train_poly = build_poly(x_train,degree).T   
    
    #find weights
    w = ridge_regression(y_train, x_train_poly, lambda_)

    # calculate the loss for train and test data
    loss_tr = np.sqrt(2*compute_mse(y_train,x_train_poly, w))
    loss_te = np.sqrt(2*compute_mse(y_test,x_test_poly, w))
    # ***************************************************
    return loss_tr, loss_te


def cross_validation_demo():
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = np.zeros(k_fold)
    rmse_te = np.zeros(k_fold)
    rmse_trlambdas = np.zeros(len(lambdas))
    rmse_telambdas = np.zeros(len(lambdas))
    
    for i in range(len(lambdas)):
        for j in range(k_fold):
            rmse_tr[j], rmse_te[j] = cross_validation(y, x, k_indices, j, lambdas[i], degree)
        rmse_trlambdas[i] = np.mean(rmse_tr)
        rmse_telambdas[i] = np.mean(rmse_te)
    cross_validation_visualization(lambdas, rmse_trlambdas, rmse_telambdas)
    # ***************************************************    

