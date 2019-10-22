from costs import compute_mse
from ridge_regression import ridge_regression
from build_polynomial import build_poly



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
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    #print("step0")
    x_test = x[k_indices[0]]
    y_test = y[k_indices[0]]
    
    x_train = np.reshape(x[k_indices[1:k]],len(x[k_indices[1]])*(k-1))
    y_train = np.reshape(y[k_indices[1:k]],len(y[k_indices[1]])*(k-1))
    

    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    #print("step1")
    x_test_poly = build_poly(x_test,degree).T
    x_train_poly = build_poly(x_train,degree).T   

    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    #print("step2")
    w = ridge_regression(y_train, x_train_poly, lambda_)

    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    #print("step3")
    loss_tr = np.sqrt(2*compute_mse(y_train,x_train_poly, w))
    loss_te = np.sqrt(2*compute_mse(y_test,x_test_poly, w))
    # ***************************************************
    return loss_tr, loss_te