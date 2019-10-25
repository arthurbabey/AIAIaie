# -*- coding: utf-8 -*-


# Useful starting lines
'exec(%matplotlib inline)'
import os
clear = lambda: os.system('clear') #on Linux System
clear()

try:
    from IPython import get_ipython
    #get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import numpy as np
import matplotlib.pyplot as plt
import math as m
from preprocessing import *
from plots import *
from crossval import *
from gradient_descent import *
from proj1_helpers import *
from split_data import split_data
from classification_accuracy import *
from stochastic_gradient_descent import *
from least_squares import *
from ridge_regression import *
#from logreg import *


print("\n",'********************************************')

#%%Import
DATA_TRAIN_PATH = '/Users/benoithohl/Desktop/epfl/master_epfl/Ma3/Machine_learning/AIAIaie/data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
print('Data loaded')

#%% Preprocessing 
Data = remove_features_with_too_many_missing_values(tX,0.66)
Data = replace_missing_values_with_global_mean(Data)
ZData = Z_score_of_each_feature(Data)
y = change_negativeOnes_into_Zeros_in_y(y)
print('Data matrix ready')

#%% parameters setting
#partition of the train set
trainx,trainy,validationx,validationy = split_data(ZData, y, 0.75, seed=1)
initial_w = np.zeros(trainx.shape[1])
max_iters = 100
gamma = 0.1
batch_size = 10;
lambda_ = 0.5;
print('parameters set',"\n")

#%% GD
losses, w = gradient_descent(trainy, trainx, initial_w, max_iters, gamma)
weights = np.asarray(w)[-1,:]
y_pred = predict_labels(weights,validationx)
y_pred = change_negativeOnes_into_Zeros_in_y(y_pred)
performance_gradient_descent = calculate_classification_accuracy(validationy, y_pred)
print('GD done')

#%% SGD
losses, w = stochastic_gradient_descent(trainy, trainx, initial_w,batch_size, max_iters, gamma)
weights = np.asarray(w)[-1,:]
y_pred = predict_labels(weights,validationx)
y_pred = change_negativeOnes_into_Zeros_in_y(y_pred)
performance_sochastic_gradient_descent = calculate_classification_accuracy(validationy, y_pred)
print('SGD done')

#%% Least-Square
losses, w = least_squares(trainy, trainx)
weights = np.asarray(w)
y_pred = predict_labels(weights,validationx)
y_pred = change_negativeOnes_into_Zeros_in_y(y_pred)
performance_least_square = calculate_classification_accuracy(validationy, y_pred)
print('Least-Square done')

#%% Ridge
w = ridge_regression(trainy, trainx, lambda_)
weights = np.asarray(w)
y_pred = predict_labels(weights,validationx)
y_pred = change_negativeOnes_into_Zeros_in_y(y_pred)
performance_ridge = calculate_classification_accuracy(validationy, y_pred)
print('Ridge done')
# =============================================================================
# #%% Logistic regression
# losses, w = learning_by_gradient_descent(y, tx, w, gamma)
# weights = np.asarray(w)[-1,:]
# y_pred = predict_labels(weights,validationx)
# performance_logistic_regression = calculate_classification_accuracy(validationy, y_pred)
#print('Logistic regression done')
# #%% Regularized Logistic regression
# losses, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
# weights = np.asarray(w)[-1,:]
# y_pred = predict_labels(weights,validationx)
# performance_regu_logistic_regression = calculate_classification_accuracy(validationy, y_pred)
#print('Regularized Logistic regression done')
# =============================================================================

#%% summary
print("\n",'Results:')
print('performance_gradient_descent: ',performance_gradient_descent)
print('performance_sochastic_gradient_descent: ',performance_sochastic_gradient_descent)
print('performance_least_square: ',performance_least_square)
print('performance_ridge: ',performance_ridge)
#print('performance_logistic_regression: ', performance_logistic_regression)
#print('performance_regu_logistic_regression: ', performance_regu_logistic_regression)











print("\n",'********************************************')
