# -*- coding: utf-8 -*-
# Exploration of lambda for ridge regression
# Useful starting lines
'exec(%matplotlib inline)'
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

import os
clear = lambda: os.system('clear') #on Linux System
clear()
print("\n",'********************************************')

#%%Import
print('Data is loading')
DATA_TRAIN_PATH = '/Users/benoithohl/Desktop/epfl/master_epfl/Ma3/Machine_learning/AIAIaie/data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
print('Data loaded')

#%% Preprocessing 
Data = remove_features_with_too_many_missing_values(tX,0.66)
Data = replace_missing_values_with_global_mean(Data)
ZData = Z_score_of_each_feature(Data)
print('Data matrix ready')

#%% parameters setting
#partition of the train set
trainx,trainy,validationx,validationy = split_data(ZData, y, 0.75, seed=1)
initial_w = np.zeros(trainx.shape[1])
max_iters = 100
gamma = 0.1
batch_size = 10;
lambdas_vector = np.logspace(-3, 0, num=15)
#lambdas_vector = np.linspace(0, 1, num=15)
print('parameters set',"\n")

#%% Ridge
performance_ridge = []
performance_training = []
for lambda_ in lambdas_vector:
    w = ridge_regression(trainy, trainx, lambda_)
    weights = np.asarray(w)
    y_pred = predict_labels(weights,validationx)
    performance_ridge = np.append(performance_ridge,calculate_classification_accuracy(validationy, y_pred))
    performance_training = np.append(performance_training,calculate_classification_accuracy(trainy, predict_labels(weights,trainx)))
print('Ridge done')

#%% Plots
def visualization_perf_wrt_lambdas(lambdas, train_perf, test_perf):
    """visualization the curves of train/test classification accuracy."""
    plt.semilogx(lambdas, train_perf, marker=".", color='b', label='train performance')
    plt.semilogx(lambdas, test_perf, marker=".", color='r', label='test performance')
    plt.xlabel("Lambda")
    plt.ylabel("Performance")
    plt.title("Exploration lambda Ridge")
    plt.legend(loc=1)
    plt.grid(True)
    plt.savefig("Exploration lambda Ridge")
    
visualization_perf_wrt_lambdas(lambdas_vector, performance_training, performance_ridge)
print("\n",'********************************************')