# -*- coding: utf-8 -*-
# Exploration of batch_size for SGD
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
batch_size_vector = np.arange(1, 16)
print('parameters set',"\n")

#%% SGD
performance_SGD = []
performance_training = []
for batch_size in batch_size_vector:
    print('Model for batch_size = ',batch_size,' building.')
    losses, w = stochastic_gradient_descent(trainy, trainx, initial_w,batch_size, max_iters, gamma)
    weights = np.asarray(w)[-1,:]
    y_pred = predict_labels(weights,validationx)
    performance_SGD = np.append(performance_SGD,calculate_classification_accuracy(validationy, y_pred))
    performance_training = np.append(performance_training,calculate_classification_accuracy(trainy, predict_labels(weights,trainx)))
    
print('SGD done')

#%% Plots
def visualization_perf_wrt_batch_size(batch_sizes, train_perf, test_perf):
    """visualization the curves of train/test classification accuracy."""
    plt.plot(batch_sizes, train_perf, marker=".", color='b', label='train performance')
    plt.plot(batch_sizes, test_perf, marker=".", color='r', label='test performance')
    plt.xlabel("batch size")
    plt.ylabel("Performance")
    plt.title("Exploration batch size SGD")
    plt.legend(loc=0)
    plt.grid(True)
    plt.savefig("Exploration_batch_size_SGD")
    
visualization_perf_wrt_batch_size(batch_size_vector, performance_training, performance_SGD)
print("\n",'********************************************')

