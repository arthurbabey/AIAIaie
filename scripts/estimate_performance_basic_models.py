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
#from crossval import *
from implementations import *
from helpers import *
from split_data import split_data
from classification_accuracy import *
from logreg import *
from create_data_with_jet import *
from build_polynomial import *

print("\n",'********************************************')

#%%Import
DATA_TRAIN_PATH = 'C:/Users/joeld/Desktop/EPFL/machine learning/AIAIaie/data/train.csv'
#DATA_TRAIN_PATH = '/Users/benoithohl/Desktop/epfl/master_epfl/Ma3/Machine_learning/AIAIaie/data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
trainx,trainy,ids_train,validationx,validationy,ids_test = split_data(tX, y, ids, 0.8, seed=1)
print('Data loaded')

#%% Preprocessing 

data_jetnum = np.array(create_data_with_jet(trainx,trainy, ids_train, validationx, validationy, ids_test))
print('Data matrix ready')


#%% parameters setting
#partition of the train set

lambda_ = 0.01
degree = 1
max_iters = 5000
gamma = 0.01
ids_pred = []
y_preds = []
ids_predtrain = []
y_predstrain = []
yvalidations = []
yvalidationstrain = []

print('parameters set',"\n")

#%% GD

for i in range(4):
    
    data_jet = np.copy(data_jetnum)


    
    data_jet[i,0] = build_poly(data_jet[i,0], degree)
    data_jet[i,2] = build_poly(data_jet[i,2], degree)

    initial_w = np.zeros(len(data_jet[i,0][0,:]))
    #w initiliser ici et passer en paramètre de running_gradient

    #pour utiliser penalized ajouter paramètre 'penalized' dans running_gradient

    weight, losses = least_squares_GD(data_jet[i,1], data_jet[i,0],initial_w, max_iters, gamma)

    y_pred = predict_labels_notlog(weight, data_jet[i,2])
    y_preds = np.append(y_preds, y_pred)
    ids_pred = np.append(ids_pred,data_jet[i,5])
    yvalidations = np.append(yvalidations, data_jet[i,3])
    
    y_pred = predict_labels_notlog(weight, data_jet[i,0])
    y_predstrain = np.append(y_predstrain, y_pred)
    ids_predtrain = np.append(ids_predtrain,data_jet[i,4])
    yvalidationstrain = np.append(yvalidationstrain, data_jet[i,1])


#puts every jet value back in order
order = ids_pred.argsort()
ids_pred = ids_pred[order]
y_preds = y_preds[order]

yvalidations = yvalidations[order]
y_preds[y_preds == -1] = 0

performance_gradient_descent = calculate_classification_accuracy(yvalidations, y_preds)


#puts every jet value back in order
order = ids_predtrain.argsort()
ids_predtrain = ids_predtrain[order]
y_predstrain = y_predstrain[order]
yvalidationstrain = yvalidationstrain[order]

y_predstrain[y_predstrain == -1] = 0

train_performance_gradient_descent = calculate_classification_accuracy(yvalidationstrain, y_predstrain)

print('GD done')

#%% SGD
for i in range(4):
    
    data_jet = np.copy(data_jetnum)


    
    data_jet[i,0] = build_poly(data_jet[i,0], degree)
    data_jet[i,2] = build_poly(data_jet[i,2], degree)

    initial_w = np.zeros(len(data_jet[i,0][0,:]))
    #w initiliser ici et passer en paramètre de running_gradient

    #pour utiliser penalized ajouter paramètre 'penalized' dans running_gradient

    weight, losses = least_squares_SGD(data_jet[i,1], data_jet[i,0],initial_w, max_iters, gamma)

    y_pred = predict_labels_notlog(weight, data_jet[i,2])
    y_preds = np.append(y_preds, y_pred)
    ids_pred = np.append(ids_pred,data_jet[i,5])
    yvalidations = np.append(yvalidations, data_jet[i,3])
    
    y_pred = predict_labels_notlog(weight, data_jet[i,0])
    y_predstrain = np.append(y_predstrain, y_pred)
    ids_predtrain = np.append(ids_predtrain,data_jet[i,4])
    yvalidationstrain = np.append(yvalidationstrain, data_jet[i,1])



#puts every jet value back in order
order = ids_pred.argsort()
ids_pred = ids_pred[order]
y_preds = y_preds[order]
yvalidations = yvalidations[order]
y_preds[y_preds == -1] = 0

performance_sochastic_gradient_descent = calculate_classification_accuracy(yvalidations, y_preds)

#puts every jet value back in order
order = ids_predtrain.argsort()
ids_predtrain = ids_predtrain[order]
y_predstrain = y_predstrain[order]
yvalidationstrain = yvalidationstrain[order]

y_predstrain[y_predstrain == -1] = 0

train_performance_sochastic_gradient_descent = calculate_classification_accuracy(yvalidationstrain, y_predstrain)


print('SGD done')

#%% Least-Square

for i in range(4):
    
    data_jet = np.copy(data_jetnum)


    
    data_jet[i,0] = build_poly(data_jet[i,0], degree)
    data_jet[i,2] = build_poly(data_jet[i,2], degree)

    initial_w = np.zeros(len(data_jet[i,0][0,:]))
    #w initiliser ici et passer en paramètre de running_gradient

    #pour utiliser penalized ajouter paramètre 'penalized' dans running_gradient

    weight, losses = least_squares(data_jet[i,1], data_jet[i,0])

    y_pred = predict_labels_notlog(weight, data_jet[i,2])
    y_preds = np.append(y_preds, y_pred)
    ids_pred = np.append(ids_pred,data_jet[i,5])
    yvalidations = np.append(yvalidations, data_jet[i,3])

    y_pred = predict_labels_notlog(weight, data_jet[i,0])
    y_predstrain = np.append(y_predstrain, y_pred)
    ids_predtrain = np.append(ids_predtrain,data_jet[i,4])
    yvalidationstrain = np.append(yvalidationstrain, data_jet[i,1])


#puts every jet value back in order
order = ids_pred.argsort()
ids_pred = ids_pred[order]
y_preds = y_preds[order]
yvalidations = yvalidations[order]
y_preds[y_preds == -1] = 0

performance_least_square = calculate_classification_accuracy(yvalidations, y_preds)

#puts every jet value back in order
order = ids_predtrain.argsort()
ids_predtrain = ids_predtrain[order]
y_predstrain = y_predstrain[order]
yvalidationstrain = yvalidationstrain[order]

y_predstrain[y_predstrain == -1] = 0

train_performance_least_square = calculate_classification_accuracy(yvalidationstrain, y_predstrain)

print('Least-Square done')

#%% Ridge

for i in range(4):
    
    data_jet = np.copy(data_jetnum)


    
    data_jet[i,0] = build_poly(data_jet[i,0], degree)
    data_jet[i,2] = build_poly(data_jet[i,2], degree)

    initial_w = np.zeros(len(data_jet[i,0][0,:]))
    #w initiliser ici et passer en paramètre de running_gradient

    #pour utiliser penalized ajouter paramètre 'penalized' dans running_gradient
    print(ridge_regression(data_jet[i,1], data_jet[i,0], lambda_))
    weight, losses = ridge_regression(data_jet[i,1], data_jet[i,0], lambda_)

    y_pred = predict_labels_notlog(weight, data_jet[i,2])
    y_preds = np.append(y_preds, y_pred)
    ids_pred = np.append(ids_pred,data_jet[i,5])
    yvalidations = np.append(yvalidations, data_jet[i,3])

    y_pred = predict_labels_notlog(weight, data_jet[i,0])
    y_predstrain = np.append(y_predstrain, y_pred)
    ids_predtrain = np.append(ids_predtrain,data_jet[i,4])
    yvalidationstrain = np.append(yvalidationstrain, data_jet[i,1])


#puts every jet value back in order
order = ids_pred.argsort()
ids_pred = ids_pred[order]
y_preds = y_preds[order]
yvalidations = yvalidations[order]
y_preds[y_preds == -1] = 0

performance_ridge = calculate_classification_accuracy(yvalidations, y_preds)

#puts every jet value back in order
order = ids_predtrain.argsort()
ids_predtrain = ids_predtrain[order]
y_predstrain = y_predstrain[order]
yvalidationstrain = yvalidationstrain[order]

y_predstrain[y_predstrain == -1] = 0

train_performance_ridge = calculate_classification_accuracy(yvalidationstrain, y_predstrain)

print('Ridge done')
# =============================================================================
# #%% Logistic regression

for i in range(4):
    
    data_jet = np.copy(data_jetnum)


    
    data_jet[i,0] = build_poly(data_jet[i,0], degree)
    data_jet[i,2] = build_poly(data_jet[i,2], degree)

    initial_w = np.zeros(len(data_jet[i,0][0,:]))
    #w initiliser ici et passer en paramètre de running_gradient

    #pour utiliser penalized ajouter paramètre 'penalized' dans running_gradient

    weight = running_gradient(data_jet[i,1], data_jet[i,0], initial_w, lambda_, method='gradient')

    y_pred = predict_labels(weight, data_jet[i,2])
    y_preds = np.append(y_preds, y_pred)
    ids_pred = np.append(ids_pred,data_jet[i,5])
    yvalidations = np.append(yvalidations, data_jet[i,3])

    y_pred = predict_labels_notlog(weight, data_jet[i,0])
    y_predstrain = np.append(y_predstrain, y_pred)
    ids_predtrain = np.append(ids_predtrain,data_jet[i,4])
    yvalidationstrain = np.append(yvalidationstrain, data_jet[i,1])


#puts every jet value back in order
order = ids_pred.argsort()
ids_pred = ids_pred[order]
y_preds = y_preds[order]
yvalidations = yvalidations[order]
y_preds[y_preds == -1] = 0

performance_logistic_regression = calculate_classification_accuracy(yvalidations, y_preds)

#puts every jet value back in order
order = ids_predtrain.argsort()
ids_predtrain = ids_predtrain[order]
y_predstrain = y_predstrain[order]
yvalidationstrain = yvalidationstrain[order]

y_predstrain[y_predstrain == -1] = 0

train_performance_logistic_regression = calculate_classification_accuracy(yvalidationstrain, y_predstrain)

print('Logistic regression done')
 #%% Regularized Logistic regression

for i in range(4):
    
    data_jet = np.copy(data_jetnum)
    
    data_jet[i,0] = build_poly(data_jet[i,0], degree)
    data_jet[i,2] = build_poly(data_jet[i,2], degree)

    initial_w = np.zeros(len(data_jet[i,0][0,:]))
    #w initiliser ici et passer en paramètre de running_gradient

    #pour utiliser penalized ajouter paramètre 'penalized' dans running_gradient

    weight = running_gradient(data_jet[i,1], data_jet[i,0], initial_w, lambda_, method='penalized')

    y_pred = predict_labels(weight, data_jet[i,2])
    y_preds = np.append(y_preds, y_pred)
    ids_pred = np.append(ids_pred,data_jet[i,5])
    yvalidations = np.append(yvalidations, data_jet[i,3])

    y_pred = predict_labels_notlog(weight, data_jet[i,0])
    y_predstrain = np.append(y_predstrain, y_pred)
    ids_predtrain = np.append(ids_predtrain,data_jet[i,4])
    yvalidationstrain = np.append(yvalidationstrain, data_jet[i,1])


#puts every jet value back in order
order = ids_pred.argsort()
ids_pred = ids_pred[order]
y_preds = y_preds[order]
yvalidations = yvalidations[order]
y_preds[y_preds == -1] = 0

performance_regu_logistic_regression = calculate_classification_accuracy(yvalidations, y_preds)

#puts every jet value back in order
order = ids_predtrain.argsort()
ids_predtrain = ids_predtrain[order]
y_predstrain = y_predstrain[order]
yvalidationstrain = yvalidationstrain[order]

y_predstrain[y_predstrain == -1] = 0

train_performance_regu_logistic_regression = calculate_classification_accuracy(yvalidationstrain, y_predstrain)

print('Regularized Logistic regression done')

# =============================================================================

#%% summary
print("\n",'Results:')
print('performance_gradient_descent: ',performance_gradient_descent)
print('difference between test and train accuracy gradient_descent: ',train_performance_gradient_descent-performance_gradient_descent)
print('performance_sochastic_gradient_descent: ',performance_sochastic_gradient_descent)
print('difference between test and train accuracy sochastic_gradient_descent: ',performance_sochastic_gradient_descent-performance_sochastic_gradient_descent)
print('performance_least_square: ',performance_least_square)
print('difference between test and train accuracy least_square: ',train_performance_least_square-performance_least_square)
print('performance_ridge: ',performance_ridge)
print('difference between test and train accuracy ridge: ',train_performance_ridge-performance_ridge)
print('performance_logistic_regression: ', performance_logistic_regression)
print('difference between test and train accuracy logistic_regression: ', train_performance_logistic_regression-performance_logistic_regression)
print('performance_regu_logistic_regression: ', performance_regu_logistic_regression)
print('difference between test and train accuracy regu_logistic_regression: ', train_performance_regu_logistic_regression-performance_regu_logistic_regression)










print("\n",'********************************************')
