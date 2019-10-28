#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


import matplotlib.pyplot as plt
import numpy as np
from preprocessing import *
from logreg import *
from split_data import *
from helpers import *
from classification_accuracy import *
from create_data_with_jet import *
from build_polynomial import *
from crossval import *

print("\n",'********************************************')
#%% Data loading
DATA_TRAIN_PATH = '/Users/benoithohl/Desktop/epfl/master_epfl/Ma3/Machine_learning/AIAIaie/data/train.csv'
#DATA_TRAIN_PATH = '/Users/arthurbabey/Documents/master2/ML/ML_course/projects/project1/data/train.csv'
y, tX, ids_train = load_csv_data(DATA_TRAIN_PATH)
print('Data loaded')

#%% preprocessing
trainx,trainy,idstrain,validationx,validationy,idstest = split_data(tX, y, ids_train, 0.8, seed=1)

"""
    data_jetnum ressemble a ça

    data_jetnum =  [[tx_0_train, y_0_train, tx_0_test, y_0_test, ids_0_train, ids_0_test ],
                [tx_1_train, y_1_train, tx_1_test, y_1_test, ids_1_train, ids_1_test],
                [tx_2_train, y_2_train, tx_2_test, y_2_test, ids_2_train, ids_2_test],
                [tx_3_train, y_3_train, tx_3_test, y_3_test, ids_3_train, ids_3_test]]
"""


data_jetnum = create_data_with_jet(trainx,trainy, idstrain, validationx, validationy, idstest)
data_jetnum = np.asarray(data_jetnum)




"""

    suite: manière de faire pour avoir des predictions avec data_jet
    autre paramètre (batchsize, max_iter...) défini dans logreg.py/running_gradient

"""

#paramètre pour penalized_logistic_regression
lambdas_vector = np.logspace(-4, 0, num=7)
degree_vector = np.arange(1,6)
k_fold = 4

#%% manière de faire pour avoir des predictions avec data_jet
test_score = np.zeros((4,len(lambdas_vector),len(degree_vector),k_fold))
train_score = np.zeros((4,len(lambdas_vector),len(degree_vector),k_fold))
ids_pred = []
ids_pred_train = []
y_preds = []
y_preds_train = []
yvalidations = []
ytrains = []
for i in range(4):
    j=0;
    for lambda_ in lambdas_vector:
        m=0
        for degree in degree_vector:
            data_jet = np.copy(data_jetnum)
            
            k_indices = build_k_indices(data_jet[i,1], k_fold, seed = 1)
            for k in range(k_fold):
                test_score[i,j,m,k],train_score[i,j,m,k] = cross_validation_log_reg(data_jet[i,1], data_jet[i,0] , k_indices, k, lambda_, degree, method='penalized')
            m=m+1    
        j=j+1

std_of_test_scores_over_folds = np.std(test_score, axis = 3)

average_train_score = np.mean(train_score, axis = 3)
average_test_score = np.mean(test_score, axis = 3)



optimal_lambda_degree_pair = np.zeros((4,2))
for i in range(4):    
    best_score_indix = np.unravel_index(np.argmax(average_test_score[i,:,:], axis=None),average_test_score[i,:,:].shape)
    optimal_lambda_degree_pair[i,0] =  lambdas_vector[best_score_indix[0]]
    optimal_lambda_degree_pair[i,1] =  degree_vector[best_score_indix[1]]
print('youuuuuuuupi :D')