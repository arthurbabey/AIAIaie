
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
lambdas_vector = np.logspace(-3, 0, num=15)
degree = 2

#%% manière de faire pour avoir des predictions avec data_jet
score = []
train_score = []
for lambda_ in lambdas_vector:
    print('Computing score for lambda = ',lambda_)
    ids_pred = []
    y_preds = []
    y_preds_train = []
    yvalidations = []
    ytrains = []
    for i in range(4):
    
        data_jet = np.copy(data_jetnum)
        data_jet[i,0] = build_poly(data_jet[i,0], degree)
        data_jet[i,2] = build_poly(data_jet[i,2], degree)
    
        w = np.zeros(len(data_jet[i,0][0,:]))
    
        #w initiliser ici et passer en paramètre de running_gradient
    
        #pour utiliser penalized ajouter paramètre 'penalized' dans running_gradient
        
        weight = running_gradient(data_jet[i,1], data_jet[i,0],w, lambda_,method = 'penalized')
    
        y_pred = predict_labels(weight, data_jet[i,2])
        y_preds = np.append(y_preds, y_pred)
        
        y_pred_train = predict_labels(weight, data_jet[i,0])
        y_preds_train = np.append(y_preds_train, y_pred_train)
        
        ids_pred = np.append(ids_pred,data_jet[i,5])
        
        yvalidations = np.append(yvalidations, data_jet[i,3])
        ytrains = np.append(ytrains, data_jet[i,1])
    
    
    #remet tout dans le bon ordre
    order = ids_pred.argsort()
    ids_pred = ids_pred[order]
    
    y_preds = y_preds[order]
    yvalidations = yvalidations[order]
    yvalidations[yvalidations==0] = -1
    
    y_preds_train = y_preds_train[order]
    ytrains = ytrains[order]
    ytrains[ytrains==0] = -1
    
    score = np.append(score,calculate_classification_accuracy(yvalidations, y_preds))
    train_score = np.append(train_score,calculate_classification_accuracy(ytrains, y_preds))

print('Scores computed')
#%% Plots
def visualization_perf_wrt_lambdas(lambdas, train_perf, test_perf):
    """visualization the curves of train/test classification accuracy."""
    plt.semilogx(lambdas, train_perf, marker=".", color='b', label='train performance')
    plt.semilogx(lambdas, test_perf, marker=".", color='r', label='test performance')
    plt.xlabel("Lambda")
    plt.ylabel("Performance")
    plt.title("Hyperparamter lambda")
    plt.legend(loc=1)
    plt.grid(True)
    plt.savefig("Hyperparameter lambda")
    
visualization_perf_wrt_lambdas(lambdas_vector, train_score, score )
print("\n",'********************************************')
