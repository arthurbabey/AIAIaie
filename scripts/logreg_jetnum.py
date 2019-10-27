import numpy as np
from preprocessing import *
from logreg import *
from split_data import *
from helpers import *
from classification_accuracy import *
from create_data_with_jet import *
from build_polynomial import *
DATA_TRAIN_PATH = 'C:/Users/joeld/Desktop/EPFL/machine learning/AIAIaie/data/train.csv'
#DATA_TRAIN_PATH = '/Users/arthurbabey/Documents/master2/ML/ML_course/projects/project1/data'
y, tX, ids_train = load_csv_data(DATA_TRAIN_PATH)


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

lambda_ = 0.1 #paramètre pour penalized_logistic_regression
degree = 2
ids_pred = []
y_preds = []
yvalidations = []

#manière de faire pour avoir des predictions avec data_jet

for i in range(4):

    data_jet = np.copy(data_jetnum)
    data_jet[i,0] = build_poly(data_jet[i,0], degree)
    data_jet[i,2] = build_poly(data_jet[i,2], degree)

    w = np.zeros(len(data_jet[i,0][0,:]))

    #w initiliser ici et passer en paramètre de running_gradient

    #pour utiliser penalized ajouter paramètre 'penalized' dans running_gradient
    
    weight = running_gradient(data_jet[i,1], data_jet[i,0],w, lambda_, 'penalized')

    y_pred = predict_labels(weight, data_jet[i,2])
    y_preds = np.append(y_preds, y_pred)
    ids_pred = np.append(ids_pred,data_jet[i,5])
    yvalidations = np.append(yvalidations, data_jet[i,3])


#remet tout dans le bon ordre
order = ids_pred.argsort()
ids_pred = ids_pred[order]
y_preds = y_preds[order]
yvalidations = yvalidations[order]
y_preds[y_preds == -1] = 0

score = calculate_classification_accuracy(yvalidations, y_preds)

print(score)
