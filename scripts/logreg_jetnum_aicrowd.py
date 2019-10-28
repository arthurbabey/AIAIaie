import numpy as np
from preprocessing import *
from logreg import *
from split_data import *
from helpers import *
from classification_accuracy import *
from create_data_with_jet import *
from build_polynomial import *
DATA_TRAIN_PATH = '/Users/arthurbabey/Documents/master2/ML/ML_course/projects/project1/data/train.csv'
DATA_TEST_PATH = '/Users/arthurbabey/Documents/master2/ML/ML_course/projects/project1/data/test.csv'

y, tX, ids_train = load_csv_data(DATA_TRAIN_PATH)
ytest, tXtest, ids_test = load_csv_data(DATA_TEST_PATH)



data_jetnum = create_data_with_jet(tX,y, ids_train, tXtest, ytest, ids_test)
data_jetnum = np.asarray(data_jetnum)

lambda_ = 0.01
degree = 3
ids_pred = []
y_preds = []

#manière de faire pour avoir des predictions avec data_jet

for i in range(4):

    data_jet = np.copy(data_jetnum)
    data_jet[i,0] = build_poly(data_jet[i,0], degree)
    data_jet[i,2] = build_poly(data_jet[i,2], degree)

    w = np.zeros(len(data_jet[i,0][0,:]))

    weight = running_gradient(data_jet[i,1], data_jet[i,0], w, lambda_, 'gradient')
    y_pred = predict_labels(weight, data_jet[i,2])
    y_preds = np.append(y_preds, y_pred)
    ids_pred = np.append(ids_pred,data_jet[i,5])


print('Boucle fini')

order = ids_pred.argsort()
ids_pred = ids_pred[order]
y_preds = y_preds[order]


create_csv_submission(ids_test, y_preds, 'logregtoaicrowd.csv')
