# -*- coding: utf-8 -*-
"""Applying functions on the separeted by jet dataset ."""

# Useful starting lines
import numpy as np
from ridge_regression import *
from preprocessing import *
from proj1_helpers import *




def create_data_with_jet(tX, y, ids_train, tXtest, ytest, ids_test ):
    """
    Take in argument a train and a test dataset and split it by the jet_num integegrer (jet_num has to be column 22)
    return data_jet a list of array train, test and ids data

    """


    for jet in range(4):

        if jet == 0:
            tx_0_train = tX[tX[:, 22] == jet]
            tx_0_test = tXtest[tXtest[:, 22] == jet]

            y_0_train = y[tX[:, 22] == jet]
            y_0_test = ytest[tXtest[:, 22] == jet]

            tx_0_train = np.delete(tx_0_train, 22, 1)
            tx_0_test = np.delete(tx_0_test, 22, 1)

            tx_0_train = np.delete(tx_0_train, -1, 1) #dernière colonne que des 0 pour jet = 0
            tx_0_test = np.delete(tx_0_test, -1, 1)   #dernière colonne que des 0 pour jet = 0

            tx_0_train, y_0_train = preprocess_(tx_0_train, y_0_train, 0.66)
            tx_0_test, y_0_test = preprocess_(tx_0_test, y_0_test, 0.66)


            ids_0_train = ids_train[tX[:, 22] == jet]
            ids_0_test = ids_test[tXtest[:, 22] == jet]

        if jet == 1:
            tx_1_train = tX[tX[:, 22] == jet]
            tx_1_test = tXtest[tXtest[:, 22] == jet]

            y_1_train = y[tX[:, 22] == jet]
            y_1_test = ytest[tXtest[:, 22] == jet]

            tx_1_train = np.delete(tx_1_train, 22, 1)
            tx_1_test = np.delete(tx_1_test, 22, 1)

            tx_1_train, y_1_train = preprocess_(tx_1_train, y_1_train, 0.66)
            tx_1_test, y_1_test = preprocess_(tx_1_test, y_1_test, 0.66)

            ids_1_train = ids_train[tX[:, 22] == jet]
            ids_1_test = ids_test[tXtest[:, 22] == jet]


        if jet == 2:
            tx_2_train = tX[tX[:, 22] == jet]
            tx_2_test = tXtest[tXtest[:, 22] == jet]

            y_2_train = y[tX[:, 22] == jet]
            y_2_test = ytest[tXtest[:, 22] == jet]

            tx_2_train = np.delete(tx_2_train, 22, 1)
            tx_2_test = np.delete(tx_2_test, 22, 1)

            tx_2_train, y_2_train = preprocess_(tx_2_train, y_2_train, 0.66)
            tx_2_test, y_2_test = preprocess_(tx_2_test, y_2_test, 0.66)


            ids_2_train = ids_train[tX[:, 22] == jet]
            ids_2_test = ids_test[tXtest[:, 22] == jet]

        if jet == 3:
            tx_3_train = tX[tX[:, 22] == jet]
            tx_3_test = tXtest[tXtest[:, 22] == jet]

            y_3_train = y[tX[:, 22] == jet]
            y_3_test = ytest[tXtest[:, 22] == jet]

            tx_3_train = np.delete(tx_3_train, 22, 1)
            tx_3_test = np.delete(tx_3_test, 22, 1)

            tx_3_train, y_3_train = preprocess_(tx_3_train, y_3_train, 0.66)
            tx_3_test, y_3_test = preprocess_(tx_3_test, y_3_test, 0.66)


            ids_3_train = ids_train[tX[:, 22] == jet]
            ids_3_test = ids_test[tXtest[:, 22] == jet]


    data_jet = [[tx_0_train, y_0_train, tx_0_test, y_0_test, ids_0_train, ids_0_test ],
             [tx_1_train, y_1_train, tx_1_test, y_1_test, ids_1_train, ids_1_test],
             [tx_2_train, y_2_train, tx_2_test, y_2_test, ids_2_train, ids_2_test],
             [tx_3_train, y_3_train, tx_3_test, y_3_test, ids_3_train, ids_3_test]]


    return data_jet
