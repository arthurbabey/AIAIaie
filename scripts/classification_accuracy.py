# -*- coding: utf-8 -*-


import numpy as np

def calculate_classification_accuracy(labels,predictions):

    num_correct = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            num_correct += 1
    accuracy = num_correct/len(labels)
    return accuracy
