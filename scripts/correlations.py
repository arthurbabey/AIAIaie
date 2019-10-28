# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
from preprocessing import *
from plots import *
from split_data import *
from helpers import *



DATA_TRAIN_PATH = 'C:/Users/joeld/Desktop/EPFL/machine learning/AIAIaie/data/train.csv' # TODO: download train data and supply path here  
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#preprocess data, removing features with too many -999
Data = remove_features_with_too_many_missing_values(tX,0.66)
Data = replace_missing_values_with_global_mean(Data)
ZData = Z_score_of_each_feature(Data)

#create a correlation matrix and create image of the matrix
cov = np.corrcoef(np.transpose(trainx))
plt.imshow(cov,cmap='Greys')
plt.colorbar()
plt.savefig('corr.png')