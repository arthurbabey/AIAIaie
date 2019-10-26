import numpy as np
import math as m

def change_negativeOnes_into_Zeros_in_y(y):
    y[y == -1] = 0
    return y

### remove features with too many missing values
def remove_features_with_too_many_missing_values(tX,proportion):

    data_removed_features = np.copy(tX)
    delete_ind = []

    for feature_number in range(0,len(tX[0,:])):
        #print(feature_number)
        mask = tX[:,feature_number] == -999.
        #print ("number of missing values for feature number: ", feature_number," = ", np.sum(mask))
        if np.sum(mask) > len(tX[:,0])*proportion:
            delete_ind = np.append(delete_ind,feature_number)

    delete_ind = np.asarray(delete_ind)
    #ajout de cette pour eviter problème si le array est vide 

    delete_ind = delete_ind.astype(int)
    data_removed_features = np.delete(data_removed_features,delete_ind, axis =1)
    #print(len(delete_ind), 'features removed','(features number: ', delete_ind, ')')
    #print('new shape of data:',data_removed_features.shape)
    return data_removed_features

#cool = remove_features_with_too_many_missing_values(tX,0.66)


def replace_missing_values_with_global_mean(tX):

    data_with_means_for_missing_values = np.copy(tX)
    #print(range(0,len(tX[0,:])))
    for feature_number in range(0,len(tX[0,:])):
        #print(feature_number)
        mask = tX[:,feature_number] == -999.
        data_with_means_for_missing_values[mask,feature_number] = np.mean(data_with_means_for_missing_values[~mask,feature_number])
        #print ('mean = ', np.mean(data_with_means_for_missing_values[~mask,feature_number]))
    return data_with_means_for_missing_values


#try_ = replace_missing_values_with_global_mean(tX)
#print(try_.shape)

#print(try_[0:50,0])
#print('**')
#print(tX[0:50,0])


def Z_score_of_each_feature(tX):
    Standardized_data = np.copy(tX)
    for feature_number in range(0,len(tX[0,:])):
        mean_ = np.mean(Standardized_data[:,feature_number])
        std_ = np.std(Standardized_data[:,feature_number])
        for sample_number in range(0,len(tX[:,0])):
            Standardized_data[sample_number,feature_number] = (Standardized_data[sample_number,feature_number] - mean_) / std_
    return Standardized_data
