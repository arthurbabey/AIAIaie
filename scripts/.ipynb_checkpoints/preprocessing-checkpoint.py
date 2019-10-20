{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import math as m"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "### remove features with too many missing values\n",
    "def remove_features_with_too_many_missing_values(tX,proportion):\n",
    "\n",
    "    data_removed_features = np.copy(tX)\n",
    "    delete_ind = []\n",
    "\n",
    "    for feature_number in range(0,len(tX[0,:])):\n",
    "        #print(feature_number)\n",
    "        mask = tX[:,feature_number] == -999.\n",
    "        #print (\"number of missing values for feature number: \", feature_number,\" = \", np.sum(mask))\n",
    "        if np.sum(mask) > len(tX[:,0])*proportion:\n",
    "            delete_ind = np.append(delete_ind,feature_number)\n",
    "\n",
    "    delete_ind = delete_ind.astype(int)\n",
    "    data_removed_features = np.delete(data_removed_features,delete_ind, axis =1)\n",
    "    print(len(delete_ind), 'features removed','(features number: ', delete_ind, ')')\n",
    "    print('new shape of data:',data_removed_features.shape)\n",
    "    return data_removed_features \n",
    "\n",
    "#cool = remove_features_with_too_many_missing_values(tX,0.66)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def replace_missing_values_with_global_mean(tX):\n",
    "    \n",
    "    data_with_means_for_missing_values = np.copy(tX)\n",
    "    print(range(0,len(tX[0,:])))\n",
    "    for feature_number in range(0,len(tX[0,:])):       \n",
    "        #print(feature_number)\n",
    "        mask = tX[:,feature_number] == -999.        \n",
    "        data_with_means_for_missing_values[mask,feature_number] = np.mean(data_with_means_for_missing_values[~mask,feature_number])\n",
    "        #print ('mean = ', np.mean(data_with_means_for_missing_values[~mask,feature_number]))\n",
    "    return data_with_means_for_missing_values\n",
    "\n",
    "    \n",
    "#try_ = replace_missing_values_with_global_mean(tX)\n",
    "#print(try_.shape)\n",
    "\n",
    "#print(try_[0:50,0])\n",
    "#print('**')\n",
    "#print(tX[0:50,0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def Z_score_of_each_feature(tX):\n",
    "    Standardized_data = np.copy(tX)\n",
    "    for feature_number in range(0,len(tX[0,:])):\n",
    "        mean_ = np.mean(Standardized_data[:,feature_number])\n",
    "        std_ = np.std(Standardized_data[:,feature_number])\n",
    "        for sample_number in range(0,len(tX[:,0])):\n",
    "            Standardized_data[sample_number,feature_number] = (Standardized_data[sample_number,feature_number] - mean_) / std_\n",
    "    return Standardized_data"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
