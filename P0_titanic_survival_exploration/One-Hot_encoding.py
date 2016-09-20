#!/usr/bin/python
#

# In this exercise we'll load the iris dataset from sklearn,
# Then we'll perform one-hot encoding on the target names.


#
# In this and the following exercises, you'll be adding train test splits to the data
# to see how it changes the performance of each classifier
#
# The code provided will load the Titanic dataset like you did in project 0, then train
# a decision tree (the method you used in your project) and a Bayesian classifier (as
# discussed in the introduction videos). You don't need to worry about how these work for
# now. 
#
# What you do need to do is import a train/test split, train the classifiers on the
# training data, and store the resulting accuracy scores in the dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')
# Limit to categorical data
X = X.select_dtypes(include=[object])

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# TODO: create a LabelEncoder object and fit it to each feature in X.
# The label encoder only takes a single feature at a time!

le = LabelEncoder()
arr=[]
for feature in X.columns.values:
    ar=le.fit_transform(X[feature])
    arr.append(ar)
    #each feature is transformed with categorical label
# arr=[n_features * n_samples]
    
arrT=map(list,zip(*arr)) #transpose arr [n_samples * n_features]

# TODO: create a OneHotEncoder object, and fit it to all of X.

enc = OneHotEncoder()
enc.fit(arrT)

#TODO: transform the categorical titanic data, and store the transformed labels in the variable `onehotlabels`
onehotlabels =enc.transform(arrT)
