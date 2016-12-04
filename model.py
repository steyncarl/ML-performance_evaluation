# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:14:46 2016

@author: Carl Penis
"""
import numpy as np
from sklearn import svm
import pandas as pd


data = pd.read_csv("/home/cedric/Documents/Projects/Numerai/November/data/train.csv")
data = np.array(data)
X = data[:,0:20]
y = data[:,21]
scalar = 0.6
n = int(scalar*len(X))
X_train = X[0:n,:]
X_test = X[n+1:,:]
y_train = y[0:n]
y_test = y[n+1:]

clf = svm.SVC(degree       = 2, 
              probability  = True, 
              class_weight = 'balanced', 
              verbose      = True)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)