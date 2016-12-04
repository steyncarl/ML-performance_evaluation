# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:14:46 2016

@author: Carl Steyn
"""
import numpy as np
import os
from sklearn import svm
os.chdir('C:\\Carl\\Work\\NumerAI')

#from numpy import genfromtxt
#data = genfromtxt('numerai_training_data.csv', delimiter=',')
#data = data[1:,1:]
#np.save('data',data)

data = np.load('data.npy')
X = data[:,0:20]
y = data[:,20]
scalar = 0.6
n = int(scalar*len(X))
X_train = X[0:n,:]
X_test = X[n+1:,:]
y_train = y[0:n]
y_test = y[n+1:]
#%%
clf = svm.SVC(degree=2,probability=True,class_weight='balanced')
clf.fit(X_train,y_train)
clf.score(X_test,y_test)