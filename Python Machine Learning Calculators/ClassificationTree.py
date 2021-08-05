#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set test_size
set dtree -> criterion & max depth
if it calls for, create 2 new np.array for training and testing data

@author: thomas
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
    
def Tut9_Q4_yeo():
    
    
    # split dataset
    X_train= np.array([[4, 2, 3, 1, 2, 1, 4, 3, 5, 6, 5, 8, 7]])
    X_test= np.array([[4, 2, 3, 1, 2, 1, 4, 3, 5, 6, 5, 8, 7]])
    y_train= np.array([[4, 2, 3, 1, 2, 1, 4, 3, 5, 6, 5, 8, 7]])
    y_test = np.array([[4, 2, 3, 1, 2, 1, 4, 3, 5, 6, 5, 8, 7]])
    
    # fit tree
    dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
    dtree = dtree.fit(X_train, y_train)
    
    # predict
    y_trainpred = dtree.predict(X_train)
    y_testpred = dtree.predict(X_test)
    
    # print accuracies
    print("Training accuracy: ", metrics.accuracy_score(y_train, y_trainpred))
    print("Test accuracy: ", metrics.accuracy_score(y_test, y_testpred))    


    
if __name__ == '__main__':
    Tut9_Q4_yeo()