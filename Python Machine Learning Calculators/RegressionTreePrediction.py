#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
input the array size at X and y

"""

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
    
def Tut9_Q3_yeo():
    

    
    X = np.array([1, 0.8, 2, 2.5, 3, 4, 4.2, 6, 6.3, 7, 8, 8.2, 9]) #input the array values
    y = np.array([2, 3, 2.5, 1, 2.3, 2.8, 1.5, 2.6, 3.5, 4, 3.5, 5, 4.5]) #input array values
    # scikit decision tree regressor
    scikit_tree = DecisionTreeRegressor(criterion='mse', max_depth=2)
    scikit_tree.fit(X.reshape(-1,1), y) # reshape necessary because tree expects 2D array 
    scikit_tree_predict = scikit_tree.predict(X.reshape(-1,1))
    

    print(scikit_tree_predict) #comparison to scikit predicted value
    
def our_own_tree(y):
    
    # split data at first level
    # L stands for left, R stands for right
    yL, yR = find_best_split(y)
    
    # split data at second level
    yLL, yLR = find_best_split(yL)
    yRL, yRR = find_best_split(yR)
    
    # compute prediction 
    yLL_pred = np.mean(yLL)*np.ones(len(yLL))
    yLR_pred = np.mean(yLR)*np.ones(len(yLR))
    yRL_pred = np.mean(yRL)*np.ones(len(yRL))
    yRR_pred = np.mean(yRR)*np.ones(len(yRR))
    y_pred = np.concatenate([yLL_pred, yLR_pred, yRL_pred, yRR_pred])
    
    return y_pred
    
def find_best_split(y):
    
    # index represents last element in the below threshold node
    sq_err_vec = np.zeros(len(y)-1)    
    for index in range(0, len(y)-1):
        
        # split the data
        data_below_threshold = y[:index+1]
        data_above_threshold = y[index+1:]
        
        # Compute estimate
        mean_below_threshold = np.mean(data_below_threshold)
        mean_above_threshold = np.mean(data_above_threshold)
        
        # Compute total square error
        # Note that MSE = total square error divided by number of data points
        below_sq_err = np.sum(np.square(data_below_threshold - mean_below_threshold))
        above_sq_err = np.sum(np.square(data_above_threshold - mean_above_threshold))
        sq_err_vec[index] = below_sq_err + above_sq_err
    
    best_index = np.argmin(sq_err_vec)
    yL = y[:best_index+1]
    yR = y[best_index+1:]
    return yL, yR

if __name__ == '__main__':
    Tut9_Q3_yeo()