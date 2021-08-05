#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:40:16 2020

@author: thomas
"""
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from numpy.linalg import inv
  
def Tut7_Q2_yeo():
    
    '''
    %true model: y = f(x) + epsilon = 0.01 + 0.1x + 0.05x^2 + 0.2*randn(1);
    '''
     
    # simulation parameters
    np.random.seed(5)
    num_training_samples = 10;  
    num_test_samples = 1;
    num_training_trials = 10;
    max_order = 5;
    np.set_printoptions(precision=4)
    
    #sample random test set; Notice test set is fixed throughout
    test_x = np.random.uniform(-10, 10, num_test_samples);
    test_x.sort();
    test_y = quadratic_model(test_x)
    test_y_nonoise = quadratic_model_without_noise(test_x)
    
    # Create test set regressors P; Note that order 1 is linear, order 2 is 
    # quadratic, etc
    P_test = [] #initialize empty list
    for order in range(1, max_order+1): 
        current_regressors = np.zeros([len(test_x), order+1])
        current_regressors[:,0] = np.ones(len(test_x));
        for i in range(1, order+1):
            current_regressors[:,i] = np.power(test_x, i)
        P_test.append(current_regressors)
        
    # Perform training trials, i.e., sample training set D
    prediction_mat = np.zeros([max_order, num_test_samples, num_training_trials])
    for trial in range(num_training_trials):
        
        # sample random training set
        train_x = np.random.uniform(-10, 10, num_training_samples);
        train_x.sort();
        train_y = quadratic_model(train_x)
        
        for order in range(1, max_order+1):
            
            # Create training regressors P
            P_train = np.zeros([len(train_x), order+1])
            P_train[:,0] = np.ones(len(train_x));
            for i in range(1, order+1):
                P_train[:,i] = np.power(train_x, i)

            # Estimate regression coefficients w
            # Note that @ is the same as np.matmul                
            w = (inv(P_train.T @ P_train) @ P_train.T) @ train_y

            # Predict test set
            test_y_hat = np.matmul(P_test[order-1], w)
            prediction_mat[order-1, :, trial] = test_y_hat



    # Compute Bias: [E_D (fhat_D(x)) - f(x)]^2
    model_bias_sq = np.zeros([max_order, 1])
    for order in range(1, max_order+1): 

        # Compute test prediction averaged across trials 
        average_test_prediction = np.mean(prediction_mat[order-1, :, :], 1)
        
        # Compute bias^2 for each test sample x
        bias_sq_x = np.power(average_test_prediction - test_y_nonoise, 2);

        # Average variance across all test samples
        model_bias_sq[order-1] = np.mean(bias_sq_x)
        
    print('Bias squared: ', model_bias_sq.T)

    # Compute Variance
    model_variance = np.zeros([max_order, 1])
    for order in range(1, max_order+1): 
        
        # Compute test prediction averaged across trials 
        average_test_prediction = np.mean(prediction_mat[order-1, :, :], 1)

        # Compute variance around the average test prediction for each test
        # sample x
        rep_avg_prediction = np.matlib.repmat(average_test_prediction, num_training_trials, 1).T
        variance_x = np.mean(np.power((prediction_mat[order-1, :, :] - rep_avg_prediction),2),1)
    
        # Average variance across all test samples
        model_variance[order-1] = np.mean(variance_x);    
        
    print('Variance: ', model_variance.T)

    # Compute MSE
    MSE = np.zeros([max_order, 1])
    for order in range(1, max_order+1): 
        
        # predictions from a particular order
        test_prediction = prediction_mat[order-1, :, :]
        
        # Compute MSE
        error_per_sample = test_prediction - np.matlib.repmat(test_y, num_training_trials, 1).T
        MSE[order-1] = np.mean(np.power(error_per_sample,2))
    
    print('MSE: ', MSE.T)

    # Bias+Variance tradeoff: Recall that 0.04 is variance of the noise
    MSE_bias_variance = model_bias_sq + model_variance + 0.04; 
    print('Bias-variance formula: ', MSE_bias_variance.T)


def quadratic_model(x):

    y = 0.01 + 0.1*x + 0.05*x**2 + np.random.normal(0,0.2,len(x))
    return(y)

def quadratic_model_without_noise(x):

    y = 0.01 + 0.1*x + 0.05*x**2
    return(y)



if __name__ == '__main__':
    Tut7_Q2_yeo() 














