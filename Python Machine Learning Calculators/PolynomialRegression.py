# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:04:23 2021

@author: elite
"""
import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
x = np.array([[1,1,3,2], [1,4,0,6], [1,-1,1,8],[1,-2,1,2],[1,-3,4,5]])
y = np.array([[1,0,0], [0,1,0], [0,1,0],[0,0,1],[1,0,0]])
test=np.array([[1,1,-3,1]])

def PolyRegression(x,y,test):
    order = 2
    poly = PolynomialFeatures(order)
    P=poly.fit_transform(x)
    test=poly.fit_transform(test)
    Pt=P.transpose()
    print("(m,d)")
    print(P.shape)
    print("P Value is:")
    print(P)
    if (P.shape[0] < P.shape[1]):
        w= Pt @ inv(P @ Pt) @ y
    else:
        w= inv(Pt @ P) @ Pt @ y
    print("The W value is")
    print(w)
    predict = test @ w
    print("Predicted y")
    print(predict)
    yclass=np.sign(predict)
    print("Class prediction")
    print(yclass)

def PolyBinaryEncoding(x,y,test):
    order = 8
    poly = PolynomialFeatures(order)
    reshape = x[:,1].reshape((len(x[:,1]),1))
    P = poly.fit_transform(reshape)
    test_reshaped = test[:,1].reshape((len(test[:,1]),1))
    test_reshaped = poly.fit_transform(test_reshaped)
    print("(m,d)")
    print(P.shape)
    Pt=P.transpose()
    print("The P matrix is")
    print(P)
    if (P.shape[0] < P.shape[1]):
        w= Pt @ inv(P @ Pt) @ y
    else:
        w= inv(Pt @ P) @ Pt @ y
    print("The W value is")
    print(w)
    predict = test_reshaped @ w
    print("Predicted y")
    print(predict)
    yclass=np.sign(predict)
    print("Class prediction")
    print(yclass)


def PolyHotEncoding(x,y,test):
    order = 3
    poly = PolynomialFeatures(order) #poly regression obj
    enc = OneHotEncoder(sparse=False) # encoding object
    reshape = x[:,1].reshape((len(x[:,1]),1))
    P = poly.fit_transform(reshape)
    #polynomial for input data
    test_reshaped = test[:,1].reshape((len(test[:,1]),1))
    test_reshaped = poly.fit_transform(test_reshaped)
    print("(m,d)")
    print(P.shape)
    
    y_onehot = enc.fit_transform(y)
    Pt=P.transpose()
    print("The P matrix is")
    print(P)
    
    if (P.shape[0] < P.shape[1]):
        w= Pt @ inv(P @ Pt) @ y_onehot
    else:
        w= inv(Pt @ P) @ Pt @ y_onehot
    print("The W value is")
    print(w)
    predict = test_reshaped @ w
    print("Predicted y")
    print(predict)
    yclass= [[1 if y == max(x) else 0 for y in x] for x in predict]
    print("Class Prediction")
    print(yclass)
    
print(PolyHotEncoding(x,y,test))
