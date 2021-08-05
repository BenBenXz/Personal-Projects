# -*- coding: utf-8 -*-



import numpy as np
from numpy.linalg import inv
x=np.array([[1,2], [1,6], [1,4],[1,5],[1,7]])
y=np.array([[0], [0], [1],[1],[1]])
test=np.array([[1,2],[1,6],[1,4],[1,5],[1,7]])




def LinearRegression(x,y,test):
    xt=x.transpose()
    if (x.shape[0] < x.shape[1]):
        w= xt @ inv(x @ xt) @ y
    elif (x.shape[0] > x.shape[1]):
        w= inv(xt @ x) @ xt @ y
    else:
        w= inv(x) @ y
    print("(m,d)")
    print(x.shape)
    print("Value of w is:")
    print(w)
    print("Predicted Value is:")
    predicted=test @ w
    print(predicted)
    yclass=np.sign(predicted)
    print("Class prediction")
    print(yclass)
    


print(LinearRegression(x,y,test))
    
    
    