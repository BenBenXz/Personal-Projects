# -*- coding: utf-8 -*-
"""
Created on Mon May  3 12:46:24 2021

@author: elite
"""

import numpy as np

x=np.array([0, 0.3, 2.5, 3.3, 4.4, 4.8, 5.3, 6.7, 7, 7.2])
y=np.array([1, 1, 2, 2, 1, 1, 2, 2, 2, 2])

def mse (x,y):
    total1=0
    total2=0
    count1=0
    count2=0
    root=0
    rootmae=0
    for i in range (0,(len(y))):
        if x[i] > 5:
            total1= total1 + y[i]
            count1+=1
        if x[i] <= 5:
            total2=total2 + y[i]
            count2+=1
        root += (y[i]-np.average(y))**2
        rootmae += abs(y[i]-np.average(y))
            
    avg1 = total1 / count1
    avg2 = total2 / count2
    summ1 = 0
    ab1=0
    summ2 = 0
    ab2=0
    for i in range (0,(len(y))):
        if x[i] > 5:
            summ1 += (y[i]-avg1)**2
            ab1 += abs(y[i]-avg1)
        if x[i] <= 5:
            summ2 += (y[i]-avg2)**2
            ab2 += abs(y[i]-avg2)
   
    print("MSE of Condition 1:")
    print(summ1/count1)
    print("\nMAE of condition 1")
    print(ab1/count1)
    
    print("\nMSE of Condition 2:")
    print(summ2/count2)
    print("\nMAE of condition 2")
    print(ab2/count2) 
    
    print("\nRoot MSE")
    print(root/len(y))
    print("\nRoot MAE")
    print(rootmae/len(y))

    
print(mse(x,y))