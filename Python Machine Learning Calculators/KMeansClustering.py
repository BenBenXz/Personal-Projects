# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:42:07 2021

@author: elite
"""

## Import necessary libraries
import random as rd
import numpy as np # linear algebra
from matplotlib import pyplot as plt
## Generate data
## Set centers, the model should predict similar results
center_1 = np.array([3,66])
center_2 = np.array([7,75])

X=np.array([[1,50],[2,60],[3,66],[4,68],[5,71],[6,72],[7,75],[8,82],[9,90],[10,99]])

m=X.shape[0] #number of training examples
d=X.shape[1] #number of features. Here d=2
n_iter=1000
##
K=2 # number of clusters
##
##Step 1: Initialize the centroids randomly from the data points:
Centroids=np.array([]).reshape(d,0)
for i in range(K):
 rand=rd.randint(0,m-1) # randomly pick a number from 0 to m-1
 Centroids=np.c_[Centroids,X[rand]] #concatenation along the second axis
Output={}
##Repeat step 2 till n_iter/convergence is achieved.
for i in range(n_iter):
 #Step 2.a: For each training example compute the euclidian distance from the centroid and assign the cluster based on the minimal distance
 EuclidianDistance=np.array([]).reshape(m,0)
 for k in range(K):
 # Compute the distance between the kth centroid and every data point
     tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
 # stack the K sets of Euclid distance in K columns
     EuclidianDistance=np.c_[EuclidianDistance,tempDist]
 # Center indicator: locate the column (argmin) that has the minimum distance
 C=np.argmin(EuclidianDistance,axis=1)+1
 #Step 2.b: We need to regroup the data points based on the cluster index C and store in the Output dictionary and also compute the mean of separated clusters and assign it as new centroids. Y is a temporary dictionary which stores the solution for one particular iteration.
 Y={}
 Idx1=[]
 Idx2=[]
 Idx3=[]
 for k in range(K):
     # each Y[k]: array([], shape=(2, 0), dtype=float64)
     Y[k+1]=np.array([]).reshape(d,0)
 for i in range(m):
     # Indicate and collect data X according to the Center indicator
     Y[C[i]]=np.c_[Y[C[i]],X[i]] #np.shape(Y[k])=(2, number of points nearest to kth center)
     # collect indices of each clustered group
     if C[i]==1: Idx1 += [i]
     if C[i]==2: Idx2 += [i]

 for k in range(K):
     Y[k+1]=Y[k+1].T # transpose the row-wise data to column-wise
 # Compute new centroids
 for k in range(K):
     Centroids[:,k]=np.mean(Y[k+1],axis=0)
Output=Y

## Plot data
plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Plot of data points')
plt.show()
## plot clusters
color=['red','blue','green','cyan','magenta']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
for k in range(K):
 plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
plt.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()

print("Centroids")
print(Centroids)