#!/user/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
X=np.array([[1,2,4,5], [3,4,6,1]])
m=X.shape[1]
n=X.shape[0]

Y_prediction = np.zeros((1,m))
w = np.array([[1],[2]])
w = w.reshape(X.shape[0],1)
print("X.shape[0]="+str(n))
print("X.shape[1]="+ str(m) )
print("Y_prediction="+str(Y_prediction))
print("w = "+str(w))
print("hello cc, small assistant call ")
