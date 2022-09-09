# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:44:28 2022

@author: alier
"""

from GP import expCorr
import numpy as np
from scipy.linalg import solve_toeplitz
import time

n = 20000


cov = expCorr(1)

covMat = cov(np.transpose([range(n)]),np.transpose([range(n)]))







from scipy.linalg._solve_toeplitz import levinson
import math





def trench(T):
    
    n = T.shape[0]
    
    c = T[0,:n-1] 
    
    a = np.concatenate((c[-1:0:-1], c))
    
    r = T[0,1:n]
    
    y = levinson(a, -r)[0]
    
    gamma = 1/(1+r@y)
    
    v = gamma*y[::-1]
    
    B = np.zeros(shape=(n,n))
    
    B[0,0] = gamma
    # B[n-1,n-1] = gamma

    B[0,1:] = v[::-1]    
    # B[1:,0] = v[::-1] 
    
    # B[n-1,0:(n-1)] = v
    # B[0:(n-1),n-1] = v
    
    for i in range(1,math.floor((n-1)/2)+1):
        for j in range(i,n-i):
            B[i,j] = B[i-1,j-1] + ( v[n-1-j]*v[n-1-i] - v[i-1]*v[j-1] )/gamma
            # B[j,i] = B[i,j]
            # B[n-i-1,n-j-1] = B[i,j]
            # B[n-j-1,n-i-1] = B[i,j]
    
    
    return(B)
    
    
 
    
    
# solve_toeplitz(covMat[0], np.identity(n))@covMat

# np.linalg.inv(covMat)@covMat

# trench(covMat)@covMat



# t1 = time.time()
# solve_toeplitz(covMat[0], np.identity(n))
# print("Levinson: "+str((time.time()-t1)/60)+" minutes")


t1 = time.time()
trench(covMat)
print("Trench: "+str(round((time.time()-t1)/60,5))+" minutes")


t1 = time.time()
np.linalg.inv(covMat)
print("Inverse: "+str(round((time.time()-t1)/60,5))+" minutes")  





