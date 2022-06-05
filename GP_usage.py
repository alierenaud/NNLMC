# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:31:57 2022

@author: alier
"""

import numpy as np
from GP import LMC
from GP import expCov
import matplotlib.pyplot as plt
from GP import makeGrid
from GP import mexpit_col
from GP import multinomial_col



res = 100
gridLoc = makeGrid([0,1], [0,1], res)


rhos = np.array([1,10])
covs = np.array([expCov(1,rho) for rho in rhos])

mean = np.array([[-1],[-1]])

A = np.array([[4,3],[-5,0]])/5


newLMC = LMC(A, mean, covs)


resLMC = newLMC.rLMC(gridLoc)
expitLMC = mexpit_col(resLMC) ### mexpit transform
mnLMC = multinomial_col(expitLMC) ### multinomial realization

#### to make plot ####


rangeCov = 5




def cov1(x):
        return(A[0,0]**2*np.exp(-rhos[0]*x) + A[0,1]**2*np.exp(-rhos[1]*x))
        
def cov2(x):
        return(A[1,0]**2*np.exp(-rhos[0]*x) + A[1,1]**2*np.exp(-rhos[1]*x))





fig, axs = plt.subplots(2, 2)

imGP = resLMC[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[0,0].set_aspect('equal')
# axs[0,0].pcolormesh(X,Y,imGP) 
axs[0,0].pcolormesh(X,Y,imGP)


axs[0,0].set_xlim(0,1)
axs[0,0].set_ylim(0,1)


imGP = resLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[1,0].set_aspect('equal')
# axs[1,0].pcolormesh(X,Y,imGP)
axs[1,0].pcolormesh(X,Y,imGP)


axs[1,0].set_xlim(0,1)
axs[1,0].set_ylim(0,1)


num = 500

x = np.linspace(0,rangeCov, num) 


axs[0,1].plot(x,cov1(x))
axs[1,1].plot(x,cov2(x))


plt.show()


#### expit LMC







fig, axs = plt.subplots(2, 2)

imGP = expitLMC[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[0,0].set_aspect('equal')
# axs[0,0].pcolormesh(X,Y,imGP) 
axs[0,0].pcolormesh(X,Y,imGP)


axs[0,0].set_xlim(0,1)
axs[0,0].set_ylim(0,1)


imGP = expitLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[1,0].set_aspect('equal')
# axs[1,0].pcolormesh(X,Y,imGP)
axs[1,0].pcolormesh(X,Y,imGP)


axs[1,0].set_xlim(0,1)
axs[1,0].set_ylim(0,1)


num = 500

x = np.linspace(0,rangeCov, num) 


axs[0,1].plot(x,cov1(x))
axs[1,1].plot(x,cov2(x))


plt.show()


#### multinomial LMC







fig, axs = plt.subplots(2, 2)

imGP = mnLMC[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[0,0].set_aspect('equal')
# axs[0,0].pcolormesh(X,Y,imGP) 
axs[0,0].pcolormesh(X,Y,imGP)


axs[0,0].set_xlim(0,1)
axs[0,0].set_ylim(0,1)


imGP = mnLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[1,0].set_aspect('equal')
# axs[1,0].pcolormesh(X,Y,imGP)
axs[1,0].pcolormesh(X,Y,imGP)


axs[1,0].set_xlim(0,1)
axs[1,0].set_ylim(0,1)


num = 500

x = np.linspace(0,rangeCov, num) 


axs[0,1].plot(x,cov1(x))
axs[1,1].plot(x,cov2(x))


plt.show()

