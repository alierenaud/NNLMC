# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:31:57 2022

@author: alier
"""

import numpy as np
from GP import LMC
from GP import expCov
import matplotlib.pyplot as plt


def makeGrid(xlim,ylim,res):
    grid = np.ndarray((res**2,2))
    xlo = xlim[0]
    xhi = xlim[1]
    xrange = xhi - xlo
    ylo = ylim[0]
    yhi = ylim[1]
    yrange = yhi - ylo
    xs = np.arange(xlo, xhi, step=xrange/res) + xrange/res*0.5
    ys = np.arange(ylo, yhi, step=yrange/res) + yrange/res*0.5
    i=0
    for x in xs:
        j=0
        for y in ys:
            grid[i*res+j,:] = [x,y]
            j+=1
        i+=1
    return(grid)


res = 100
gridLoc = makeGrid([0,1], [0,1], res)


rhos = np.array([0.5,100])
covs = np.array([expCov(1,rho) for rho in rhos])

mean = np.array([[0],[0]])

A = np.array([[10,4],[-np.sqrt(116),0]])


newLMC = LMC(A, mean, covs)


resLMC = newLMC.rLMC(gridLoc)

#### to make plot ####







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
axs[0,0].pcolormesh(X,Y,imGP)

axs[0,0].set_xlim(0,1)
axs[0,0].set_ylim(0,1)


imGP = resLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[1,0].set_aspect('equal')
axs[1,0].pcolormesh(X,Y,imGP)

axs[1,0].set_xlim(0,1)
axs[1,0].set_ylim(0,1)


num = 500

x = np.linspace(0,5, num) 


axs[0,1].plot(x,cov1(x))
axs[1,1].plot(x,cov2(x))


plt.show()


