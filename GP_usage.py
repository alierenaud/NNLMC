# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:31:57 2022

@author: alier
"""

import numpy as np
from GP import LMC
from GP import expCov


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


res = 4
gridLoc = makeGrid([0,1], [0,1], res)


rhos = np.array([0.1,10])
covs = np.array([expCov(1,rho) for rho in rhos])

mean = np.array([[0],[2]])

A = np.array([[1,5],[-5,1]])


newLMC = LMC(A, mean, covs)


newLMC.rLMC(gridLoc)