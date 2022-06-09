# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:42:08 2020

@author: alierenaud
"""

import numpy as np
from numpy import random
import numpy.linalg
import numpy.matlib
from scipy.spatial import distance_matrix


    
### LMC


def rLMC(A, corrFuncs, mean, locs):
    
    p = A.shape[0]
    n = locs.shape[0]
    
    Rs = np.array([ corrFuncs[j](locs,locs) for j in range(p) ])
    Xs = np.array([np.linalg.cholesky(Rs[j])@random.normal(size=n) for j in range(p)])
    
    
        
    return( A @ Xs + mean)


def expCorr(rho):
    def evalCov(x,y):

        
        return(np.exp(-distance_matrix(x,y)*rho))
    return(evalCov)



    
    
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

def mexpit(v):
    
    ev = np.exp(v)
    sv = np.sum(ev)
    
    return(ev/(1+sv))


def mexpit_col(V):
    P = np.zeros(shape = V.shape)
    n = V.shape[1]
    for i in range(n):
        P[:,i] = mexpit(V[:,i])
        
    return(P)


def multinomial_col(P):
    
    Pfull = np.vstack([P,1-np.sum(P,axis=0)])
    Y = np.zeros(shape = Pfull.shape)
    n = P.shape[1]
    
    for i in range(n):
        Y[:,i] = random.multinomial(1, Pfull[:,i])
        
    return(Y[:-1])
    
    
    
