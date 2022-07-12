# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:41:39 2022

@author: alier
"""


import numpy as np
from GP import mexpit_col


def pairs(K):
    arr = np.empty(shape=(int((K+1)*K/2),2), dtype=int)
    i = 0
    c = 0
    while i<K:
        j=i
        while j<K:
            arr[c] = [i,j]
            c += 1
            j += 1
        i += 1
    return arr



def g0(N,A,mu):
    

    p = A.shape[0]
    
    Y = A@np.random.normal(size=(p,N)) + np.outer(mu,np.ones(N))
    
    Z = mexpit_col(Y)
    
    
    means = np.mean(Z, axis=1)

    return np.array([np.mean(Z[pr[0]]*Z[pr[1]])/(means[pr[0]]*means[pr[1]]) for pr in pairs(p)])




def gd(N,A,mu,rhos,d):
    
    p = A.shape[0]
    
    B = np.linalg.cholesky( np.sum([np.kron([[1,np.exp(-rhos[j]*d)],[np.exp(-rhos[j]*d),1]],np.outer(A[:,j],A[:,j])) for j in range(p)], axis=0) )

    
    
    Y = B@np.random.normal(size=(2*p,N)) + np.outer(np.concatenate((mu,mu)),np.ones(N))
    
    
    
    Zeta = mexpit_col(Y[:p])
    Zxi = mexpit_col(Y[p:])
    
    
    meansEta = np.mean(Zeta, axis=1)
    meansXi = np.mean(Zxi, axis=1)
    

    return np.array([np.mean(Zeta[pr[0]]*Zxi[pr[1]])/(meansEta[pr[0]]*meansXi[pr[1]]) for pr in pairs(p)])


def gest(N,A,mu,rhos,d):
    
    if d ==0:
        return g0(N,A,mu)
    else:
        return gd(N,A,mu,rhos,d)
    
    
def gfuncest(N,A,mu,rhos,ds):
    
    return np.array([gest(N,A,mu,rhos,d) for d in ds])








