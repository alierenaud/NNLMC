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

### inversion utilities

def blockwiseInv(Sigma,Sigma22_inv):

    A = np.array([[Sigma[0,0]]])
    B = np.array([Sigma[0,1:Sigma.shape[0]]])
    C = np.transpose(B)


    BD = B@Sigma22_inv
    BDC = BD@C
    DC = np.transpose(BD)
    AmBDC = 1/(A-BDC) 
    AmBDCBD = AmBDC@BD

    Sigma_inv = np.block([[AmBDC,-AmBDCBD],[-DC@AmBDC,Sigma22_inv + DC@AmBDCBD]])
    
    return(Sigma_inv)




### gaussian process

class GP:  
    
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        
    
    def meanVec(self, loc):
        nloc = loc.shape[0]
        mu = np.zeros(nloc)
        i=0
        for x in loc:
            mu[i] = self.mean(x)
            i+=1
        return(mu)
    
    
    def covMatrix(self, loc):
        return(self.cov(loc,loc))
    
    def rGP(self, loc):
        Sigma = self.covMatrix(loc)
        nloc = loc.shape[0]
        Z = random.normal(size=nloc)
        L = np.linalg.cholesky(Sigma)
        return(np.matmul(L,Z)+self.meanVec(loc))
    
    def rCondGP(self, locPred, locObs, valObs):
        loc = np.concatenate((locPred, locObs))
        Sigma = self.covMatrix(loc)
        # mu = self.meanVec(loc)
        
        nlocPred = locPred.shape[0]
        nlocObs = locObs.shape[0]
        nloc = nlocPred + nlocObs
        
        # mu1 = mu[0:nlocPred]
        # mu2 = mu[nlocPred:nloc]
        
        Sigma11 = Sigma[0:nlocPred, 0:nlocPred]
        Sigma12 = Sigma[0:nlocPred, nlocPred:nloc]
        Sigma21 = np.transpose(Sigma12)
        Sigma22 = Sigma[nlocPred:nloc, nlocPred:nloc]
        
        invSigma22 = np.linalg.inv(Sigma22)
        
        mu1c2 = np.matmul(np.matmul(Sigma12,invSigma22),  valObs)
        Sigma1c2 = Sigma11 - np.matmul(np.matmul(Sigma12,invSigma22),  Sigma21)
        
        L = np.linalg.cholesky(Sigma1c2)
        Z = random.normal(size=(nlocPred,1))
        return(np.matmul(L,Z)+mu1c2)
    
    def rCondGP1DSigma(self, locPred, locObs, valObs, Sigma, Sigma_inv):

        # mu = self.meanVec(loc)
        

        # nlocObs = locObs.shape[0]
        
        # mu1 = mu[0:nlocPred]
        # mu2 = mu[nlocPred:nloc]
        
        Sigma11 = self.cov(locPred,locPred)
        Sigma12 =  self.cov(locPred,locObs)
        Sigma21 = np.transpose(Sigma12)
        Sigma22 = Sigma
        
        newSigma = np.block([[Sigma11,Sigma12],[Sigma21,Sigma22]])
        
        invSigma22 = Sigma_inv
        
        newSigma_inv = blockwiseInv(newSigma,Sigma_inv)
        
        mu1c2 = Sigma12@invSigma22@valObs
        Sigma1c2 = Sigma11 - Sigma12@invSigma22@Sigma21
        
        L = np.linalg.cholesky(Sigma1c2)
        Z = random.normal(size=(1,1))
        return(np.matmul(L,Z)+mu1c2, newSigma, newSigma_inv)
    
    
def zeroMean(x):
    return(0)     
    
### LMC

class LMC:  
    
    def __init__(self, A, mean, covs):
        self.coregMat = A
        self.mean = mean
        self.GPs = np.array([GP(zeroMean,cov) for cov in covs])    
        
    def rLMC(self, loc):
        vs = np.array([gp.rGP(loc) for gp in self.GPs])
        return(np.matmul(self.coregMat,vs)+self.mean)
        

    
    
def gaussianCov(sigma2,l):
    def evalCov(x,y):

        
        return(sigma2*np.exp(-distance_matrix(x,y)**2/2/l**2))
    return(evalCov)

def expCov(tau,rho):
    def evalCov(x,y):

        
        return(1/tau*np.exp(-distance_matrix(x,y)*rho))
    return(evalCov)

def indCov(sigma2):
    def evalCov(x,y):
        if np.linalg.norm(x-y)==0:
            return(sigma2)
        else:
            return(0)
    return(evalCov)     

def rMultNorm(mu,Sigma):
    Z = random.normal(size=(Sigma.shape[0],1))
    L = np.linalg.cholesky(Sigma)
    return(np.matmul(L,Z)+mu)
    
    

    
    
    
