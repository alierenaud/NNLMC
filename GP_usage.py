# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:31:57 2022

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt
from GP import expCorr
from GP import rLMC
from GP import rCondLMC
from GP import makeGrid
from GP import mexpit_col
from GP import multinomial_col



res = 100
gridLoc = makeGrid([0,1], [0,1], res)
n = gridLoc.shape[0]

rhos = np.array([1,10])
corrFuncs = np.array([expCorr(rho) for rho in rhos])

mu = np.array([-1,-1])
mean = np.outer(mu,np.ones(n))

A = np.array([[4,3],[-5,0]])/5





resLMC = rLMC(A, corrFuncs, mean, gridLoc)
expitLMC = mexpit_col(resLMC) ### mexpit transform
mnLMC = multinomial_col(expitLMC) ### multinomial realization

#### to make plot ####


# rangeCov = 5




# def cov1(x):
#         return(A[0,0]**2*np.exp(-rhos[0]*x) + A[0,1]**2*np.exp(-rhos[1]*x))
        
# def cov2(x):
#         return(A[1,0]**2*np.exp(-rhos[0]*x) + A[1,1]**2*np.exp(-rhos[1]*x))





fig, axs = plt.subplots(2, 3)

axs[0,0].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)
axs[1,0].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)
axs[0,1].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)
axs[1,1].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)
axs[0,2].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)
axs[1,2].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)

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


# num = 500

# x = np.linspace(0,rangeCov, num) 


# axs[0,1].plot(x,cov1(x))
# axs[1,1].plot(x,cov2(x))





#### expit LMC









imGP = expitLMC[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[0,1].set_aspect('equal')
# axs[0,0].pcolormesh(X,Y,imGP) 
axs[0,1].pcolormesh(X,Y,imGP)


axs[0,1].set_xlim(0,1)
axs[0,1].set_ylim(0,1)


imGP = expitLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[1,1].set_aspect('equal')
# axs[1,0].pcolormesh(X,Y,imGP)
axs[1,1].pcolormesh(X,Y,imGP)


axs[1,1].set_xlim(0,1)
axs[1,1].set_ylim(0,1)


# num = 500

# x = np.linspace(0,rangeCov, num) 


# axs[0,1].plot(x,cov1(x))
# axs[1,1].plot(x,cov2(x))





#### multinomial LMC









imGP = mnLMC[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[0,2].set_aspect('equal')
# axs[0,0].pcolormesh(X,Y,imGP) 
axs[0,2].pcolormesh(X,Y,imGP)


axs[0,2].set_xlim(0,1)
axs[0,2].set_ylim(0,1)


imGP = mnLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[1,2].set_aspect('equal')
# axs[1,0].pcolormesh(X,Y,imGP)
axs[1,2].pcolormesh(X,Y,imGP)


axs[1,2].set_xlim(0,1)
axs[1,2].set_ylim(0,1)


# num = 500

# x = np.linspace(0,rangeCov, num) 


# axs[0,1].plot(x,cov1(x))
# axs[1,1].plot(x,cov2(x))


plt.show()


#### conditional

res = 99
locNew = makeGrid([0,1], [0,1], res)
k = locNew.shape[0]

# rhos = np.array([1,10])
# corrFuncs = np.array([expCorr(rho) for rho in rhos])

# mu = np.array([-1,-1])
meanNew = np.outer(mu,np.ones(k))

# A = np.array([[4,3],[-5,0]])/5


p = A.shape[0]
Rinvs = np.array([ np.linalg.inv(corrFuncs[j](gridLoc,gridLoc)) for j in range(p) ])

resCondLMC = rCondLMC(A, corrFuncs, mean, meanNew, gridLoc, locNew, Rinvs, resLMC)
expitCondLMC = mexpit_col(resCondLMC) ### mexpit transform
mnCondLMC = multinomial_col(expitCondLMC) ### multinomial realization




fig, axs = plt.subplots(2, 3)

axs[0,0].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)
axs[1,0].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)
axs[0,1].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)
axs[1,1].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)
axs[0,2].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)
axs[1,2].tick_params(left = False,  labelleft = False ,
                labelbottom = False, bottom = False)

imGP = resCondLMC[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[0,0].set_aspect('equal')
# axs[0,0].pcolormesh(X,Y,imGP) 
axs[0,0].pcolormesh(X,Y,imGP)


axs[0,0].set_xlim(0,1)
axs[0,0].set_ylim(0,1)


imGP = resCondLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[1,0].set_aspect('equal')
# axs[1,0].pcolormesh(X,Y,imGP)
axs[1,0].pcolormesh(X,Y,imGP)


axs[1,0].set_xlim(0,1)
axs[1,0].set_ylim(0,1)


# num = 500

# x = np.linspace(0,rangeCov, num) 


# axs[0,1].plot(x,cov1(x))
# axs[1,1].plot(x,cov2(x))





#### expit LMC









imGP = expitCondLMC[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[0,1].set_aspect('equal')
# axs[0,0].pcolormesh(X,Y,imGP) 
axs[0,1].pcolormesh(X,Y,imGP)


axs[0,1].set_xlim(0,1)
axs[0,1].set_ylim(0,1)


imGP = expitCondLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[1,1].set_aspect('equal')
# axs[1,0].pcolormesh(X,Y,imGP)
axs[1,1].pcolormesh(X,Y,imGP)


axs[1,1].set_xlim(0,1)
axs[1,1].set_ylim(0,1)


# num = 500

# x = np.linspace(0,rangeCov, num) 


# axs[0,1].plot(x,cov1(x))
# axs[1,1].plot(x,cov2(x))





#### multinomial LMC









imGP = mnCondLMC[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[0,2].set_aspect('equal')
# axs[0,0].pcolormesh(X,Y,imGP) 
axs[0,2].pcolormesh(X,Y,imGP)


axs[0,2].set_xlim(0,1)
axs[0,2].set_ylim(0,1)


imGP = mnCondLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 

axs[1,2].set_aspect('equal')
# axs[1,0].pcolormesh(X,Y,imGP)
axs[1,2].pcolormesh(X,Y,imGP)


axs[1,2].set_xlim(0,1)
axs[1,2].set_ylim(0,1)



