# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:33:48 2022

@author: alier
"""

import numpy as np
from numpy import random

from MCMC_LMC import MCMC_LMC_MN
import matplotlib.pyplot as plt


from GP import expCorr
from GP import rLMC
from GP import makeGrid
from GP import mexpit_col
from GP import multinomial_col



res = 15
gridLoc = makeGrid([0,1], [0,1], res)
n = gridLoc.shape[0]


rhos = np.array([2,10])
corrFuncs = np.array([expCorr(rho) for rho in rhos])

mu = np.array([-1,-1])
mean = np.outer(mu,np.ones(n))

A = np.array([[4,3],[-5,0]])/10





resLMC = rLMC(A, corrFuncs, mean, gridLoc)
expitLMC = mexpit_col(resLMC) ### mexpit transform
mnLMC = multinomial_col(expitLMC) ### multinomial realization

###################



sigma_prior = 10 ### for A

mean_prior = 6
sd_prior = 4
var_prior = sd_prior**2

sigma_prior_mu = 10
m_prior = np.array([0,0])

alpha_prior = mean_prior**2 / var_prior
beta_prior = mean_prior/var_prior

sigma_prop_A = 0.2
sigma_prop_rho = 0.1
sigma_mom_V = 0.5

delta = 0.005
L = 20

# A_init = np.identity(2)
A_init = np.linalg.inv(A)

# rho_init = np.array([10,10])
rho_init = rhos

# mu_init = np.array([0,0])
mu_init = mean[:,0]

p = mnLMC.shape[0]
n = mnLMC.shape[1]

# V_init = newLMC.rLMC(gridLoc)
V_init = np.outer(mu_init,np.ones(n)) + 0.1*random.normal(size=(p,n))
# V_init = resLMC

size = 10000




import time

t0 = time.time()
A_mcmc, rho_mcmc, mu_mcmc, V_mcmc = MCMC_LMC_MN(mnLMC, gridLoc, sigma_prior, alpha_prior, beta_prior, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, sigma_mom_V, delta, L, A_init, rho_init, mu_init, V_init, size, diag_V=True)
t1 = time.time()


total1 = t1-t0


print(np.linalg.inv(A))

print(np.mean(A_mcmc, axis=0))

plt.plot(A_mcmc[:,0,0])
plt.plot(A_mcmc[:,0,1])
plt.plot(A_mcmc[:,1,0])
plt.plot(A_mcmc[:,1,1])

plt.show()

print(rhos)

print(np.mean(rho_mcmc, axis=0))

plt.plot(rho_mcmc[:,0])
plt.plot(rho_mcmc[:,1])

plt.show()

print(mu)

print(np.mean(mu_mcmc, axis=0))

plt.plot(mu_mcmc[:,0])
plt.plot(mu_mcmc[:,1])

plt.show()

# plt.plot(V_mcmc[:,0,0])
# plt.plot(V_mcmc[:,1,0])

# mnLMC[:,0]

# def nbMove(V):
#     n=0
#     lgth = V.shape[0]
#     for i in range(lgth-1):
#         if V[i]!=V[i+1]:
#             n+=1
#     return(n)

# nbMove(V_mcmc[:,0,0])



# for i in range(n):

#     if mnLMC[0,i] == 1:
#         plt.plot(V_mcmc[:,0,i], color="tab:orange")
#     else:    
#         plt.plot(V_mcmc[:,0,i], color="tab:blue")
    
#     if mnLMC[1,i] == 1:
#         plt.plot(V_mcmc[:,1,i], color="tab:orange")
#     else:    
#         plt.plot(V_mcmc[:,1,i], color="tab:blue")
#     plt.show()






meanLMC = np.mean(V_mcmc, axis=0)


########







# i=0
# while i < size:

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal')
    
#     imGP = V_mcmc[i,0].reshape(res,res)
    
#     x = np.linspace(0,1, res+1) 
#     y = np.linspace(0,1, res+1) 
#     X, Y = np.meshgrid(x,y) 
        
    
#     ax.set_aspect('equal')
        
#     ff = ax.pcolormesh(X,Y,imGP)
    
#     fig.colorbar(ff) 
    
#     plt.show()
    
#     i+=100


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

imGP = resLMC[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax.set_aspect('equal')
    
ff = ax.pcolormesh(X,Y,imGP)

fig.colorbar(ff) 

plt.show()   

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

imGP = meanLMC[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax.set_aspect('equal')
    
ff = ax.pcolormesh(X,Y,imGP)

fig.colorbar(ff) 

plt.show() 
    
# i=0
# while i < size:

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal')
    
#     imGP = V_mcmc[i,1].reshape(res,res)
    
#     x = np.linspace(0,1, res+1) 
#     y = np.linspace(0,1, res+1) 
#     X, Y = np.meshgrid(x,y) 
        
    
#     ax.set_aspect('equal')
        
#     ff = ax.pcolormesh(X,Y,imGP)
    
#     fig.colorbar(ff) 
    
#     plt.show()
    
#     i+=100


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

imGP = resLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax.set_aspect('equal')
    
ff = ax.pcolormesh(X,Y,imGP)

fig.colorbar(ff) 

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

imGP = meanLMC[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax.set_aspect('equal')
    
ff = ax.pcolormesh(X,Y,imGP)

fig.colorbar(ff) 

plt.show()
