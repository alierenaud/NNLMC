# -*- coding: utf-8 -*-
"""
Created on Wed May 18 01:00:21 2022

@author: alier
"""
import numpy as np

from MCMC_LMC import MCMC_LMC
import matplotlib.pyplot as plt


from GP import expCorr
from GP import rLMC
from GP import makeGrid



res = 15
gridLoc = makeGrid([0,1], [0,1], res)
n = gridLoc.shape[0]


rhos = np.array([5,25])
corrFuncs = np.array([expCorr(rho) for rho in rhos])

mu = np.array([0,2])
mean = np.outer(mu,np.ones(n))

A = np.array([[0.02,-0.01],[-0.01,0.02]])





resLMC = rLMC(A, corrFuncs, mean, gridLoc)

###################



sigma_prior = 60

mean_prior = 10
sd_prior = 8
var_prior = sd_prior**2

sigma_prior_mu = 10
m_prior = np.array([0,0])

alpha_prior = mean_prior**2 / var_prior
beta_prior = mean_prior/var_prior

sigma_prop_A = 1
sigma_prop_rho = 0.5

# A_init = np.identity(2)
A_init = np.linalg.inv(A)

# rho_init = np.array([10,10])
rho_init = rhos

# mu_init = np.array([0,0])
mu_init = mean[:,0]

size = 1000




import time

t0 = time.time()
A_mcmc, rho_mcmc, mu_mcmc = MCMC_LMC(resLMC, gridLoc, sigma_prior, alpha_prior, beta_prior, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, A_init, rho_init, mu_init, size)
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


# ### id constraints

# rho_mcmc = np.array([np.sort(rhos) for rhos in rho_mcmc])

# rhos

# np.mean(rho_mcmc, axis=0)

# plt.plot(rho_mcmc[:,0])
# plt.plot(rho_mcmc[:,1])

# ordre = np.array([np.argsort(rhos) for rhos in rho_mcmc])

# for i in range(size):
    
#     A_mcmc[i] = np.transpose([np.sign(A_mcmc[i,:,0])]) * A_mcmc[i]
#     A_mcmc[i] = A_mcmc[i,ordre[i]]
    
    
    
# np.linalg.inv(A)

# np.mean(A_mcmc, axis=0)

# plt.plot(A_mcmc[:,0,0])
# plt.plot(A_mcmc[:,0,1])
# plt.plot(A_mcmc[:,1,0])
# plt.plot(A_mcmc[:,1,1])    


# ###
