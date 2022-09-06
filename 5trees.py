# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:27:52 2022

@author: alier
"""



import numpy as np
import matplotlib.pyplot as plt

from MCMC_LMC import MCMC_LMC_MN_POIS
from gfunc_est import gfuncest

### import 5 types of trees

maple = np.loadtxt("maple.csv", delimiter=",")
hickory = np.loadtxt("hickory.csv", delimiter=",")
blackoak = np.loadtxt("blackoak.csv", delimiter=",")
redoak = np.loadtxt("redoak.csv", delimiter=",")
whiteoak = np.loadtxt("whiteoak.csv", delimiter=",")

trees = np.concatenate((maple,hickory,blackoak,redoak,whiteoak))

### construct type matrix

n_maple = maple.shape[0]
n_hickory = hickory.shape[0]
n_blackoak = blackoak.shape[0]
n_redoak = redoak.shape[0]
n_whiteoak = whiteoak.shape[0]

maple_type = np.array([np.ones(n_maple),np.zeros(n_maple),np.zeros(n_maple),np.zeros(n_maple),np.zeros(n_maple)])
hickory_type = np.array([np.zeros(n_hickory),np.ones(n_hickory),np.zeros(n_hickory),np.zeros(n_hickory),np.zeros(n_hickory)])
blackoak_type = np.array([np.zeros(n_blackoak),np.zeros(n_blackoak),np.ones(n_blackoak),np.zeros(n_blackoak),np.zeros(n_blackoak)])
redoak_type = np.array([np.zeros(n_redoak),np.zeros(n_redoak),np.zeros(n_redoak),np.ones(n_redoak),np.zeros(n_redoak)])
whiteoak_type = np.array([np.zeros(n_whiteoak),np.zeros(n_whiteoak),np.zeros(n_whiteoak),np.zeros(n_whiteoak),np.ones(n_whiteoak)])

types = np.concatenate((maple_type,hickory_type,blackoak_type,redoak_type,whiteoak_type), axis=1)


### unif thinning

ntree = trees.shape[0]

kept = np.random.binomial(1,0.25,ntree)

kept = kept.astype(np.bool)

trees = trees[kept]
types = types[:,kept]

### neutral parameters (initial state)

p = 5


A = np.identity(p)*1
mu = np.ones(p)*-3.0
rhos = np.ones(p)*8.0
lam = 500

# ##### g func check
# N=10000

# steps = np.linspace(0.0, 1, num=50)
# gsR = gfuncest(N,A,mu,rhos,steps)

# plt.plot(steps,gsR[:,0])
# plt.plot(steps,gsR[:,5])
# plt.plot(steps,gsR[:,1],  c="grey")
# plt.show()


### priors

sigma_prior_A = 1 ### for A

### rhos
mean_prior = 8
sd_prior = 4
var_prior = sd_prior**2

alpha_prior = mean_prior**2 / var_prior
beta_prior = mean_prior/var_prior

### mu
sigma_prior_mu = 1
m_prior = np.array([-2])

### lambda
mean_prior_lam = 500
sd_prior_lam = 100
var_prior_lam = sd_prior_lam**2

alpha_prior_lam = mean_prior_lam**2 / var_prior_lam
beta_prior_lam = mean_prior_lam/var_prior_lam


### proposals

sigma_prop_A = 0.1
sigma_prop_rho = 0.1
sigma_mom_V = 0.5

delta = 0.005
L = 20

nbd = 10

### other initial param

thinLocs_init = np.zeros((0,2))
V_init = types -2

### other

diag = False
parr = False


size = 6000

import time

t0 = time.time()
A_mcmc, rho_mcmc, mu_mcmc, lam_mcmc = MCMC_LMC_MN_POIS(types, trees, sigma_prior_A, alpha_prior, beta_prior, alpha_prior_lam, beta_prior_lam, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, sigma_mom_V, delta, L, nbd, A, rhos, mu, V_init, thinLocs_init, lam, size, diag, parr)
t1 = time.time()


print((t1-t0)/60)


#### convergence plots

plt.plot(rho_mcmc[:,0])
plt.plot(rho_mcmc[:,1])
plt.plot(rho_mcmc[:,2])
plt.plot(rho_mcmc[:,3])
plt.plot(rho_mcmc[:,4])

plt.show()



plt.plot(mu_mcmc[:,0])
plt.plot(mu_mcmc[:,1])
plt.plot(mu_mcmc[:,2])
plt.plot(mu_mcmc[:,3])
plt.plot(mu_mcmc[:,4])

plt.show()


plt.plot(lam_mcmc)

plt.show()


#### gfunc

reso = 50    

tail = 3000
head = size - tail
    
gsR = np.zeros(shape=(tail,reso,15))   

N=1000
steps = np.linspace(0.0, 1.0, num=reso)

t0 = time.time()
for i in range(tail):    
    gsR[i] = gfuncest(N,np.linalg.inv(A_mcmc[head + i]),mu_mcmc[head + i],rho_mcmc[head + i],steps)
    print(i)
print(time.time()-t0)


gfunc_min = np.quantile(gsR, q=0.025, axis=0)
gfunc_max = np.quantile(gsR, q=0.975, axis=0)
gfunc_mean = np.mean(gsR, axis=0)


### plots pair corr functions

### maple & hickory

plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,0], label="maple", color="tab:blue")
plt.fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:blue", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,5], label="hickory", color="tab:orange")
plt.fill_between(steps, gfunc_min[:,5], gfunc_max[:,5], color="tab:orange", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,1],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,1], gfunc_max[:,1], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 

plt.show()


### maple & blackoak

plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,0], label="maple")
plt.fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:blue", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,9], label="blackoak", color="tab:green")
plt.fill_between(steps, gfunc_min[:,9], gfunc_max[:,9], color="tab:green", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,2],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,2], gfunc_max[:,2], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 

plt.show()



### maple & redoak

plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,0], label="maple")
plt.fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:blue", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,12], label="redoak", color="tab:red")
plt.fill_between(steps, gfunc_min[:,12], gfunc_max[:,12], color="tab:red", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,3],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,3], gfunc_max[:,3], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 

plt.show()


### maple & whiteoak

plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,0], label="maple")
plt.fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:blue", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,14], label="whiteoak", color="tab:purple")
plt.fill_between(steps, gfunc_min[:,14], gfunc_max[:,14], color="tab:purple", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,4],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,4], gfunc_max[:,4], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 

plt.show()

### hickory & blackoak

plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,5], label="hickory", color="tab:orange")
plt.fill_between(steps, gfunc_min[:,5], gfunc_max[:,5], color="tab:orange", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,9], label="blackoak", color="tab:green")
plt.fill_between(steps, gfunc_min[:,9], gfunc_max[:,9], color="tab:green", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,6],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,6], gfunc_max[:,6], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 

plt.show()



### hickory & redoak

plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,5], label="hickory", color="tab:orange")
plt.fill_between(steps, gfunc_min[:,5], gfunc_max[:,5], color="tab:orange", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,12], label="redoak", color="tab:red")
plt.fill_between(steps, gfunc_min[:,12], gfunc_max[:,12], color="tab:red", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,7],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,7], gfunc_max[:,7], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 

plt.show()


### hickory & whiteoak

plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,5], label="hickory", color="tab:orange")
plt.fill_between(steps, gfunc_min[:,5], gfunc_max[:,5], color="tab:orange", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,14], label="whiteoak", color="tab:purple")
plt.fill_between(steps, gfunc_min[:,14], gfunc_max[:,14], color="tab:purple", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,8],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,8], gfunc_max[:,8], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 

plt.show()


### blackoak & redoak

plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,9], label="blackoak", color="tab:green")
plt.fill_between(steps, gfunc_min[:,9], gfunc_max[:,9], color="tab:green", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,12], label="redoak", color="tab:red")
plt.fill_between(steps, gfunc_min[:,12], gfunc_max[:,12], color="tab:red", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,10],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,10], gfunc_max[:,10], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 

plt.show()


### blackoak & whiteoak

plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,9], label="blackoak", color="tab:green")
plt.fill_between(steps, gfunc_min[:,9], gfunc_max[:,9], color="tab:green", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,14], label="whiteoak", color="tab:purple")
plt.fill_between(steps, gfunc_min[:,14], gfunc_max[:,14], color="tab:purple", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,11],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,11], gfunc_max[:,11], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 

plt.show()


### redoak & whiteoak

plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,12], label="redoak", color="tab:red")
plt.fill_between(steps, gfunc_min[:,12], gfunc_max[:,12], color="tab:red", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,14], label="whiteoak", color="tab:purple")
plt.fill_between(steps, gfunc_min[:,14], gfunc_max[:,14], color="tab:purple", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,13],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,13], gfunc_max[:,13], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 

plt.show()











