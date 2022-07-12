# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 20:38:38 2022

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt

from gfunc_est import gfuncest
from gfunc_est import pairs

from MCMC_LMC import MCMC_LMC_MN_POIS

blackoak = np.loadtxt("blackoak.csv", delimiter=",")
redoak = np.loadtxt("redoak.csv", delimiter=",")
whiteoak = np.loadtxt("whiteoak.csv", delimiter=",")

n_blackoak = blackoak.shape[0]
n_redoak = redoak.shape[0]
n_whiteoak = whiteoak.shape[0]

nobs = n_blackoak+n_redoak+n_whiteoak

trees = np.concatenate((blackoak,redoak,whiteoak))

blackoak_type = np.array([np.ones(n_blackoak),np.zeros(n_blackoak),np.zeros(n_blackoak)])
redoak_type = np.array([np.zeros(n_redoak),np.ones(n_redoak),np.zeros(n_redoak)])
whiteoak_type = np.array([np.zeros(n_whiteoak),np.zeros(n_whiteoak),np.ones(n_whiteoak)])

types = np.concatenate((blackoak_type,redoak_type,whiteoak_type), axis=1)



p = 3
scale = 1/2

A = np.identity(p)*scale


mu = np.array([-2.0,-2.0,-2.0])
rhos = np.array([5.0,5.0,5.0])


#### check for neutral param


# N=10000

# steps = np.linspace(0.0, 1, num=50)
# gsR = gfuncest(N,A,mu,rhos,steps)

# plt.plot(steps,gsR[:,0], c="tab:green")
# # plt.show()
# plt.plot(steps,gsR[:,3], c="tab:red")
# # plt.show()
# plt.plot(steps,gsR[:,1],  c="grey")
# plt.show()


sigma_prior_A = 1 ### for A

mean_prior = 5
sd_prior = 4
var_prior = sd_prior**2

alpha_prior = mean_prior**2 / var_prior
beta_prior = mean_prior/var_prior

sigma_prior_mu = 1
m_prior = np.array([-2])




mean_prior_lam = 1500
sd_prior_lam = 200
var_prior_lam = sd_prior_lam**2

alpha_prior_lam = mean_prior_lam**2 / var_prior_lam
beta_prior_lam = mean_prior_lam/var_prior_lam

sigma_prop_A = 0.1
sigma_prop_rho = 0.1
sigma_mom_V = 0.5

delta = 0.005
L = 20

# nbd = lam/10
nbd = 20

lam_init = 1500





thinLocs_init = np.random.uniform(size=(int((n_blackoak+n_redoak+n_whiteoak)/p),2))

diag = False
parr = True


A_init = np.linalg.inv(A)


rho_init = rhos


mu_init = mu


ntot = n_blackoak+n_redoak+n_whiteoak + thinLocs_init.shape[0]

V_init = np.outer(mu_init,np.ones(ntot)) + 0.1*np.random.normal(size=(p,ntot))


size = 2000

import time

t0 = time.time()
A_mcmc, rho_mcmc, mu_mcmc, lam_mcmc = MCMC_LMC_MN_POIS(types, trees, sigma_prior_A, alpha_prior, beta_prior, alpha_prior_lam, beta_prior_lam, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, sigma_mom_V, delta, L, nbd, A_init, rho_init, mu_init, V_init, thinLocs_init, lam_init, size, diag, parr)
t1 = time.time()

total1 = t1-t0


print(np.mean(A_mcmc, axis=0))

plt.plot(A_mcmc[:,0,0])
plt.plot(A_mcmc[:,0,1])
plt.plot(A_mcmc[:,0,2])
plt.plot(A_mcmc[:,1,0])
plt.plot(A_mcmc[:,1,1])
plt.plot(A_mcmc[:,1,2])
plt.plot(A_mcmc[:,2,0])
plt.plot(A_mcmc[:,2,1])
plt.plot(A_mcmc[:,2,2])

plt.show()



print(np.mean(rho_mcmc, axis=0))

plt.plot(rho_mcmc[:,0])
plt.plot(rho_mcmc[:,1])
plt.plot(rho_mcmc[:,2])

plt.show()



print(np.mean(mu_mcmc, axis=0))

plt.plot(mu_mcmc[:,0])
plt.plot(mu_mcmc[:,1])
plt.plot(mu_mcmc[:,2])

plt.show()



print(np.mean(lam_mcmc))


plt.plot(lam_mcmc)

plt.show()



##### g function



reso = 50    
    
gsR = np.zeros(shape=(1000,reso,6))   

N=3000  
steps = np.linspace(0.0, 1.0, num=reso)

t0 = time.time()
for i in range(1000):    
    gsR[i] = gfuncest(N,np.linalg.inv(A_mcmc[i+1000]),mu_mcmc[i+1000],rho_mcmc[i+1000],steps)
print(time.time()-t0)


gfunc_min = np.quantile(gsR, q=0.025, axis=0)
gfunc_max = np.quantile(gsR, q=0.975, axis=0)
gfunc_mean = np.mean(gsR, axis=0)



plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,0], label="blackoak", color="tab:green")
plt.fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:green", alpha=0.3, linewidth=0)
# plt.show()
plt.plot(steps,gfunc_mean[:,3], label="redoak", color="tab:red")
plt.fill_between(steps, gfunc_min[:,3], gfunc_max[:,3], color="tab:red", alpha=0.3, linewidth=0)
# plt.show()
plt.plot(steps,gfunc_mean[:,1],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,1], gfunc_max[:,1], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 
plt.savefig('pcf1z.pdf', bbox_inches='tight')
plt.show()



plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,0], label="blackoak", color="tab:green")
plt.fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:green", alpha=0.3, linewidth=0)
# plt.show()
plt.plot(steps,gfunc_mean[:,5], label="whiteoak", color="tab:purple")
plt.fill_between(steps, gfunc_min[:,5], gfunc_max[:,5], color="tab:purple", alpha=0.3, linewidth=0)
# plt.show()
plt.plot(steps,gfunc_mean[:,2],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,2], gfunc_max[:,2], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 
plt.savefig('pcf2z.pdf', bbox_inches='tight')
plt.show()


plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,3], label="redoak", color="tab:red")
plt.fill_between(steps, gfunc_min[:,3], gfunc_max[:,3], color="tab:red", alpha=0.3, linewidth=0)
# plt.show()
plt.plot(steps,gfunc_mean[:,5], label="whiteoak", color="tab:purple")
plt.fill_between(steps, gfunc_min[:,5], gfunc_max[:,5], color="tab:purple", alpha=0.3, linewidth=0)
# plt.show()
plt.plot(steps,gfunc_mean[:,4],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,4], gfunc_max[:,4], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 
plt.savefig('pcf3z.pdf', bbox_inches='tight')
plt.show()


