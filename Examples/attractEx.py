# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:22:35 2022

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt
from GP import expCorr
from gfunc_est import gfuncest
from LogitGCP import rLGCP
from MCMC_LMC import MCMC_LMC_MN_POIS

lam=1000

p = 2
scale = 1/2

Atrue = np.array([[1,0.5],[0.5,1]])*scale
AinvTrue = np.linalg.inv(Atrue)

mu = np.array([-2.0,-2.0])
rhos = np.array([5.0,5.0])
corrFuncs = np.array([expCorr(rho) for rho in rhos])


Ainit = np.identity(p)*scale



#### check for neutral param


N=10000

steps = np.linspace(0.0, 1, num=50)
gsR = gfuncest(N,Atrue,mu,rhos,steps)

plt.plot(steps,gsR[:,0], c="tab:blue")
# plt.show()
plt.plot(steps,gsR[:,2], c="tab:orange")
# plt.show()
plt.plot(steps,gsR[:,1],  c="grey")
plt.show()


##### random sample

locs, types = rLGCP(lam, Atrue, corrFuncs, mu)

### plot
locs1 = locs[types[0].astype(np.bool)]
locs2 = locs[types[1].astype(np.bool)]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.scatter(locs1[:,0], locs1[:,1])
plt.scatter(locs2[:,0], locs2[:,1])

plt.xlim(0,1)
plt.ylim(0,1)

plt.show()

#####

sigma_prior_A = 2 ### for A

mean_prior = 5
sd_prior = 4
var_prior = sd_prior**2

alpha_prior = mean_prior**2 / var_prior
beta_prior = mean_prior/var_prior

sigma_prior_mu = 1
m_prior = np.array([-2])




mean_prior_lam = 1000
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

lam_init = lam

p = 2



thinLocs_init = np.random.uniform(size=(int((locs.shape[0])/p),2))

diag = False
parr = True


A_init = Ainit


rho_init = rhos


mu_init = mu


ntot = locs.shape[0] + thinLocs_init.shape[0]

V_init = np.outer(mu_init,np.ones(ntot)) + 0.1*np.random.normal(size=(p,ntot))


size = 2000

import time

t0 = time.time()
A_mcmc, rho_mcmc, mu_mcmc, lam_mcmc = MCMC_LMC_MN_POIS(types, locs, sigma_prior_A, alpha_prior, beta_prior, alpha_prior_lam, beta_prior_lam, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, sigma_mom_V, delta, L, nbd, A_init, rho_init, mu_init, V_init, thinLocs_init, lam_init, size, diag, parr)
t1 = time.time()

total1 = t1-t0


print(np.mean(A_mcmc, axis=0))

plt.plot(A_mcmc[:,0,0])
plt.plot(A_mcmc[:,0,1])
plt.plot(A_mcmc[:,1,0])
plt.plot(A_mcmc[:,1,1])

plt.show()



print(np.mean(rho_mcmc, axis=0))

plt.plot(rho_mcmc[:,0])
plt.plot(rho_mcmc[:,1])

plt.show()



print(np.mean(mu_mcmc, axis=0))

plt.plot(mu_mcmc[:,0])
plt.plot(mu_mcmc[:,1])

plt.show()



print(np.mean(lam_mcmc))


plt.plot(lam_mcmc)

plt.show()

##### pcf


reso = 50    
    
gsR = np.zeros(shape=(1000,reso,3))   

N=1000  
steps = np.linspace(0.0, 1, num=reso)

t0 = time.time()
for i in range(1000):    
    gsR[i] = gfuncest(N,np.linalg.inv(A_mcmc[i+1000]),mu_mcmc[i+1000],rho_mcmc[i+1000],steps)
print(time.time()-t0)


gfunc_min = np.quantile(gsR, q=0.025, axis=0)
gfunc_max = np.quantile(gsR, q=0.975, axis=0)
gfunc_mean = np.mean(gsR, axis=0)


plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,0])
plt.fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:blue", alpha=0.3, linewidth=0)
# plt.show()
plt.plot(steps,gfunc_mean[:,2])
plt.fill_between(steps, gfunc_min[:,2], gfunc_max[:,2], color="tab:orange", alpha=0.3, linewidth=0)
# plt.show()
plt.plot(steps,gfunc_mean[:,1],  c="grey")
plt.fill_between(steps, gfunc_min[:,1], gfunc_max[:,1], color="grey", alpha=0.3, linewidth=0)
# plt.legend(bbox_to_anchor=(1, 1)) 
plt.savefig('pcfsynth.pdf', bbox_inches='tight')
plt.show()




