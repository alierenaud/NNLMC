# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:36:29 2022

@author: alier
"""

from LogitGCP import rLGCP
import numpy as np
from GP import expCorr
import matplotlib.pyplot as plt

from MCMC_LMC import MCMC_LMC_MN_POIS

from numpy import random

from GP import makeGrid
from GP import rCondLMC
from GP import mexpit_col



lam = 2000
A = np.array([[4,3],[-5,0]])
rhos = np.array([2,6])
corrFuncs = np.array([expCorr(rho) for rho in rhos])

mu = np.array([0,0])

locs, types = rLGCP(lam, A, corrFuncs, mu)

nobs = locs.shape[0]

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

### end plot

sigma_prior_A = 1 ### for A

mean_prior = 6
sd_prior = 4
var_prior = sd_prior**2

sigma_prior_mu = 10
m_prior = np.array([0,0])

alpha_prior = mean_prior**2 / var_prior
beta_prior = mean_prior/var_prior


mean_prior_lam = lam
sd_prior_lam = lam*0.9
# mean_prior_lam = lam
# sd_prior_lam = lam*0.9
var_prior_lam = sd_prior_lam**2

alpha_prior_lam = mean_prior_lam**2 / var_prior_lam
beta_prior_lam = mean_prior_lam/var_prior_lam

sigma_prop_A = 0.02
sigma_prop_rho = 0.1
sigma_mom_V = 0.5

delta = 0.005
L = 20

# nbd = lam/10
nbd = 20

lam_init = lam

p = types.shape[0]
n = locs.shape[0]


thinLocs_init = random.uniform(size=(int(n/p),2))

diag = False
parr = True

# A_init = np.identity(2)
A_init = np.linalg.inv(A)

# rho_init = np.array([10,10])
rho_init = rhos

# mu_init = np.array([0,0])
mu_init = mu


ntot = n + thinLocs_init.shape[0]
# V_init = newLMC.rLMC(gridLoc)
V_init = np.outer(mu_init,np.ones(ntot)) + 0.1*random.normal(size=(p,ntot))
# V_init = resLMC

size = 100

import time

t0 = time.time()
A_mcmc, rho_mcmc, mu_mcmc, lam_mcmc = MCMC_LMC_MN_POIS(types, locs, sigma_prior_A, alpha_prior, beta_prior, alpha_prior_lam, beta_prior_lam, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, sigma_mom_V, delta, L, nbd, A_init, rho_init, mu_init, V_init, thinLocs_init, lam_init, size, diag, parr)
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

print(lam)

print(np.mean(lam_mcmc))


plt.plot(lam_mcmc)

plt.show()



res = 25
gridLoc = makeGrid([0,1], [0,1], res)




resGP = np.empty(shape=(size,p,res**2))

i=0
while(i < size):
    locs = np.loadtxt("locs"+str(i)+".csv", delimiter=",")
    values = np.loadtxt("V"+str(i)+".csv", delimiter=",")
    Rinvs = np.array([np.loadtxt(str(j)+"R"+str(i)+".csv", delimiter=",") for j in range(p)])
    
    #### scatter
    
    # locs1 = locs[:nobs][types[0].astype(np.bool)]
    # locs2 = locs[:nobs][types[1].astype(np.bool)]
    
    # locsThin = locs[nobs:]


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal')

    # plt.scatter(locs1[:,0], locs1[:,1])
    # plt.scatter(locs2[:,0], locs2[:,1])
    
    # plt.scatter(locsThin[:,0], locsThin[:,1])

    # plt.xlim(0,1)
    # plt.ylim(0,1)

    # plt.show()
    
    #### scatter
    
    ntot = locs.shape[0]
    corrFuncs = np.array([expCorr(rho) for rho in rho_mcmc[i]])
    
    # np.savetxt("resGP"+str(i)+".csv",lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))) ,delimiter=",")
    newGP,u,v = rCondLMC(np.linalg.inv(A_mcmc[i]), corrFuncs, np.outer(mu_mcmc[i],np.ones(ntot)), np.outer(mu_mcmc[i],np.ones(res**2)), locs, gridLoc, Rinvs, values)
    resGP[i] = lam_mcmc[i]*mexpit_col(newGP)
    # meanGP = ((i+1)*meanGP + lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))))/(i+2)
    
    # imGP = np.transpose(resGP[i].reshape(res,res))
    
    # x = np.linspace(0,1, res+1) 
    # y = np.linspace(0,1, res+1) 
    # X, Y = np.meshgrid(x,y) 
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal')
    
    # plt.pcolormesh(X,Y,imGP, cmap='cool')
    
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    # plt.colorbar()
    # plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1], color= "black", s=1)
    
    # plt.show()
    
    print(i)
    i+=1


meanGP = np.mean(resGP, axis=0)


locs1 = locs[:nobs][types[0].astype(np.bool)]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

imGP = meanGP[0].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax.set_aspect('equal')
    
ff = ax.pcolormesh(X,Y,imGP)

fig.colorbar(ff) 

plt.scatter(locs1[:,1], locs1[:,0], c="black")

plt.show() 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

locs2 = locs[:nobs][types[1].astype(np.bool)]

imGP = meanGP[1].reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax.set_aspect('equal')
    
ff = ax.pcolormesh(X,Y,imGP)

plt.scatter(locs2[:,1], locs2[:,0], c="black")

fig.colorbar(ff) 

plt.show() 


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

imGP = (meanGP[0]+meanGP[1]).reshape(res,res)

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    


ax.set_aspect('equal')
    
ff = ax.pcolormesh(X,Y,imGP)

plt.scatter(locs1[:,1], locs1[:,0], c="black")
plt.scatter(locs2[:,1], locs2[:,0], c="black")

fig.colorbar(ff) 

plt.show() 



### points dancing arround

i=0

while i<size-1:

    locations = np.loadtxt("locs"+str(i)+".csv", delimiter=",")

    locs1 = locations[:nobs][types[0].astype(np.bool)]
    locs2 = locations[:nobs][types[1].astype(np.bool)]
    
    locsThin = locations[nobs:]


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    plt.scatter(locs1[:,0], locs1[:,1])
    plt.scatter(locs2[:,0], locs2[:,1])
    
    plt.scatter(locsThin[:,0], locsThin[:,1])

    plt.xlim(0,1)
    plt.ylim(0,1)
    
    # plt.savefig('dance'+str(i)+'.pdf')
    plt.show()
    
    i+=1



## GPs

values = np.empty(shape=(size,p,nobs))

i=0
while(i < size):

    values[i] = np.loadtxt("V"+str(i)+".csv", delimiter=",")[:,:nobs]

    
    
    print(i)
    i+=1

for i in range(nobs):

    if types[0,i] == 1:
        plt.plot(values[:,0,i], color="tab:orange")
    else:    
        plt.plot(values[:,0,i], color="tab:blue")
    
    if types[1,i] == 1:
        plt.plot(values[:,1,i], color="tab:orange")
    else:    
        plt.plot(values[:,1,i], color="tab:blue")
        
    # plt.savefig('gp'+str(i)+'.pdf')    
    plt.show()




