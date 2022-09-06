# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:11:24 2022

@author: alier
"""




import numpy as np
import matplotlib.pyplot as plt

from MCMC_LMC import MCMC_LMC_MN_POIS
from gfunc_est import gfuncest
from GP import makeGrid
from GP import expCorr
from GP import rCondLMC
from GP import mexpit_col


### import 2 types of trees

maple = np.loadtxt("maple.csv", delimiter=",")
hickory = np.loadtxt("hickory.csv", delimiter=",")

trees = np.concatenate((maple,hickory))


### construct type matrix

n_maple = maple.shape[0]
n_hickory = hickory.shape[0]

maple_type = np.array([np.ones(n_maple),np.zeros(n_maple)])
hickory_type = np.array([np.zeros(n_hickory),np.ones(n_hickory)])

types = np.concatenate((maple_type,hickory_type), axis=1)

### neutral parameters (initial state)

p = 2


A = A = np.array([[1,0.35],[0.35,1]])
mu = np.ones(p)*-1.0
rhos = np.ones(p)*10
lam = 1500

# ##### g func check
# N=10000

# steps = np.linspace(0.0, 1, num=50)
# gsR = gfuncest(N,A,mu,rhos,steps)

# plt.plot(steps,gsR[:,0])
# plt.plot(steps,gsR[:,2])
# plt.plot(steps,gsR[:,1],  c="grey")
# plt.show()




### priors

sigma_prior_A = 1 ### for A

### rhos
mean_prior = 10
sd_prior = 5
var_prior = sd_prior**2

alpha_prior = mean_prior**2 / var_prior
beta_prior = mean_prior/var_prior

### mu
sigma_prior_mu = 1
m_prior = np.array([0])

### lambda
mean_prior_lam = 1500
sd_prior_lam = 100
var_prior_lam = sd_prior_lam**2

alpha_prior_lam = mean_prior_lam**2 / var_prior_lam
beta_prior_lam = mean_prior_lam/var_prior_lam



### proposals

sigma_prop_A = 0.05
sigma_prop_rho = 0.1
sigma_mom_V = 0.5

delta = 0.005
L = 20

nbd = 20

### other initial param

thinLocs_init = np.zeros((0,2))
V_init = types -1

### other

diag = True
parr = True


size = 10000

import time

t0 = time.time()
A_mcmc, rho_mcmc, mu_mcmc, lam_mcmc = MCMC_LMC_MN_POIS(types, trees, sigma_prior_A, alpha_prior, beta_prior, alpha_prior_lam, beta_prior_lam, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, sigma_mom_V, delta, L, nbd, A, rhos, mu, V_init, thinLocs_init, lam, size, diag, parr)
t1 = time.time()


tmcmc = (t1-t0)/60
print(tmcmc)

# np.save('A_mcmc', A_mcmc)
# np.save('rho_mcmc', rho_mcmc)
# np.save('mu_mcmc', mu_mcmc)
# np.save('lam_mcmc', lam_mcmc)

### trace plots



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



#### intensity plots

tail = 6000
head = size - tail

res = 52
gridLoc = makeGrid([0,1], [0,1], res)

gridGP = np.empty(shape=(size,p,res**2))

t0 = time.time()
for i in range(tail):
    locs = np.loadtxt("locs"+str(head+i)+".csv", delimiter=",")
    values = np.loadtxt("V"+str(head+i)+".csv", delimiter=",")
    
    ntot = locs.shape[0]
    
    corrFuncs = np.array([expCorr(rho) for rho in rho_mcmc[head+i]])
    Rinvs = [np.linalg.inv(corrFuncs[j](locs,locs)) for j in range(p)]
    
    # np.savetxt("resGP"+str(i)+".csv",lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))) ,delimiter=",")
    newGP,u,v = rCondLMC(np.linalg.inv(A_mcmc[head+i]), corrFuncs, np.outer(mu_mcmc[head+i],np.ones(ntot)), np.outer(mu_mcmc[head+i],np.ones(res**2)), locs, gridLoc, Rinvs, values)
    gridGP[i] = lam_mcmc[head+i]*mexpit_col(newGP)
    
    print(i)
t1 = time.time()


tintensity = (t1-t0)/60
print(tintensity)

# np.save('gridGP', gridGP)


meanGP = np.mean(gridGP, axis=0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# ax1 = fig.add_subplot(111)
ax1.set_aspect('equal')

imGP = meanGP[0].reshape(res,res,order="F")

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax1.set_aspect('equal')
ax1.set_title("Maple")    
ff = ax1.pcolormesh(X,Y,imGP,cmap="Blues")

fig.colorbar(ff,ax=ax1, shrink=0.75) 


ax1.scatter(maple[:,0], maple[:,1], c="black", marker="$\clubsuit$", s=20)


# plt.savefig('mapleInt.png', bbox_inches='tight', dpi=1200)
# plt.show() 



# fig = plt.figure()
# ax2 = fig.add_subplot(111)
ax2.set_aspect('equal')

imGP = meanGP[1].reshape(res,res,order="F")

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax2.set_aspect('equal')
ax2.set_title("Hickory")    
ff = ax2.pcolormesh(X,Y,imGP,cmap="Oranges")

fig.colorbar(ff, ax= ax2, shrink=0.75) 


ax2.scatter(hickory[:,0], hickory[:,1], c="black", marker="$\clubsuit$", s=20)


plt.savefig('hickoryMapleInt.png', bbox_inches='tight', dpi=1200)
plt.show() 


#### pair correlation functions



reso = 52   

    
gsR = np.zeros(shape=(tail,reso,3))   

N=5000
steps = np.linspace(0.0, 0.8, num=reso)

t0 = time.time()
for i in range(tail):    
    gsR[i] = gfuncest(N,np.linalg.inv(A_mcmc[head + i]),mu_mcmc[head + i],rho_mcmc[head + i],steps)
    print(i)
t1 = time.time()

tpcf = (t1-t0)/60
print(tpcf)

# np.save('gsR', gsR)

gfunc_min = np.quantile(gsR, q=0.025, axis=0)
gfunc_max = np.quantile(gsR, q=0.975, axis=0)
gfunc_mean = np.mean(gsR, axis=0)


np.save("gsR5x.npy", gsR)

### plots pair corr functions

### maple & hickory

plt.title("Pair Correlation Function")
plt.plot(steps,gfunc_mean[:,0], label="maple", color="tab:blue")
plt.fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:blue", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,2], label="hickory", color="tab:orange")
plt.fill_between(steps, gfunc_min[:,2], gfunc_max[:,2], color="tab:orange", alpha=0.3, linewidth=0)

plt.plot(steps,gfunc_mean[:,1],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,1], gfunc_max[:,1], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 


plt.savefig('pcfhickmap.pdf', bbox_inches='tight')
plt.show()




#### number of thinned locations

ntree = n_maple + n_hickory

nthin = np.zeros(tail)

for i in range(tail):
    locs = np.loadtxt("locs"+str(head+i)+".csv", delimiter=",")
    
    nthin[i] = locs.shape[0] - ntree
    
    
    print(i)


fig = plt.figure(figsize =(5*1.5, 1*1.5))
ax = fig.add_subplot(111)




bp = ax.boxplot(nthin, vert = 0, patch_artist=True,widths=[0.8])

bp["medians"][0].set(color="black")
bp["boxes"][0].set(color="grey")
bp["boxes"][0].set_facecolor(color="silver")

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft=False)

plt.xlabel("Number of Thinned Locations")


plt.savefig('bpnbthinned.pdf', bbox_inches='tight')
plt.show()



#### hickory and mape plot





randInd = np.arange(ntree)
np.random.shuffle(randInd)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.scatter(maple[:,0], maple[:,1], c="tab:blue", label="maple", marker="$\clubsuit$", s=20)
plt.scatter(hickory[:,0], hickory[:,1], c="tab:orange", label="hickory", marker="$\clubsuit$", s=20)
# plt.scatter(blackoak[:,0], blackoak[:,1], c="tab:green", label="blackoak", marker="$\clubsuit$", s=20)
# plt.scatter(redoak[:,0], redoak[:,1], c="tab:red", label="redoak", marker="$\clubsuit$", s=20)
# plt.scatter(whiteoak[:,0], whiteoak[:,1], c="tab:purple", label="whiteoak", marker="$\clubsuit$", s=20)


 
for i in range(ntree):
    if types[0,randInd[i]] == 1:
        plt.scatter(trees[randInd[i],0], trees[randInd[i],1], c="tab:blue", marker="$\clubsuit$", s=20)
    elif types[1,randInd[i]] == 1:
        plt.scatter(trees[randInd[i],0], trees[randInd[i],1], c="tab:orange", marker="$\clubsuit$", s=20)
    # elif treeTypes[2,randInd[i]] == 1:
    #     plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:green", marker="$\clubsuit$", s=20)
    # elif treeTypes[3,randInd[i]] == 1:
    #     plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:red", marker="$\clubsuit$", s=20)
    # elif treeTypes[4,randInd[i]] == 1:
    #     plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:purple", marker="$\clubsuit$", s=20)

ax.legend(bbox_to_anchor=(1, 0.8))

plt.xlim(0,1)
plt.ylim(0,1)
plt.title("Lansing Woods")

plt.show()
fig.savefig("allTrees.pdf", bbox_inches='tight')


