# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:47:57 2022

@author: alier
"""

import numpy as np
from gfunc_est import gfuncest

import matplotlib.pyplot as plt

from MCMC_LMC import MCMC_LMC_MN_POIS

from GP import makeGrid
from GP import expCorr
from GP import rCondLMC
from GP import mexpit_col

p = 2
scale = 1/2

A = np.identity(p)*scale


mu = np.array([-2.0,-2.0])
rhos = np.array([5.0,5.0])


maple = np.loadtxt("maple.csv", delimiter=",")
hickory = np.loadtxt("hickory.csv", delimiter=",")
trees = np.concatenate((maple,hickory))

n_maple = maple.shape[0]
n_hickory = hickory.shape[0]

maple_type = np.array([np.ones(n_maple),np.zeros(n_maple)])
hickory_type = np.array([np.zeros(n_hickory),np.ones(n_hickory)])

types = np.concatenate((maple_type,hickory_type), axis=1)


# ##### g func
# N=10000

# steps = np.linspace(0.0, 0.8, num=50)
# gsR = gfuncest(N,A,mu,rhos,steps)

# plt.plot(steps,gsR[:,0])
# # plt.show()
# plt.plot(steps,gsR[:,2])
# # plt.show()
# plt.plot(steps,gsR[:,1],  c="grey")
# plt.show()




# A = np.array([[1,-0.5],[-0.5,1]])*scale


# gsR = gfuncest(N,A,mu,rhos,steps)

# plt.plot(steps,gsR[:,0])
# # plt.show()
# plt.plot(steps,gsR[:,2])
# # plt.show()
# plt.plot(steps,gsR[:,1],  c="grey")
# plt.show()


# A = np.array([[1,0.5],[0.5,1]])*scale

# gsR = gfuncest(N,A,mu,rhos,steps)

# plt.plot(steps,gsR[:,0])
# # plt.show()
# plt.plot(steps,gsR[:,2])
# # plt.show()
# plt.plot(steps,gsR[:,1],  c="grey")
# plt.show()
# #### g func



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

p = 2



thinLocs_init = np.random.uniform(size=(int((n_maple + n_hickory)/p),2))

diag = False
parr = True


A_init = np.linalg.inv(A)


rho_init = rhos


mu_init = mu


ntot = n_maple + n_hickory + thinLocs_init.shape[0]

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




# res = 40
# gridLoc = makeGrid([0,1], [0,1], res)




# resGP = np.empty(shape=(size,p,res**2))

# i=0
# while(i < size):
#     locs = np.loadtxt("locs"+str(i)+".csv", delimiter=",")
#     values = np.loadtxt("V"+str(i)+".csv", delimiter=",")
#     # Rinvs = np.array([np.loadtxt(str(j)+"R"+str(i)+".csv", delimiter=",") for j in range(p)])
    
#     #### scatter
    
#     # locs1 = locs[:nobs][types[0].astype(np.bool)]
#     # locs2 = locs[:nobs][types[1].astype(np.bool)]
    
#     # locsThin = locs[nobs:]


#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # ax.set_aspect('equal')

#     # plt.scatter(locs1[:,0], locs1[:,1])
#     # plt.scatter(locs2[:,0], locs2[:,1])
    
#     # plt.scatter(locsThin[:,0], locsThin[:,1])

#     # plt.xlim(0,1)
#     # plt.ylim(0,1)

#     # plt.show()
    
#     #### scatter
    
#     ntot = locs.shape[0]
#     corrFuncs = np.array([expCorr(rho) for rho in rho_mcmc[i]])
#     Rinvs = [np.linalg.inv(corrFuncs[j](locs,locs)) for j in range(p)]
    
#     # np.savetxt("resGP"+str(i)+".csv",lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))) ,delimiter=",")
#     newGP,u,v = rCondLMC(np.linalg.inv(A_mcmc[i]), corrFuncs, np.outer(mu_mcmc[i],np.ones(ntot)), np.outer(mu_mcmc[i],np.ones(res**2)), locs, gridLoc, Rinvs, values)
#     resGP[i] = lam_mcmc[i]*mexpit_col(newGP)
#     # meanGP = ((i+1)*meanGP + lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))))/(i+2)
    
#     # imGP = np.transpose(resGP[i].reshape(res,res))
    
#     # x = np.linspace(0,1, res+1) 
#     # y = np.linspace(0,1, res+1) 
#     # X, Y = np.meshgrid(x,y) 
    
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # ax.set_aspect('equal')
    
#     # plt.pcolormesh(X,Y,imGP, cmap='cool')
    
#     # plt.xlim(0,1)
#     # plt.ylim(0,1)
#     # plt.colorbar()
#     # plt.scatter(pointpo.loc[:,0],pointpo.loc[:,1], color= "black", s=1)
    
#     # plt.show()
    
#     print(i)
#     i+=1


# meanGP = np.mean(resGP, axis=0)
# nobs = n_maple + n_hickory


# locs1 = locs[:nobs][types[0].astype(np.bool)]

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')

# imGP = meanGP[0].reshape(res,res,order="F")

# x = np.linspace(0,1, res+1) 
# y = np.linspace(0,1, res+1) 
# X, Y = np.meshgrid(x,y) 
    

# ax.set_aspect('equal')
    
# ff = ax.pcolormesh(X,Y,imGP, cmap="Blues")

# fig.colorbar(ff) 

# plt.scatter(locs1[:,0], locs1[:,1], c="black", marker="$\clubsuit$", s=20)

# plt.show() 

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')

# locs2 = locs[:nobs][types[1].astype(np.bool)]

# imGP = meanGP[1].reshape(res,res,order="F")

# x = np.linspace(0,1, res+1) 
# y = np.linspace(0,1, res+1) 
# X, Y = np.meshgrid(x,y) 
    

# ax.set_aspect('equal')
    
# ff = ax.pcolormesh(X,Y,imGP, cmap="Oranges")

# plt.scatter(locs2[:,0], locs2[:,1], c="black", marker="$\clubsuit$", s=20)

# fig.colorbar(ff) 

# plt.show() 


# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_aspect('equal')

# imGP = (meanGP[0]+meanGP[1]).reshape(res,res,order="F")

# x = np.linspace(0,1, res+1) 
# y = np.linspace(0,1, res+1) 
# X, Y = np.meshgrid(x,y) 
    


# ax.set_aspect('equal')
    
# ff = ax.pcolormesh(X,Y,imGP, cmap="Greys")

# plt.scatter(locs1[:,0], locs1[:,1], c="black", marker="$\clubsuit$", s=20)
# plt.scatter(locs2[:,0], locs2[:,1], c="black", marker="$\clubsuit$", s=20)

# fig.colorbar(ff) 

# plt.show() 

# i=0

# while i<size-1:

#     locations = np.loadtxt("locs"+str(i)+".csv", delimiter=",")



    
#     locsThin = locations[nobs:]


#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal')


#     plt.scatter(locsThin[:,0], locsThin[:,1], c="grey", marker="$\clubsuit$", s=20)

#     plt.scatter(locs1[:,0], locs1[:,1],c="tab:blue", marker="$\clubsuit$", s=20)
#     plt.scatter(locs2[:,0], locs2[:,1],c="tab:orange", marker="$\clubsuit$", s=20)

    
    

#     plt.xlim(0,1)
#     plt.ylim(0,1)
    
#     # plt.savefig('dance'+str(j)+'.pdf')
#     plt.show()
#     # j+=1
#     i+=10
    
reso = 50    
    
gsR = np.zeros(shape=(1000,reso,3))   

N=3000  
steps = np.linspace(0.0, 1, num=reso)

t0 = time.time()
for i in range(1000):    
    gsR[i] = gfuncest(N,np.linalg.inv(A_mcmc[i+1000]),mu_mcmc[i+1000],rho_mcmc[i+1000],steps)
print(time.time()-t0)


gfunc_min = np.quantile(gsR, q=0.025, axis=0)
gfunc_max = np.quantile(gsR, q=0.975, axis=0)
gfunc_mean = np.mean(gsR, axis=0)



fig, axs = plt.subplots(2, 2, figsize=(9,7))

axs[0,0].title.set_text('Maple')

axs[0,0].plot(steps,gfunc_mean[:,0], label="maple")
axs[0,0].fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:blue", alpha=0.3, linewidth=0)

axs[0,1].title.set_text('Hickory')

axs[0,1].plot(steps,gfunc_mean[:,2], color="tab:orange", label="hickory")
axs[0,1].fill_between(steps, gfunc_min[:,2], gfunc_max[:,2], color="tab:orange", alpha=0.3, linewidth=0)

axs[1,0].title.set_text('Cross Pair Correlation')

axs[1,0].plot(steps,gfunc_mean[:,1],  c="grey", label="cross")
axs[1,0].fill_between(steps, gfunc_min[:,1], gfunc_max[:,1], color="grey", alpha=0.3, linewidth=0)

axs[1,1].plot(steps,gfunc_mean[:,0], label="maple")
axs[1,1].fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:blue", alpha=0.3, linewidth=0)
# plt.show()
axs[1,1].plot(steps,gfunc_mean[:,2], label="hickory")
axs[1,1].fill_between(steps, gfunc_min[:,2], gfunc_max[:,2], color="tab:orange", alpha=0.3, linewidth=0)
# plt.show()
axs[1,1].plot(steps,gfunc_mean[:,1],  c="grey", label="cross")
axs[1,1].fill_between(steps, gfunc_min[:,1], gfunc_max[:,1], color="grey", alpha=0.3, linewidth=0)
axs[1,1].legend(bbox_to_anchor=(1, 1)) 

# plt.savefig('pcf.pdf', bbox_inches='tight')
plt.show()



plt.title("Pair Correlation Functions")
plt.plot(steps,gfunc_mean[:,0], label="maple")
plt.fill_between(steps, gfunc_min[:,0], gfunc_max[:,0], color="tab:blue", alpha=0.3, linewidth=0)
# plt.show()
plt.plot(steps,gfunc_mean[:,2], label="hickory")
plt.fill_between(steps, gfunc_min[:,2], gfunc_max[:,2], color="tab:orange", alpha=0.3, linewidth=0)
# plt.show()
plt.plot(steps,gfunc_mean[:,1],  c="grey", label="cross")
plt.fill_between(steps, gfunc_min[:,1], gfunc_max[:,1], color="grey", alpha=0.3, linewidth=0)
plt.legend(bbox_to_anchor=(1, 1)) 
plt.savefig('pcf.pdf', bbox_inches='tight')
plt.show()




