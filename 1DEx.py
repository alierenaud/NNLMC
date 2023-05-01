# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:22:24 2023

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt

from MCMC_LMC import MCMC_LMC_MN_POIS

from GP import makeGrid
from GP import expCorr
from GP import rCondLMC
from GP import mexpit_col



np.random.seed(3)

lam = 200

n = np.random.poisson(lam)

locs_init = np.random.uniform(size=(n,1))



def fct(x):
    return((np.sin(x*6*np.pi -np.pi/2)*0.25 + 0.5))



res = 1000
ds = np.linspace(0,1,res)





obs = np.full(n,False)


for i in range(n):
    
    if np.random.uniform() < fct(np.array([locs_init[i]])):
    
        obs[i] = True
        
        
        
      

plt.plot(ds,fct(ds))
plt.scatter(x=locs_init[obs], y=np.zeros(np.sum(obs)), marker="|", s=100, c="black")



plt.xlim(0,1)
plt.ylim(0,1)

plt.show()




#### run mcmc


maple = locs_init[obs]

n_maple = maple.shape[0]

maple_type = np.array([np.ones(n_maple)])

######

sigma_prior_A = 1 ### for A

mean_prior = 5
sd_prior = 4
var_prior = sd_prior**2

alpha_prior = mean_prior**2 / var_prior
beta_prior = mean_prior/var_prior

sigma_prior_mu = 10
m_prior = np.array([0])




mean_prior_lam = 200
sd_prior_lam = mean_prior_lam*0.9
var_prior_lam = sd_prior_lam**2

alpha_prior_lam = mean_prior_lam**2 / var_prior_lam
beta_prior_lam = mean_prior_lam/var_prior_lam

sigma_prop_A = 0.2
sigma_prop_rho = 0.5
sigma_mom_V = 0.5

delta = 0.005
L = 20

# nbd = lam/10
nbd = 20

lam_init = 200

p = 1



thinLocs_init = np.random.uniform(size=(lam_init-n_maple,1))

diag = True
parr = False


A_init = np.array([[1.0]])


rho_init = np.array([5.0])


mu_init = np.array([0.0])


ntot = n_maple + thinLocs_init.shape[0]

V_init = np.outer(mu_init,np.ones(ntot)) + 0.1*np.random.normal(size=(p,ntot))


size = 10000

import time

t0 = time.time()
A_mcmc, rho_mcmc, mu_mcmc, lam_mcmc = MCMC_LMC_MN_POIS(maple_type, maple, sigma_prior_A, alpha_prior, beta_prior, alpha_prior_lam, beta_prior_lam, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, sigma_mom_V, delta, L, nbd, A_init, rho_init, mu_init, V_init, thinLocs_init, lam_init, size, diag, parr)
t1 = time.time()

total1 = t1-t0

np.save("A_mcmc",A_mcmc)
np.save("rho_mcmc",rho_mcmc)
np.save("mu_mcmc",mu_mcmc)
np.save("lam_mcmc",lam_mcmc)


A_mcmc = np.load("A_mcmc.npy")
rho_mcmc = np.load("rho_mcmc.npy")
mu_mcmc = np.load("mu_mcmc.npy")
lam_mcmc = np.load("lam_mcmc.npy")

print(np.mean(A_mcmc, axis=0))

plt.plot(A_mcmc[:,0,0])


plt.show()



print(np.mean(rho_mcmc, axis=0))

plt.plot(rho_mcmc[:,0])


plt.show()



print(np.mean(mu_mcmc, axis=0))

plt.plot(mu_mcmc[:,0])


plt.show()



print(np.mean(lam_mcmc))


plt.plot(lam_mcmc)

plt.show()



res = 1000
gridLoc = np.transpose([np.linspace(0,1,res)])

tail = 0


resGP = np.empty(shape=(size-tail,p,res))

i=tail
while(i < size):
    locs = np.load("locs"+str(i)+".npy")
    values = np.load("V"+str(i)+".npy")

    # Rinvs = np.array([np.loadtxt(str(j)+"R"+str(i)+".csv", delimiter=",") for j in range(p)])
    
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
    Rinvs = [np.linalg.inv(corrFuncs[j](locs,locs)) for j in range(p)]
    
    # np.savetxt("resGP"+str(i)+".csv",lams[i+1]*expit(newGP.rCondGP(gridLoc,locations,np.transpose([values]))) ,delimiter=",")
    newGP,u,v = rCondLMC(np.linalg.inv(A_mcmc[i]), corrFuncs, np.outer(mu_mcmc[i],np.ones(ntot)), np.outer(mu_mcmc[i],np.ones(res)), locs, gridLoc, Rinvs, values)
    resGP[i-tail] = lam_mcmc[i]*mexpit_col(newGP)
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


np.save("resGP",resGP)

meanGP = np.mean(resGP, axis=0)
q5GP = np.quantile(resGP, 0.05, axis=0)
q95GP = np.quantile(resGP, 0.95, axis=0)

plt.plot(ds,lam*fct(ds),color = "black")
plt.plot(ds,meanGP[0])
plt.fill_between(ds, q5GP[0], q95GP[0] , color="tab:blue", alpha=0.5) 
plt.scatter(x=locs_init[obs], y=np.zeros(np.sum(obs)), marker="|", s=100, c="black")
# plt.scatter(x=locs[obs], y=np.zeros(np.sum(obs)), marker="|", s=100, c="black")
plt.savefig('1DEx.pdf', bbox_inches='tight')




plt.show()






# i=1
# j=1

# while i<size:

#     locations = np.load("locs"+str(i)+".npy")



    
#     locsThin = locations[n_maple:]
#     locs1 = locations[:n_maple]

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_aspect('equal')

#     plt.title("Lansing Woods")
#     plt.scatter(locsThin[:,0], locsThin[:,1], c="silver", s=20)

#     # plt.scatter(locs1[:,0], locs1[:,1],c="tab:blue", marker="$\clubsuit$", s=20, label="maple")

#     # #get handles and labels
#     # handles, labels = plt.gca().get_legend_handles_labels()

#     # #specify order of items in legend
#     # order = [1,0]

#     # #add legend to plot
#     # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1, 0.8)) 
    
    
    

#     plt.xlim(0,1)
#     plt.ylim(0,1)
    
#     plt.savefig('dancea'+str(j)+'.pdf', bbox_inches='tight')
#     plt.show()
#     j+=1
#     i+=400















