# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:14:30 2022

@author: alier
"""

import numpy as np
from numpy import random
from scipy.spatial import distance_matrix
from scipy.stats import gamma
from GP import mexpit_col
from GP import rCondLMC
from GP import expCorr

# import ray

def covMatrix(x,y,rho):

        
    return(np.exp(-distance_matrix(x,y)*rho))


def A_move(sigma_prior, A_j, Ainv_j, n, p, Y, Rinv_j, sigma_prop):
    
    A_j_prime = A_j + sigma_prop * random.normal(size=p)
    
    AjpmAj = A_j_prime - A_j
    
    if random.uniform() < np.abs(1+AjpmAj@Ainv_j)**n * np.exp(-1/2*AjpmAj@(Y@Rinv_j@np.transpose(Y)+1/sigma_prior**2*np.identity(p))@np.transpose(A_j_prime + A_j)):
        return(A_j_prime)
    else:
        return(A_j)

   
def rho_move(A_j, Y, R_j_inv, rho_j, alpha_prior, beta_prior, sigma_prop, locs):
    
    rho_j_prime = np.abs(rho_j + sigma_prop * random.normal())
    
    R_j_prime = covMatrix(locs,locs,rho_j_prime)
    R_j_inv_prime = np.linalg.inv(R_j_prime)
    
    # detR_j_inv_prime = np.linalg.det(R_j_inv_prime)
    
    AjY = A_j@Y
    
    if random.uniform() < 1/np.sqrt(np.linalg.det(R_j_inv@R_j_prime)) * np.exp(-1/2*AjY@(R_j_inv_prime-R_j_inv)@np.transpose(AjY)) * (rho_j_prime/rho_j)**(alpha_prior-1) * np.exp(-beta_prior*(rho_j_prime-rho_j)):
        return(rho_j_prime, R_j_inv_prime)
    else:
        return(rho_j, R_j_inv)

# @ray.remote 
# def rho_mover(A_j, Y, R_j_inv, rho_j, alpha_prior, beta_prior, sigma_prop, locs):
    
#     rho_j_prime = np.abs(rho_j + sigma_prop * random.normal())
    
#     R_j_prime = covMatrix(locs,locs,rho_j_prime)
#     R_j_inv_prime = np.linalg.inv(R_j_prime)
    
#     # detR_j_inv_prime = np.linalg.det(R_j_inv_prime)
    
#     AjY = A_j@Y
    
#     if random.uniform() < 1/np.sqrt(np.linalg.det(R_j_inv@R_j_prime)) * np.exp(-1/2*AjY@(R_j_inv_prime-R_j_inv)@np.transpose(AjY)) * (rho_j_prime/rho_j)**(alpha_prior-1) * np.exp(-beta_prior*(rho_j_prime-rho_j)):
#         return(rho_j_prime, R_j_inv_prime)
#     else:
#         return(rho_j, R_j_inv)

def mu_move(A,n,p,Rs,m,sigma,Y):
    
    ind_n = np.ones(shape=n)
    
    Us = np.zeros(shape=(p,p,p))
    vs = np.zeros(shape=(p,p))
    
    for j in range(p):
        AjTAj = np.outer(A[j],A[j])
        Rjm1indn = Rs[j] @ ind_n
        
        Us[j] = np.inner(ind_n,Rjm1indn) * AjTAj
        vs[j] = AjTAj @ Y @ Rjm1indn
        
    M = np.sum(Us,axis=0) + 1/sigma**2*np.identity(p)
    Minv = np.linalg.inv(M)
    b = np.sum(vs,axis=0) + 1/sigma**2*m
    
    
    
    
    
    return(np.linalg.cholesky(Minv)@random.normal(size=p) + Minv@b)



def MCMC_LMC(thisLMC, locs, sigma_prior_A, alpha_prior, beta_prior, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, A_init, rho_init, mu_init, size):
    
    
    
    p = thisLMC.shape[0] # nb of lines/types 
    n = thisLMC.shape[1] # nb of columns/locations

    ## chain containers
    
    A_mcmc = np.zeros(shape=(size,p,p))
    rho_mcmc = np.zeros(shape=(size,p))
    mu_mcmc = np.zeros(shape=(size,p))
    
    ## initial state
    
    A_mcmc[0] = A_init 
    rho_mcmc[0] = rho_init
    mu_mcmc[0] = mu_init
    
    ## current state
    
    Rinv_current = np.array([np.linalg.inv(covMatrix(locs,locs,rho)) for rho in rho_init])
    # detRinv_current = np.array([np.linalg.det(R) for R in Rinv_current])
    
    A_current = A_init
    
    ind_n = np.ones(n)
    centeredLMC_current = thisLMC - np.outer(mu_init,ind_n)
    
    state = 1
    
    while state<size:
        
        for j in range(p):
            
            A_current[j] = A_move(sigma_prior_A, A_current[j], np.linalg.inv(A_current)[:,j], n, p, centeredLMC_current, Rinv_current[j], sigma_prop_A)
            
        
        A_mcmc[state] = A_current
        
        for j in range(p):
            
            rho_mcmc[state,j], Rinv_current[j] = rho_move(A_current[j], centeredLMC_current, Rinv_current[j], rho_mcmc[state-1,j], alpha_prior, beta_prior, sigma_prop_rho, locs)
        
            
        mu_mcmc[state] = mu_move(A_current, n, p, Rinv_current, m_prior, sigma_prior_mu, thisLMC)
        
        centeredLMC_current = thisLMC - np.outer(mu_mcmc[state],ind_n)
        
        state+=1
        print(state)

    
    return(A_mcmc, rho_mcmc, mu_mcmc)




def U_MN(V,Y):
    
    return(np.sum(np.log(1+np.sum(np.exp(V), axis=0)) - np.sum(V*Y, axis=0)))

def U_MN_prime(V,Y):
    
    return(mexpit_col(V)-Y)



def U_LMC(V,A,Rinvs,mu,p,n):
    
    ind_n = np.ones(n)
    

    Vmmu = V-np.outer(mu,ind_n)
    
    AVmmu = [a@Vmmu for a in A]

    return(1/2*np.sum([AVmmu[j]@Rinvs[j]@AVmmu[j] for j in range(p)]))

def U_LMC_prime(V,A,Rinvs,mu,p,n):
    
    ind_n = np.ones(n)
    

    Vmmu = V-np.outer(mu,ind_n)
    
    return(np.sum([np.outer(A[j],A[j])@Vmmu@Rinvs[j] for j in range(p)],axis=0))

def V_move(sigma_prop_V,p,n,delta,L,x,Y,Rinvs,A,mu):
    
    v = sigma_prop_V*random.normal(size=(p,n))
    
    xtemp=x
    vtemp=v

    
    for l in range(L):
        vstar = vtemp - delta/2*(U_MN_prime(xtemp,Y)+U_LMC_prime(xtemp,A,Rinvs,mu,p,n))
        xtemp = xtemp + delta*vstar/sigma_prop_V**2
        vtemp = vstar - delta/2*(U_MN_prime(xtemp,Y)+U_LMC_prime(xtemp,A,Rinvs,mu,p,n))



        
    # vnew=-vnew
    
    lpi_old=-(U_MN(x,Y) + U_LMC(x,A,Rinvs,mu,p,n)) - 1/2*np.sum(v**2)/sigma_prop_V**2
    lpi_new=-(U_MN(xtemp,Y) + U_LMC(xtemp,A,Rinvs,mu,p,n)) - 1/2*np.sum(vtemp**2)/sigma_prop_V**2
    
    if np.log(random.uniform()) < lpi_new - lpi_old:
        return(xtemp)
    else:
        return(x)
    


def MCMC_LMC_MN(thisLMC_MN, locs, sigma_prior_A, alpha_prior, beta_prior, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, sigma_mom_V, delta, L, A_init, rho_init, mu_init, V_init, size, diag_V):
    
    
    
    p = thisLMC_MN.shape[0] # nb of lines/types 
    n = thisLMC_MN.shape[1] # nb of columns/locations

    ## chain containers
    
    A_mcmc = np.zeros(shape=(size,p,p))
    rho_mcmc = np.zeros(shape=(size,p))
    mu_mcmc = np.zeros(shape=(size,p))
    
    if diag_V:
        V_mcmc = np.zeros(shape=(size,p,n))
    
    ## initial state
    
    A_mcmc[0] = A_init 
    rho_mcmc[0] = rho_init
    mu_mcmc[0] = mu_init
    
    if diag_V:
        V_mcmc[0] = V_init
    
    ## current state
    
    Rinv_current = np.array([np.linalg.inv(covMatrix(locs,locs,rho)) for rho in rho_init])
    # detRinv_current = np.array([np.linalg.det(R) for R in Rinv_current])
    
    A_current = A_init
    
    V_current = V_init
    
    ind_n = np.ones(n)
    centeredV_current = V_current - np.outer(mu_init,ind_n)
    
    state = 1
    
    while state<size:
        
        V_current = V_move(sigma_mom_V,p,n,delta,L,V_current,thisLMC_MN,Rinv_current,A_current,mu_mcmc[state-1]) 
        
        if diag_V:
            V_mcmc[state] = V_current
        
        centeredV_current = V_current - np.outer(mu_mcmc[state-1],ind_n)
        
        for j in range(p):
            
            A_current[j] = A_move(sigma_prior_A, A_current[j], np.linalg.inv(A_current)[:,j], n, p, centeredV_current, Rinv_current[j], sigma_prop_A)
            
        
        A_mcmc[state] = A_current
        
        for j in range(p):
            
            rho_mcmc[state,j], Rinv_current[j] = rho_move(A_current[j], centeredV_current, Rinv_current[j], rho_mcmc[state-1,j], alpha_prior, beta_prior, sigma_prop_rho, locs)
        
            
        mu_mcmc[state] = mu_move(A_current, n, p, Rinv_current, m_prior, sigma_prior_mu, V_current)
        

        
        state+=1
        # print(state)

    if diag_V:
        return(A_mcmc, rho_mcmc, mu_mcmc, V_mcmc)
    else: 
        return(A_mcmc, rho_mcmc, mu_mcmc, 0)


def b(n):
    if n == 0:
        return(1)
    else:
        return(0.5)

    
def birthUpdate(Rinv,r,Rprime):
    
    n = Rinv.shape[0]
    
    U = np.zeros(shape = (n+1,n+1))
    
    Rm1r = Rinv @ r
    u = 1/(Rprime - Rm1r@r)
    uRM1r = -u*Rm1r
    
    U[n,n] = u
    U[:-1,n] = uRM1r
    U[n,:-1] = uRM1r
    U[:-1,:-1] = Rinv - np.outer(Rm1r,uRM1r)
    
    return(U)

# @ray.remote 
# def birthUpdater(Rinv,r,Rprime):
    
#     n = Rinv.shape[0]
    
#     U = np.zeros(shape = (n+1,n+1))
    
#     Rm1r = Rinv @ r
#     u = 1/(Rprime - Rm1r@r)
#     uRM1r = -u*Rm1r
    
#     U[n,n] = u
#     U[:-1,n] = uRM1r
#     U[n,:-1] = uRM1r
#     U[:-1,:-1] = Rinv - np.outer(Rm1r,uRM1r)
    
#     return(U)

def deathUpdate(Rninv,r,i):

    n = Rninv.shape[0]
    
    ind = [x for x in range(n) if x!=i]

    U = np.zeros(shape=(n,2))
    U[ind,1] = r
    U[i,0] = 1
    V = np.transpose(U)[::-1]
    
    
    Rnm1U = Rninv@U
    return((Rninv + Rnm1U@np.linalg.inv(np.identity(2) - V@Rnm1U)@V@Rninv)[ind][:,ind])

# @ray.remote 
# def deathUpdater(Rninv,r,i):

#     n = Rninv.shape[0]
    
#     ind = [x for x in range(n) if x!=i]

#     U = np.zeros(shape=(n,2))
#     U[ind,1] = r
#     U[i,0] = 1
#     V = np.transpose(U)[::-1]
    
    
#     Rnm1U = Rninv@U
#     return((Rninv + Rnm1U@np.linalg.inv(np.identity(2) - V@Rnm1U)@V@Rninv)[ind][:,ind])


def birthLoc(A, rhos, ntot, nthin, mu, locs_current, Rinvs, V_list, lam, index, nlocs, parr):
    
    p = A.shape[0]
    
    x_new = random.uniform(size=2)
    
    ind_ntot = np.ones(ntot)
    
    corrFuncs = np.array([expCorr(rho) for rho in rhos])
    
    V_new, rs, Rprimes = rCondLMC(np.linalg.inv(A), corrFuncs, np.outer(mu,ind_ntot), [[m] for m in mu], locs_current[index], np.array([x_new]), Rinvs, V_list[:,index])
    
    if random.uniform() < lam/(nthin+1)*(1-b(nthin+1))/b(nthin)/(1+np.sum(np.exp(V_new))):
        
        # print("birth")
        
        locs_current[nlocs] = x_new
        V_list[:,nlocs] = V_new[:,0]
        
        index.append(nlocs)
        nlocs += 1
        
        # if parr:
        #     Rinvs = np.array(ray.get([ birthUpdater.remote(Rinvs[j],rs[j,:,0],Rprimes[j]) for j in range(p)]))
        # else:
        Rinvs = np.array([ birthUpdate(Rinvs[j],rs[j,:,0],Rprimes[j]) for j in range(p)])
        nthin += 1 
        ntot += 1
        
        
        return(nthin, ntot, nlocs, Rinvs)
    else:
        return(nthin, ntot, nlocs, Rinvs)
   
def deathLoc(A, rhos, ntot, nthin, locs_current, Rinvs, V_list, lam, index, nlocs, parr):
    
    i = random.randint(nthin)
    nobs = ntot - nthin
    p = A.shape[0]
    
    V_del = V_list[:,index[nobs+i]]
    
    if random.uniform() < nthin/lam * (1+np.sum(np.exp(V_del))) * b(nthin-1)/(1-b(nthin)):
        
        # print("death")
        
        oldLoc = locs_current[index[nobs + i]]
        
        index.pop(nobs + i)
        
        dbase = distance_matrix(locs_current[index],[oldLoc])
        dbase = dbase[:,0]
        
        rs = [np.exp(-dbase*rho) for rho in rhos]
        
        # if parr:
        #     Rinvs = np.array(ray.get([ deathUpdater.remote(Rinvs[j],rs[j],nobs + i) for j in range(p) ]))
        # else:
        Rinvs = np.array([ deathUpdate(Rinvs[j],rs[j],nobs + i) for j in range(p) ])
        
        
        nthin -= 1 
        ntot -= 1
        
        
        return(nthin, ntot, nlocs, Rinvs)
    
    else:
        return(nthin, ntot, nlocs, Rinvs)


def locs_move(A, rhos, ntot, nthin, mu, locs_current, Rinvs, V_list, lam, index, nlocs, parr):
    
    if random.uniform() < b(nthin):
        
        return(birthLoc(A, rhos, ntot, nthin, mu, locs_current, Rinvs, V_list, lam, index, nlocs, parr))
    else:
        
        return(deathLoc(A, rhos, ntot, nthin, locs_current, Rinvs, V_list, lam, index, nlocs, parr))
    
    


def MCMC_LMC_MN_POIS(thisLMC_MN, locs, sigma_prior_A, alpha_prior, beta_prior, alpha_prior_lam, beta_prior_lam, sigma_prior_mu, m_prior, sigma_prop_A, sigma_prop_rho, sigma_mom_V, delta, L, nbd, A_init, rho_init, mu_init, V_init, thinLocs_init, lam_init, size, diag, parr):
    
    # if parr:
    #     ray.init()
    
    p = thisLMC_MN.shape[0] # nb of lines/types 
    nobs = locs.shape[0] # nb of columns/locations
    
    nthin_current = thinLocs_init.shape[0]
    ntot_current = nobs + nthin_current
    nlocs = ntot_current
    
    index_current = [x for x in range(ntot_current)]
    
    ## lists for parameters with variable shape
    
    V_list = np.zeros(shape=(p,int(ntot_current+size*nbd)))
    locs_list = np.zeros(shape=(int(ntot_current+size*nbd),2))
    
    Y_list = np.zeros(shape=(p,ntot_current*50))
    

    
    ## chain containers
    
    A_mcmc = np.zeros(shape=(size,p,p))
    rho_mcmc = np.zeros(shape=(size,p))
    mu_mcmc = np.zeros(shape=(size,p))
    lam_mcmc = np.zeros(shape=size)

    
    ## initial state
    
    
    V_list[:,:ntot_current] = V_init
    
    locs_list[:nobs] = locs
    locs_list[nobs:ntot_current] = thinLocs_init
    
    Y_list[:,:nobs] = thisLMC_MN
    
    A_mcmc[0] = A_init 
    rho_mcmc[0] = rho_init
    mu_mcmc[0] = mu_init
    lam_mcmc[0] = lam_init
    
    
        
        
    
    ## current state
    

    
    Rinv_current = np.array([np.linalg.inv(covMatrix(locs_list[index_current],locs_list[index_current],rho)) for rho in rho_init])
    # detRinv_current = np.array([np.linalg.det(R) for R in Rinv_current])
    
    A_current = A_init
    
    # V_current = V_init
    
    # ind_ntot = np.ones(ntot_current)
    # centeredV_current = V_list[:,index_current] - np.outer(mu_init,np.ones(ntot_current))
    
    if diag:
        
        np.save("locs0", locs_list[index_current])
        np.save("V0.csv", V_list[:,index_current])
        # for j in range(p):
        #         np.savetxt(str(j)+"R0.csv", Rinv_current[j], delimiter=",")
    
    
    state = 1
    
    while state<size:
        
        for i in range(int(nbd)):
            
            # print(i)
        
            nthin_current, ntot_current, nlocs, Rinv_current = locs_move(A_current, rho_mcmc[state-1], ntot_current, nthin_current, mu_mcmc[state-1], locs_list, Rinv_current, V_list, lam_mcmc[state-1], index_current, nlocs, parr)
       
        
        V_list[:,index_current] = V_move(sigma_mom_V,p,ntot_current,delta,L,V_list[:,index_current],Y_list[:,:ntot_current],Rinv_current,A_current,mu_mcmc[state-1]) 
        

        
        centeredV_current = V_list[:,index_current] - np.outer(mu_mcmc[state-1],np.ones(ntot_current))
        
        for j in range(p):
            
            
            A_current[j] = A_move(sigma_prior_A, A_current[j], np.linalg.inv(A_current)[:,j], ntot_current, p, centeredV_current, Rinv_current[j], sigma_prop_A)
            
            
            
        
        A_mcmc[state] = A_current
        
        
        # if parr:
        #     rho_mcmc[state], Rinv_current = zip(*ray.get([rho_mover.remote(A_current[j], centeredV_current, Rinv_current[j], rho_mcmc[state-1,j], alpha_prior, beta_prior, sigma_prop_rho, locs_list[index_current]) for j in range(p)]))
        # else:
        for j in range(p):
            rho_mcmc[state,j], Rinv_current[j] = rho_move(A_current[j], centeredV_current, Rinv_current[j], rho_mcmc[state-1,j], alpha_prior, beta_prior, sigma_prop_rho, locs_list[index_current])
    
        
        
        if diag:
            np.save("locs"+str(state), locs_list[index_current])
            np.save("V"+str(state), V_list[:,index_current])
            # for j in range(p):
            #     np.savetxt(str(j)+"R"+str(state)+".csv", Rinv_current[j], delimiter=",")
            
        mu_mcmc[state] = mu_move(A_current, ntot_current, p, Rinv_current, m_prior, sigma_prior_mu, V_list[:,index_current])
        
        lam_mcmc[state] = gamma.rvs(alpha_prior_lam + ntot_current, scale=1/(beta_prior_lam+1))
        
        state+=1
        print(state)

    # if parr:
    #     ray.shutdown()

    return(A_mcmc, rho_mcmc, mu_mcmc, lam_mcmc)


















    