# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:14:30 2022

@author: alier
"""

import numpy as np
from numpy import random
from scipy.spatial import distance_matrix
from GP import mexpit_col



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
    
    v = random.normal(size=(p,n))
    
    xtemp=x
    vtemp=v

    
    for l in range(L):
        vstar = vtemp - delta/2*(U_MN_prime(xtemp,Y)+U_LMC_prime(xtemp,A,Rinvs,mu,p,n))
        xtemp = xtemp + delta*vstar
        vtemp = vstar - delta/2*(U_MN_prime(xtemp,Y)+U_LMC_prime(xtemp,A,Rinvs,mu,p,n))



        
    # vnew=-vnew
    
    lpi_old=-(U_MN(x,Y) + U_LMC(x,A,Rinvs,mu,p,n)) - 1/2*np.sum(v**2)
    lpi_new=-(U_MN(xtemp,Y) + U_LMC(xtemp,A,Rinvs,mu,p,n)) - 1/2*np.sum(vtemp**2)
    
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





















    