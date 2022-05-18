# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:14:30 2022

@author: alier
"""

import numpy as np
from numpy import random
from scipy.spatial import distance_matrix


def covMatrix(x,y,rho):

        
    return(np.exp(-distance_matrix(x,y)*rho))


def A_move(sigma_prior, A_j, Ainv_j, n, p, Y, Rinv_j, sigma_prop):
    
    A_j_prime = A_j + sigma_prop * random.normal(size=p)
    
    AjpmAj = A_j_prime - A_j
    
    if random.uniform < np.abs(1+AjpmAj@Ainv_j)**n * np.exp(-1/2*AjpmAj@(Y@Rinv_j@np.transpose(Y)+1/sigma_prior**2*np.identity(p))@np.transpose(A_j_prime + A_j)):
        return(A_j_prime)
    else:
        return(A_j)
    
def rho_move(detR_j_inv, A_j, Y, R_j_inv, rho_j, alpha_prior, beta_prior, sigma_prop, locs):
    
    rho_j_prime = np.abs(rho_j + sigma_prop * random.normal())
    
    R_j_inv_prime = np.linalg.inv(covMatrix(locs,locs,rho_j_prime))
    
    detR_j_inv_prime = np.linalg.det(R_j_inv_prime)
    
    AjY = A_j@Y
    
    if random.uniform < np.sqrt(detR_j_inv_prime/detR_j_inv) * np.exp(-1/2*AjY@(R_j_inv_prime-R_j_inv)@np.transpose(AjY)) * (rho_j_prime/rho_j)**(alpha_prior-1) * np.exp(-beta_prior(rho_j_prime-rho_j)):
        return(rho_j_prime, R_j_inv_prime, detR_j_inv_prime)
    else:
        return(rho_j, R_j_inv, detR_j_inv)































    