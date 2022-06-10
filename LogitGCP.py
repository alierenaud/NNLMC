# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 19:07:46 2022

@author: alier
"""

import numpy as np
from numpy import random

from GP import rLMC
from GP import mexpit_col
from GP import multinomial_col


def rLGCP(lam, A, corrFuncs, meanVec):
    
    n = random.poisson(lam)
    locs = random.uniform(size = (n,2))
    
    
    thisLMC = rLMC(A, corrFuncs, np.outer(meanVec,np.ones(n)),locs)
    types = multinomial_col(mexpit_col(thisLMC))
    
    notDeleted = np.sum(types, axis=0)
    
    return(locs[notDeleted.astype(np.bool)], types[:,notDeleted.astype(np.bool)])



