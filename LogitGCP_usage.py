# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 19:46:07 2022

@author: alier
"""

from LogitGCP import rLGCP
import numpy as np
from GP import expCorr
import matplotlib.pyplot as plt


lam = 1000
A = np.array([[0,4],[-4,0]])
rhos = np.array([1,100])
corrFuncs = np.array([expCorr(rho) for rho in rhos])

mu = np.array([-2,-2])

locs, types = rLGCP(lam, A, corrFuncs, mu)



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