# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 22:31:51 2022

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt

from GP import makeGrid

np.random.seed(3)

lam = 500

n = np.random.poisson(lam)

locs = np.random.uniform(size=(n,2))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')


plt.scatter(locs[:,0], locs[:,1])



plt.xlim(0,1)
plt.ylim(0,1)

fig.savefig("pplot1.pdf", bbox_inches='tight')
plt.show()



def fct(x):
    return(np.exp(-(0.25-np.sqrt((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))**2/0.005))




res = 100
gridLoc = makeGrid([0,1], [0,1], res)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

imGP = fct(gridLoc).reshape(res,res,order="F")

x = np.linspace(0,1, res+1) 
y = np.linspace(0,1, res+1) 
X, Y = np.meshgrid(x,y) 
    

ax.set_aspect('equal')
   
ff = ax.pcolormesh(X,Y,imGP,cmap="Blues", vmin=0, vmax=1)

fig.colorbar(ff) 



fig.savefig("pplot2.pdf", bbox_inches='tight')
plt.show() 




obs = np.full(n,False)


for i in range(n):
    
    if np.random.uniform() < fct(np.array([locs[i]])):
    
        obs[i] = True
    

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')


plt.scatter(locs[~obs][:,0], locs[~obs][:,1], c="silver")
plt.scatter(locs[obs][:,0], locs[obs][:,1])



plt.xlim(0,1)
plt.ylim(0,1)

fig.savefig("pplot3.pdf", bbox_inches='tight')
plt.show()

### observed

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')



plt.scatter(locs[obs][:,0], locs[obs][:,1])



plt.xlim(0,1)
plt.ylim(0,1)

fig.savefig("pplot4.pdf", bbox_inches='tight')
plt.show()

### thinned

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')


plt.scatter(locs[~obs][:,0], locs[~obs][:,1], c="silver")




plt.xlim(0,1)
plt.ylim(0,1)

fig.savefig("pplot5.pdf", bbox_inches='tight')
plt.show()


    
