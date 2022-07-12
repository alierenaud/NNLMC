# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 19:42:38 2022

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt



np.random.seed(0)

n = 65

locs = np.random.uniform(size=(n,2))

rad = 0.15


def dist(x,y):
    
    
    return(np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2))
    

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')


plt.scatter(locs[:,0], locs[:,1])


for i in range(n):

    

    plt.annotate(i, # this is the text
                  locs[i], # these are the coordinates to position the label
                  textcoords="offset points", # how to position the text
                  xytext=(0,10), # distance from text to points (x,y)
                  ha='center', c="none") # horizontal alignment can be left, right or center


plt.xlim(0,1)
plt.ylim(0,1)

fig.savefig("mplot0.pdf", bbox_inches='tight')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.scatter(locs[:,0], locs[:,1])


for i in range(n):

    

    plt.annotate(i, # this is the text
                 locs[i], # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.xlim(0,1)
plt.ylim(0,1)

fig.savefig("mplot1.pdf", bbox_inches='tight')
plt.show()



notdeleted = np.full(n, True)


for i in range(n):
    
    if notdeleted[i]:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        plt.scatter(locs[notdeleted][:,0], locs[notdeleted][:,1])
        plt.scatter(locs[~notdeleted][:,0], locs[~notdeleted][:,1], c="silver")
        plt.scatter(locs[i,0], locs[i,1])
        circle1 = plt.Circle(locs[i], rad, fc = 'none', ec = 'black')
        ax.add_patch(circle1)
    
    
        for z in range(n):
        
            
        
            plt.annotate(z, # this is the text
                         locs[z], # these are the coordinates to position the label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
        
        
        plt.xlim(0,1)
        plt.ylim(0,1)
        
        plt.show()
        fig.savefig("mp"+str(i)+".pdf", bbox_inches='tight')
        
        for j in range(i+1,n):
            if dist(locs[i],locs[j]) < rad:
                notdeleted[j] = False
    




fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

plt.scatter(locs[notdeleted][:,0], locs[notdeleted][:,1])


for i in range(n):

    

    plt.annotate(i, # this is the text
                  locs[i], # these are the coordinates to position the label
                  textcoords="offset points", # how to position the text
                  xytext=(0,10), # distance from text to points (x,y)
                  ha='center', c="none") # horizontal alignment can be left, right or center


plt.xlim(0,1)
plt.ylim(0,1)

fig.savefig("mplotf.pdf", bbox_inches='tight')
plt.show()
































