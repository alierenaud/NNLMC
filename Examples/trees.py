# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:41:31 2022

@author: alier
"""

import numpy as np
import matplotlib.pyplot as plt


maple = np.loadtxt("maple.csv", delimiter=",")
hickory = np.loadtxt("hickory.csv", delimiter=",")
blackoak = np.loadtxt("blackoak.csv", delimiter=",")
redoak = np.loadtxt("redoak.csv", delimiter=",")
whiteoak = np.loadtxt("whiteoak.csv", delimiter=",")

treeLocs = np.concatenate((maple,hickory,blackoak,redoak,whiteoak))

n_maple = maple.shape[0]
n_hickory = hickory.shape[0]
n_blackoak = blackoak.shape[0]
n_redoak = redoak.shape[0]
n_whiteoak = whiteoak.shape[0]

type_maple = np.transpose(np.repeat([[1,0,0,0,0]],n_maple,axis=0))
type_hickory = np.transpose(np.repeat([[0,1,0,0,0]],n_hickory,axis=0))
type_blackoak = np.transpose(np.repeat([[0,0,1,0,0]],n_blackoak,axis=0))
type_redoak = np.transpose(np.repeat([[0,0,0,1,0]],n_redoak,axis=0))
type_whiteoak = np.transpose(np.repeat([[0,0,0,0,1]],n_whiteoak,axis=0))


treeTypes = np.concatenate((type_maple,type_hickory,type_blackoak,type_redoak,type_whiteoak), axis=1)



#### plot all trees

ntree = treeLocs.shape[0]


randInd = np.arange(ntree)
np.random.shuffle(randInd)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
plt.title("Lansing Woods")
plt.scatter(maple[:,0], maple[:,1], c="tab:blue", label="maple", marker="$\clubsuit$", s=20)
plt.scatter(hickory[:,0], hickory[:,1], c="tab:orange", label="hickory", marker="$\clubsuit$", s=20)
plt.scatter(blackoak[:,0], blackoak[:,1], c="tab:green", label="blackoak", marker="$\clubsuit$", s=20)
plt.scatter(redoak[:,0], redoak[:,1], c="tab:red", label="redoak", marker="$\clubsuit$", s=20)
plt.scatter(whiteoak[:,0], whiteoak[:,1], c="tab:purple", label="whiteoak", marker="$\clubsuit$", s=20)


 
for i in range(ntree):
    if treeTypes[0,randInd[i]] == 1:
        plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:blue", marker="$\clubsuit$", s=20)
    elif treeTypes[1,randInd[i]] == 1:
        plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:orange", marker="$\clubsuit$", s=20)
    elif treeTypes[2,randInd[i]] == 1:
        plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:green", marker="$\clubsuit$", s=20)
    elif treeTypes[3,randInd[i]] == 1:
        plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:red", marker="$\clubsuit$", s=20)
    elif treeTypes[4,randInd[i]] == 1:
        plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:purple", marker="$\clubsuit$", s=20)

ax.legend(bbox_to_anchor=(1, 0.8))

plt.xlim(0,1)
plt.ylim(0,1)

plt.show()
fig.savefig("allTrees.pdf", bbox_inches='tight')

#####


#### plot hickory maple

ntree = treeLocs.shape[0]


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
    if treeTypes[0,randInd[i]] == 1:
        plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:blue", marker="$\clubsuit$", s=20)
    elif treeTypes[1,randInd[i]] == 1:
        plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:orange", marker="$\clubsuit$", s=20)
    # elif treeTypes[2,randInd[i]] == 1:
    #     plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:green", marker="$\clubsuit$", s=20)
    # elif treeTypes[3,randInd[i]] == 1:
    #     plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:red", marker="$\clubsuit$", s=20)
    # elif treeTypes[4,randInd[i]] == 1:
    #     plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:purple", marker="$\clubsuit$", s=20)

ax.legend(bbox_to_anchor=(1, 0.8))

plt.xlim(0,1)
plt.ylim(0,1)

plt.show()
fig.savefig("allTrees.pdf", bbox_inches='tight')

#####


#### plot oaks

ntree = treeLocs.shape[0]


randInd = np.arange(ntree)
np.random.shuffle(randInd)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

# plt.scatter(maple[:,0], maple[:,1], c="tab:blue", label="maple", marker="$\clubsuit$", s=20)
# plt.scatter(hickory[:,0], hickory[:,1], c="tab:orange", label="hickory", marker="$\clubsuit$", s=20)
plt.scatter(blackoak[:,0], blackoak[:,1], c="tab:green", label="blackoak", marker="$\clubsuit$", s=20)
plt.scatter(redoak[:,0], redoak[:,1], c="tab:red", label="redoak", marker="$\clubsuit$", s=20)
plt.scatter(whiteoak[:,0], whiteoak[:,1], c="tab:purple", label="whiteoak", marker="$\clubsuit$", s=20)


 
# for i in range(ntree):
#     # if treeTypes[0,randInd[i]] == 1:
#     #     plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:blue", marker="$\clubsuit$", s=20)
#     # elif treeTypes[1,randInd[i]] == 1:
#     #     plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:orange", marker="$\clubsuit$", s=20)
#     if treeTypes[2,randInd[i]] == 1:
#         plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:green", marker="$\clubsuit$", s=20)
#     elif treeTypes[3,randInd[i]] == 1:
#         plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:red", marker="$\clubsuit$", s=20)
#     elif treeTypes[4,randInd[i]] == 1:
#         plt.scatter(treeLocs[randInd[i],0], treeLocs[randInd[i],1], c="tab:purple", marker="$\clubsuit$", s=20)

ax.legend(bbox_to_anchor=(1, 0.8))

plt.xlim(0,1)
plt.ylim(0,1)

plt.show()
fig.savefig("allTrees.pdf", bbox_inches='tight')

#####





