# # -*- coding: utf-8 -*-
# """
# Created on Mon May 30 21:54:56 2022

# @author: alier
# """

# import numpy as np
# from scipy.stats import wishart

# dim = 1000

# Ws = wishart.rvs(dim+5,np.identity(dim),size=120)


# import time

# t0 = time.time()

# # result1 = [np.linalg.inv(w) for w in Ws]

# # for i in [np.linalg.inv(w) for w in Ws]:
# #     pass

# print([np.linalg.inv(w) for w in Ws])    

# t1 = time.time()

# total1 = t1-t0


# import multiprocessing as mp


# pool = mp.Pool(4)

# t0 = time.time()

# # result2 = pool.imap(np.linalg.inv, Ws, chunksize=30)

# # for i in pool.imap_unordered(np.linalg.inv, Ws, chunksize=30):
# #     pass

# print(pool.imap(np.linalg.inv, Ws, chunksize=30))

# t1 = time.time()

# total2 = t1-t0

# pool.close()



# import numpy as np
# from time import time
# import defs

# # Prepare data
# np.random.RandomState(100)
# arr = np.random.randint(0, 10, size=[200000, 5])
# data = arr.tolist()
# data[:5]


# # def howmany_within_range(row, minimum, maximum):
# #     """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
# #     count = 0
# #     for n in row:
# #         if minimum <= n <= maximum:
# #             count = count + 1
# #     return count

# results = []
# for row in data:
#     results.append(defs.howmany_within_range(row, minimum=4, maximum=8))

# print(results[:10])
# #> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]



# # Parallelizing using Pool.apply()

# import multiprocessing as mp


# # Step 1: Init multiprocessing.Pool()
# pool = mp.Pool(mp.cpu_count())

# # Step 2: `pool.apply` the `howmany_within_range()`
# results = [pool.apply(defs.howmany_within_range, args=(row, 4, 8)) for row in data]

# # Step 3: Don't forget to close
# pool.close()    

# print(results[:10])
# #> [3, 1, 4, 4, 4, 2, 1, 1, 3, 3]




# from multiprocessing import Pool, TimeoutError
# import time
# import os

# def f(x):
#     return x*x

# if __name__ == '__main__':
#     # start 4 worker processes
#     with Pool(processes=4) as pool:

#         # print "[0, 1, 4,..., 81]"
#         print(pool.map(f, range(10)))

#         # print same numbers in arbitrary order
#         for i in pool.imap_unordered(f, range(10)):
#             print(i)

#         # evaluate "f(20)" asynchronously
#         res = pool.apply_async(f, (20,))      # runs in *only* one process
#         print(res.get(timeout=1))             # prints "400"

#         # evaluate "os.getpid()" asynchronously
#         res = pool.apply_async(os.getpid, ()) # runs in *only* one process
#         print(res.get(timeout=1))             # prints the PID of that process

#         # launching multiple evaluations asynchronously *may* use more processes
#         multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
#         print([res.get(timeout=1) for res in multiple_results])

#         # make a single worker sleep for 10 secs
#         res = pool.apply_async(time.sleep, (10,))
#         try:
#             print(res.get(timeout=1))
#         except TimeoutError:
#             print("We lacked patience and got a multiprocessing.TimeoutError")

#         print("For the moment, the pool remains available for more work")

#     # exiting the 'with'-block has stopped the pool
#     print("Now the pool is closed and no longer available")


# from multiprocessing import Pool
# import numpy as np
# from scipy.stats import wishart
# import time

# dim = 1000

# Ws = wishart.rvs(dim+5,np.identity(dim),size=400)

# def stupid(w):
#     return np.linalg.det(w@np.linalg.inv(w))

# t0 = time.time()
# if __name__ == '__main__':
#     with Pool(4) as p:
#         print(p.map(stupid, Ws, chunksize=4))
# print(time.time() - t0)

# t0 = time.time()
# print([stupid(w) for w in Ws])
# print(time.time() - t0)


# from multiprocessing import Pool
# import numpy as np
# from scipy.stats import wishart
# import time

# def stupid(w):
#     return np.linalg.det(w@np.linalg.inv(w))

# dim = 10000

# W1 = wishart.rvs(dim+5,np.identity(dim),size=1)
# W2 = wishart.rvs(dim+5,np.identity(dim),size=1)



# if __name__ == '__main__':
#     pool = Pool(2)
#     t0 = time.time()
#     result1 = pool.apply_async(stupid, [W1])    # evaluate "solve1(A)" asynchronously
#     result2 = pool.apply_async(stupid, [W2])    # evaluate "solve2(B)" asynchronously
#     answer1 = result1.get()
#     answer2 = result2.get()
#     print(time.time() - t0)
    

#     t0 = time.time()
#     args = [W1, W2]
#     results = pool.map(stupid, args)
#     print(time.time() - t0)


# t0 = time.time()
# answer1s = stupid(W1)
# answer2s = stupid(W2)
# print(time.time() - t0)


import numpy as np
from scipy.stats import wishart
import time
import ray

ray.init()

# dim = 5000

# W1 = wishart.rvs(dim+5,np.identity(dim),size=1)
# W2 = wishart.rvs(dim+5,np.identity(dim),size=1)
# W3 = wishart.rvs(dim+5,np.identity(dim),size=1)
# W4 = wishart.rvs(dim+5,np.identity(dim),size=1)
# W5 = wishart.rvs(dim+5,np.identity(dim),size=1)
# W6 = wishart.rvs(dim+5,np.identity(dim),size=1)

dim = 500

Ws = wishart.rvs(dim+5,np.identity(dim),size=5)

def stupid(w):
    return(np.linalg.det(w@np.linalg.inv(w)),w)

# t0 = time.time()
# x1 = stupid(W1)
# x2 = stupid(W2)
# x3 = stupid(W3)
# x4 = stupid(W4)
# x5 = stupid(W5)
# x6 = stupid(W6)
# print(time.time() - t0)

t0 = time.time()
for i in range(10):
    xs, ws = zip(*[stupid(w) for w in Ws])
print(time.time() - t0)



# Define the functions.

@ray.remote
def stupid(w):
    return(np.linalg.det(w@np.linalg.inv(w)),w)


# t0 = time.time()
# # Start two tasks in the background.
# x1_id = stupid.remote(W1)
# x2_id = stupid.remote(W2)
# x3_id = stupid.remote(W3)
# x4_id = stupid.remote(W4)
# x5_id = stupid.remote(W5)
# x6_id = stupid.remote(W6)

# # Block until the tasks are done and get the results.
# x1p, x2p, x3p, x4p, x5p, x6p = ray.get([x1_id, x2_id, x3_id, x4_id, x5_id, x6_id])
# print(time.time() - t0)





t0 = time.time()
for i in range(10):
    xp, wp = zip(*ray.get([stupid.remote(w) for w in Ws]))
print(time.time() - t0)


ray.shutdown()





