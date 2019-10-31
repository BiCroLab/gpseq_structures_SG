#!/usr/bin/env python
# coding: utf-8

# In[130]:


import os
import csv
import numpy as np
import random
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
import scipy.sparse
import sys

path = '/home/garner1/Work/dataset/gpseq/10000'
files = os.listdir(path) # dir list in path with different configurations

samples = int(sys.argv[1])
rank = int(sys.argv[2]) 

config_sample = random.sample(files, k=samples) # sample k times without replacement from configurations

array_list = [] #initialize array list
for config in config_sample:
    dirname = os.fsdecode(config)
    filename = os.path.join(path, dirname+'/coords.csv_sparse_graph.npz')
    if os.path.isfile(filename): 
        with open(filename, 'r') as f:
            mat = scipy.sparse.load_npz(filename)
            array_list.append(mat.todense())  # densify the sparse mat and add to list
        continue
    else:
        continue

T = np.array(array_list) # the final tensor kXnodesXnodes
factors = non_negative_parafac(T, rank=rank, verbose=1, tol=1e-03)
