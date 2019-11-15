#!/usr/bin/env python3.7
# coding: utf-8
import os
import csv
import numpy as np
import random
import tensorly as tl
from tensorly.decomposition import non_negative_parafac, parafac, randomised_parafac
import scipy.sparse
import sys

tl.set_backend('numpy')

path = '/home/garner1/Work/dataset/gpseq/10000'
files = os.listdir(path) # dir list in path with different configurations

samples = int(sys.argv[1])
rank = int(sys.argv[2]) 
ind = int(sys.argv[3])

config_sample = random.sample(files, k=samples) # sample k times without replacement from configurations

T = np.zeros(shape = (samples, 3043, 3043), dtype = np.float32 )
count = 0
for config in config_sample:
    dirname = os.fsdecode(config)
    filename = os.path.join(path, dirname+'/coords.csv_sparse_graph.npz')
    if os.path.isfile(filename): 
        with open(filename, 'r') as f:
            mat = scipy.sparse.load_npz(filename).astype(np.float32).todense()
            # sub_threshold_indices = mat < 0.5
            # mat[sub_threshold_indices] = 0
            u,s,vh = tl.partial_svd(mat, n_eigenvecs=3) # 3 should be enough because the data is x,y,z times beads
            T[count,:,:] = np.dot(u, np.dot(np.diag(s), vh))
            del mat
        continue
    else:
        continue
    count += 1

factors = non_negative_parafac(T, rank=rank, n_iter_max=10000, verbose=1, init='svd', tol=1e-10)
# print(factors[0])
# print([f.shape for f in factors[1]])
# print([tl.norm(factors[1][0][:,ind],2)*tl.norm(factors[1][1][:,ind],2)*tl.norm(factors[1][2][:,ind],2) for ind in range(rank)])
del T

import pickle as pkl
save = True
load = False

fileName = path + '/NNparafac' + '_rank' + str(rank) + '_sample' + str(ind) + '_size' + str(samples) + '.pkl'
fileObject = open(fileName, 'wb')

if save:
    pkl.dump(factors, fileObject)
    fileObject.close()

fileName = path + '/cf-sampled' + '_rank' + str(rank) + '_sample' + str(ind) + '_size' + str(samples) + '.pkl'
fileObject = open(fileName, 'wb')
if save:
    pkl.dump(config_sample, fileObject)
    fileObject.close()

# if load:
#     fileObject2 = open(fileName, 'wb')
#     modelInput = pkl.load(fileObject2)
#     fileObject2.close()
