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
import pickle as pkl

tl.set_backend('numpy')

path = '/media/garner1/hdd1/gpseq/10000'
files = os.listdir(path) # dir list in path with different 3D and graph SingleCell representations

samples = int(sys.argv[1]) #number of structure to consider
comm = int(sys.argv[2])  # number of communities requested
ind = int(sys.argv[3])   # label the sampling of #samples over the 10000 population of SC (you need many of these)

config_sample = random.sample(files, k=samples) # sample k times without replacement from configurations
T = np.zeros(shape = (samples, 3043, 3043), dtype = np.float32 )
graph_idx = 0

for config in config_sample:
    filename = os.path.join(path, config)+'/coords.csv_sparse_graph.npz'
    print(filename)
    if os.path.isfile(filename): 
        T[graph_idx,:,:] = scipy.sparse.load_npz(filename).todense()
        continue
    else:
        continue
    graph_idx += 1

factors = non_negative_parafac(T, rank=comm, n_iter_max=1000, verbose=1, init='random', tol=1e-8)
del T

save = True
load = False

fileName = '/media/garner1/hdd1/gpseq' + '/info_10000' + '/nnparafac_no-svd' + '_comm' + str(comm) + '_sample' + str(ind) + '_size' + str(samples) + '.pkl'
fileObject = open(fileName, 'wb')

if save:
    pkl.dump(factors, fileObject)
    fileObject.close()

fileName = '/media/garner1/hdd1/gpseq' + '/info_10000' + '/cf-sampled_no-svd' + '_comm' + str(comm) + '_sample' + str(ind) + '_size' + str(samples) + '.pkl'
fileObject = open(fileName, 'wb')
if save:
    pkl.dump(config_sample, fileObject)
    fileObject.close()

# if load:
#     fileObject2 = open(fileName, 'wb')
#     modelInput = pkl.load(fileObject2)
#     fileObject2.close()
