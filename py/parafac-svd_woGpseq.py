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

path = '/media/garner1/hdd1/gpseq/10000'
files = os.listdir(path) # dir list in path with different configurations

samples = int(sys.argv[1])
rank = int(sys.argv[2]) 
ind = int(sys.argv[3])

config_sample = random.sample(files, k=samples) # sample k times without replacement from configurations
T = np.zeros(shape = (samples, 3043, 3043), dtype = np.float32 )
count = 0

# print('loading data')           
for config in config_sample:
    # filename = os.path.join(path, config)
    print(filename)
    if os.path.isfile(filename): 
        with open(filename, 'r') as f:
            mat = scipy.load(filename).astype(np.float32)
            T[count,:,:] = mat
            del mat
        continue
    else:
        continue
    count += 1
# print('done!')

factors = non_negative_parafac(T, rank=rank, n_iter_max=100, verbose=1, init='random', tol=1e-6)
del T

import pickle as pkl
save = True
load = False

fileName = path + '/info' + '/NNparafac' + '_rank' + str(rank) + '_sample' + str(ind) + '_size' + str(samples) + '.pkl'
fileObject = open(fileName, 'wb')

if save:
    pkl.dump(factors, fileObject)
    fileObject.close()

fileName = path + '/info' + '/cf-sampled' + '_rank' + str(rank) + '_sample' + str(ind) + '_size' + str(samples) + '.pkl'
fileObject = open(fileName, 'wb')
if save:
    pkl.dump(config_sample, fileObject)
    fileObject.close()

# if load:
#     fileObject2 = open(fileName, 'wb')
#     modelInput = pkl.load(fileObject2)
#     fileObject2.close()
