#!/usr/bin/env python
# coding: utf-8
import os
import csv
import numpy as np
import random
import scipy.sparse
import sys
import tensorly as tl
import warnings
warnings.filterwarnings('ignore')

with open(sys.argv[1], 'r') as f:
    mat = scipy.sparse.load_npz(str(sys.argv[1]))  #load the graph adjacency matrix coords.csv_sparse_graph.npz
    u,s,vt = tl.partial_svd(mat.todense(), n_eigenvecs=101) # 3 should be enough because the data is x,y,z times beads
    np.savez(str(sys.argv[1])+'_tsvd', u=u,s=s,vt=vt)

