#!/usr/bin/env python
# coding: utf-8
import os
import csv
import numpy as np
import umap
import random
import scipy.sparse
import sys
import tensorly as tl
import warnings
warnings.filterwarnings('ignore')

with open(sys.argv[1], 'r') as f:  # the input is the coord.csv file
    XYZ = np.array(list(csv.reader(f, delimiter=',')))[:,:3].astype(np.float) # load coordinates
    mat = umap.umap_.fuzzy_simplicial_set(
        XYZ,
        n_neighbors=100,     #hard-coded
        random_state=np.random.RandomState(seed=42),
        metric='l2', 
        metric_kwds={},
        knn_indices=None,
        knn_dists=None,
        angular=False,
        set_op_mix_ratio=1.0,
        local_connectivity=2.0,
        verbose=False
    )

    # Truncated SVD
    u,s,vt = tl.partial_svd(mat.todense(), n_eigenvecs=101) # the 101 truncation is hard-coded

    scipy.sparse.save_npz(str(sys.argv[1])+'_sparse_graph.npz', mat)
    np.savez(str(sys.argv[1])+'_tsvd', u=u,s=s,vt=vt)

