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

print(str(sys.argv[1]).split('/')[6:8])
with open(sys.argv[1], 'r') as f:
    XYZ = np.array(list(csv.reader(f, delimiter=',')))[:,:3].astype(np.float) # load coordinates
    mat = umap.umap_.fuzzy_simplicial_set(
        XYZ,
        n_neighbors=300,
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
    u,s,vh = tl.partial_svd(mat.todense(), n_eigenvecs=3) # 3 should be enough because the data is x,y,z times beads
    mat_svd = np.dot(u, np.dot(np.diag(s), vh))
    scipy.sparse.save_npz(str(sys.argv[1])+'_sparse_graph.npz', mat)
    np.save('/media/garner1/hdd1/gpseq/'+str(sys.argv[1]).split('/')[6]+'_'+str(sys.argv[1]).split('/')[7]+'_svd_graph.npz', mat_svd)

