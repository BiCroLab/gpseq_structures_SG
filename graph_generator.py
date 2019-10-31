#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import numpy as np
import umap
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from mpl_toolkits.mplot3d import Axes3D
import igraph
from igraph import *
import random
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
import scipy.sparse
import sys

with open(sys.argv[1], 'r') as f:
    XYZ = np.array(list(csv.reader(f, delimiter=',')))[:,:3].astype(np.float) # load coordinates
    mat = umap.umap_.fuzzy_simplicial_set(
        XYZ,
        n_neighbors=50, # this will affect sparsity of the final Hadamard graph
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
    scipy.sparse.save_npz(str(sys.argv[1])+'_sparse_graph.npz', mat)

