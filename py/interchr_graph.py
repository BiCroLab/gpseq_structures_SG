#!/usr/bin/env python
# coding: utf-8

# In[62]:


import os
import csv
import numpy as np
import random
import scipy.sparse
import sys
import tensorly as tl
import warnings
warnings.filterwarnings('ignore')

with open('../csv/chrs.csv', newline='') as csvfile:
    chroms = list(csv.reader(csvfile,delimiter='\t'))

filename=sys.argv[1] #'/media/garner1/hdd1/gpseq/10000G/cf_000001/coords.csv_sparse_graph.npz'

with open(filename, 'rb') as f:
    mat = scipy.sparse.load_npz(f).tocoo()  #load the graph adjacency matrix coords.csv_sparse_graph.npz

    rcds = list(zip(mat.row,mat.col,mat.data))

    newr = []; newc = []; newv = []
    for rcd in rcds:
        row = rcd[0]
        col = rcd[1]
        proximity = rcd[2]
        row_chr = chroms[row][0]; col_chr = chroms[col][0] 
        if row_chr != col_chr:
            newr.append(row)
            newc.append(col)
            newv.append(proximity)

    newmat = scipy.sparse.coo_matrix((newv, (newr, newc)), shape=mat.shape)
    scipy.sparse.save_npz(filename+'_interchr.npz', newmat)

