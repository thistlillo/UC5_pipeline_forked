# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import fire
import json
import math
import numpy as np
from numpy import count_nonzero as nnz
import pandas as pd
import pickle
from posixpath import join
import random

import text.reports as reports


def make_splits(ds, train_p, valid_p, shuffle_seed, normal_lab, rdm=None):
    negs = ds.labels.apply(lambda l: normal_lab in l)
    poss = ~negs
    print(f"normal images: {nnz(negs)}")
    print(f"other images: {nnz(poss)}")
    neg_idxs = ds[negs].index.values
    pos_idxs = ds[poss].index.values
    
    # rdm not None when multiple splits needs to be generated
    if rdm is None:
        rdm = random.Random()
        rdm.seed(shuffle_seed)
        print(f"using shuffling seed: {shuffle_seed}")
    #< 

    def split(idxs):
        rdm.shuffle(idxs)
        n = len(idxs)
        n_tr, n_va = math.ceil(n * train_p), math.ceil(n * valid_p)
        n_te = n - n_tr - n_va
        print(f"(pos or neg, msg intentionally repeated) requested test size: {(1-train_p-valid_p) * n:.2f}, actual size: {n_te}")
        return idxs[:n_tr], idxs[n_tr:n_tr+n_va], idxs[n_tr+n_va:]
    
    neg_tr, neg_va, neg_te = split(neg_idxs)
    pos_tr, pos_va, pos_te = split(pos_idxs)
    training = neg_tr.tolist() + pos_tr.tolist()
    validation = neg_va.tolist() + pos_va.tolist()
    test = neg_te.tolist() + pos_te.tolist()

    print(f"training, requested size {train_p * len(ds):.2f}, actual size: {len(training)}")
    print(f"validation, requested size {valid_p * len(ds):.2f}, actual size: {len(validation)}")
    print(f"test, requested size {(1-valid_p-train_p) * len(ds):.2f}, actual size: {len(test)}")
    return training, validation, test