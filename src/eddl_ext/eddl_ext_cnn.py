from collections import defaultdict
import fire
import numpy as np
import pandas as pd
import pyeddl.eddl as eddl
import pyecvl.ecvl as ecvl

from text.reports import csv_sep, list_sep

def main(out_fn, exp_fld, img_fld, tsv_file, img_size=224, dev=False):
    df = pd.read_csv(tsv_file, sep=csv_sep)
    print(f"dataset read: {df.shape}")
    print(f"columns: {df.columns}")
    
    # filenames and labels
    filenames = df.filename.tolist()
    labels = df.labels.tolist()
    
    label_hist = defaultdict(int)
    for l in labels:
        print(l)
        n_labels = len(l.split(list_sep))
        label_hist[str(n_labels)] += 1

    for k, v in label_hist.items():
        print(f"{k}: {v} images")

    # counter
    def count_labels(labels):
        cnt = defaultdict(int)
        lab2img = defaultdict(list)
        img2lab = defaultdict(list)
        for i, l in enumerate(labels):
            ll = l.split(list_sep)[:5]  # ll is a seq of labels separated with list_sep
            # print(f"{i:03d}: {len(ll)} [{l if len(ll)==1 else '+'}]")
            for value in ll:
                if ll == "misc":
                    print("MISC")
                    exit()
                cnt[ str(value) ] += 1
                lab2img[ str(value) ].append( filenames[i] )
                img2lab[ filenames[i] ].append(value)
        return cnt, lab2img, img2lab

    cnt, l2i, i2l = 
    (labels)
    n_orig_labels = len(l2i.keys())
    print(f"Originally {n_orig_labels} labels")
    #<

    print("***")
    print(f"number of labels: {len(cnt.keys())}")
    for k, v in cnt.items():
        print(f"label {k}: {v}")
    input()
    
    counts = np.array(list(cnt.values()))    
    np_labels = np.array(list(cnt.keys()))
    

    
    #<

    print("***")
    thresholds = list(range(30, 120, 10))
    for t in thresholds:
        print(f"*** threshold {t}")
        iii = counts < t
        nnz = np.count_nonzero(iii)
        removed = np_labels[iii]
        assert nnz == removed.shape[0]
        removed = set(removed.tolist())
        print(f"threshold {t}, number of removed labels: {nnz}=={len(removed)} - total: {np_labels.shape[0]}, remaining {np_labels.shape[0]-nnz}")
        input()

        

    
