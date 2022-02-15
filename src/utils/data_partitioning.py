# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import fire
import json
import math
import numpy as np
import pandas as pd
import pickle
from posixpath import join
import random

import text.reports as reports

class DataPartitioner:
    def __init__(self, conf, tsv=None):
        self.conf = conf
        self.tsv = tsv

        self.in_tsv = conf["in_tsv"]
        self.exp_fld = conf["exp_fld"]

        self.seed = int(conf["shuffle_seed"])
        self.random = random.Random()
        self.random.seed(self.seed)
        if self.conf["verbose"]:
            print(f"Using shuffling seed: {self.seed}")
        
        self.train_p = float(conf["train_p"])
        self.valid_p = float(conf["valid_p"])
        self.test_p = 1 - ( self.train_p + self.valid_p)
        assert self.test_p > 0
        
        if self.tsv is None:
            if self.conf["verbose"]:
                print(f"DataPartitioner, reading input tsv: {self.in_tsv}")
            self.tsv = pd.read_csv(self.in_tsv, sep=reports.csv_sep, na_filter=False)
        # load support files, needed for encoding input data (image labels, text)
        with open(join(self.exp_fld, "lab2index.json"), "r") as fin:
            self.label2index = json.load(fin)

        with open(join(self.exp_fld, "index2lab.json"), "r") as fin:
            self.index2label = json.load(fin)

        self.n_classes = len(self.label2index)
        if self.conf["verbose"]:
                print(f"number of image labels: {self.n_classes}")
            
        # self.tsv.set_index("id", inplace=True, drop=False)
        self.normal_ids, self.other_ids = self.init_ids()
        if self.conf["verbose"]:
            print("Partioning into train, test, valid. Labels:")
            print("|normal|: ", len(self.normal_ids))
            print("|other|: ", len(self.other_ids))

        self.n_norm_train = math.ceil(self.train_p * len(self.normal_ids))
        self.n_other_train = math.ceil(self.train_p * len(self.other_ids))
        self.n_norm_valid = math.ceil(self.valid_p * len(self.normal_ids))
        self.n_other_valid = math.ceil(self.valid_p * len(self.other_ids))
        self.n_norm_test = len(self.normal_ids) - self.n_norm_train - self.n_norm_valid
        self.n_other_test = len(self.other_ids) - self.n_other_train - self.n_other_valid
    # >init

    def shuffle(self):
        self.random.shuffle(self.stage_ids)
    #<

    def init_ids(self):
        tsv = self.tsv
        c = self.conf
        iii = tsv[ "labels" ] == str(self.label2index["normal"])  # otherwise label2index is int and match fails
        normal_ids = tsv.loc[iii, "filename"].to_list()
        other_ids = tsv.loc[~iii, "filename"].to_list()
        return normal_ids, other_ids
    #<

    def partition(self):
        tsv = self.tsv
        normal_ids = self.normal_ids
        other_ids = self.other_ids
    
        n_norm_train = self.n_norm_train  # math.ceil(self.train_p * len(normal_ids))
        n_other_train = self.n_other_train  # math.ceil(self.train_p * len(other_ids))
        n_norm_valid = self.n_norm_valid  # math.ceil(self.valid_p * len(normal_ids))
        n_other_valid = self.n_other_valid  # math.ceil(self.valid_p * len(other_ids))

        self.random.shuffle(normal_ids)
        self.random.shuffle(other_ids)
        train_ids = normal_ids[:n_norm_train] + other_ids[:n_other_train]
        valid_ids = normal_ids[n_norm_train:n_norm_train+n_norm_valid] + other_ids[n_other_train:n_other_train+n_other_valid]
        test_ids = normal_ids[n_norm_train+n_norm_valid:] + other_ids[n_other_train+n_other_valid:]
        
        assert len(test_ids) == self.n_norm_test + self.n_other_test
        assert len(train_ids) + len(test_ids) + len(valid_ids) == len(tsv.loc[:, "filename"])
        
        self.random.shuffle(train_ids)
        self.random.shuffle(valid_ids)
        self.random.shuffle(test_ids)

        print("data split into training, validation and test:")
        print(f" - normal reports: {len(normal_ids)}, other: {len(other_ids)}")
        print(f" - training set: {len(train_ids)}, normal: {n_norm_train} + {n_other_train}: other")
        print(f" - validation set: {len(valid_ids)}, normal: {n_norm_valid} + {n_other_valid}: other")
        print(f" - test set: {len(test_ids)}, normal : {self.n_norm_test} + {self.n_other_test}: other")

        return train_ids, valid_ids, test_ids
    #> partition

    def partition_and_save(self, out_suffix=None):
        parts = (train_ids, valid_ids, test_ids) = self.partition()
        self.save(parts, out_suffix)
    #<
       
    def save(self, partitions, out_suffix):
        out_filenames = ["train_ids", "valid_ids", "test_ids"]
        if out_suffix:
            out_filenames = [s + out_suffix for s in out_filenames]
            print(f"split, output filenames with suffix {out_suffix}")
        filenames = [join(self.conf["exp_fld"], fn) + ".txt" for fn in out_filenames]
        for a, fn in zip(partitions, filenames):
            with open(fn, "w", encoding="utf-8") as fout:
                fout.write("\n".join(a))
            if self.conf["verbose"]:
                print(f"Saved {fn} with {len(a)} ids")
    #<
#< class

def load_data_split(exp_fld):
    with open(join(exp_fld, "train_ids.txt"), "r", encoding="utf-8") as fin:
        train_ids = [line.strip() for line in fin.readlines()]
    with open(join(exp_fld, "valid_ids.txt"), "r", encoding="utf-8") as fin:
        valid_ids = [line.strip() for line in fin.readlines()]
    with open(join(exp_fld, "test_ids.txt"), "r", encoding="utf-8") as fin:
        test_ids = [line.strip() for line in fin.readlines()]
    print(f"data split read from disk. |train|={len(train_ids)}, |valid|={len(valid_ids)}, |test|={len(test_ids)}")
    return train_ids, valid_ids, test_ids
#<


def create_partitions(args, tsv=None):
    c = {}
    c["in_tsv"] = args.in_tsv
    c["exp_fld"] = args.exp_fld
    c["train_p"] = args.train_p
    c["valid_p"] = args.valid_p
    c["shuffle_seed"] = args.shuffle_seed
    c["term_column"] = args.term_column
    c["verbose"] = args.verbose
    partitioner = DataPartitioner(c, tsv)   
    return partitioner.partition()
#<
