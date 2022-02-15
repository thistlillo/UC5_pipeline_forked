# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
from data_partitioning import DataPartitioner
import numpy as np
from posixpath import join

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

def encode_labels_one_hot(labels, n_classes, l1normalization=False):
    out = np.zeros((n_classes,), dtype=float)
    n_labels = len(labels)
    out[labels] = 1.0
    # l1normalization should be applied with a CrossEntropyLoss
    if l1normalization:
        out = out / n_labels
    return out
#<