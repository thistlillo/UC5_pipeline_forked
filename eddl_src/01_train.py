import fire
import json
import pandas as pd
from posixpath import join
import pyecvl.ecvl as ecvl
from sklearn.model_selection import ParameterGrid
from utils.data_partitioning import make_splits, make_splits_without_normal
from utils.ecvl_ds_utils import to_img_ecvl_yml_str

from neural_nets.cnn_module import EddlCnnModule_ecvl
from neural_nets.eddl_augmentations import train_augs, test_augs
import numpy as np

import neptune.new as neptune
import yaml

def ecvl_yaml(name, description, filenames, labels, train_ids, valid_ids, test_ids):
    print(f"filenames, {type(filenames)}, {len(filenames)}")
    print(f"labels, {type(labels)}, {len(labels)}")
    
    d = {
        "name"        : name,
        "description" : description,
        "classes"     : [], 
        "images"      : [],
        "split"       : dict(training = train_ids, 
                            validation = valid_ids, 
                            test=test_ids)
    }
    imgs = []
    for fn, l in zip(filenames, labels):
        imgs.append({
            "location": fn,
            "label": l
        })
    d["images"] = imgs
    d["classes"] = sorted(list(set(l for ll in labels for l in ll)))
    
    return d

def to_list(value):
    return [value] if type(value) is not list else value
   

def build_ecvl_ds(exp_fld=".", out_fn="cnn_ds.yml", train_p=0.7, valid_p=0.1, shuffle_seed=1, labels="mesh", descr="n/a", dev=False, use_normal_class=True):
    ds = pd.read_csv( join(exp_fld, "img_reports.tsv"), sep="\t").set_index("filename")
    ds["labels"] = ds[labels+"_labels"].apply(lambda x: sorted(list(set([int(l) for l in x.split(";")]))) )  # add new column without mesh or auto prefix
    #>
    with open( join(exp_fld, labels + "_lab2index.json")) as fin:
        l2i = json.load(fin)
    if dev:
        with open( join(exp_fld, labels + "_index2lab.json")) as fin:
            i2l = json.load(fin)
            for i, lab in i2l.items():
                print(f" - {i}: {lab}")
    
    #>
    if use_normal_class:
        lab = l2i["normal"]
        print(f"normal label: {lab}")
        #<

        train_idxs, valid_idxs, test_idxs = make_splits(ds, train_p, valid_p, shuffle_seed, lab)
    else:
        train_idxs, valid_idxs, test_idxs = make_splits_without_normal(ds, train_p, valid_p, shuffle_seed)
    #<
    filenames = train_idxs + valid_idxs + test_idxs
    labels = []
    train_indexes = np.arange(len(train_idxs))
    labels += ds.loc[train_idxs, "labels"].values.tolist()
    valid_indexes = np.arange(len(valid_idxs)) + len(train_idxs)
    labels += ds.loc[valid_idxs, "labels"].values.tolist()
    test_indexes = np.arange(len(test_idxs)) + len(train_idxs) + len(valid_idxs)
    labels += ds.loc[test_idxs, "labels"].values.tolist()

    for fn, ids in zip( ["train_ids.txt", "valid_ids.txt", "test_ids.txt"], [train_idxs, valid_idxs, test_idxs]):
        with open(join(exp_fld, fn), "w") as fout:
            fout.write("\n".join([str(id) for id in ids]))
    #>
  
    ecvl_ds = ecvl_yaml("without_normal", "ds without normal class", filenames, labels, train_indexes.tolist(), valid_indexes.tolist(), test_indexes.tolist())
    with open(out_fn, "w") as fout:
        yaml.dump(ecvl_ds, fout)
    #<
    
    #ecvl_str = to_img_ecvl_yml_str(ds, train_idxs, valid_idxs, test_idxs, descr)
    ecvl_fn = join(exp_fld, out_fn)
    with open(ecvl_fn, "w") as fout:
        yaml.safe_dump(ecvl_ds, fout, default_flow_style=None)
    #<

    print(f"ECVL dataset (cnn) saved at: {ecvl_fn}")
#<

# --------------------------------------------------

def init_neptune(dev, remote_log):
    if dev:
        neptune_mode = "debug"
    elif remote_log:
        neptune_mode = "async"
    else:
        neptune_mode = "offline"
    print(f"NEPTUNE REMOTE LOG, mode set to {neptune_mode}")
    run = neptune.init(project="UC5-DeepHealth", mode = neptune_mode)
    # run["description"] = description if description else "cnn_module"
    # run["configuration"] = conf
    # run["num image classes"] = self.n_classes
    return run 


def cross_validate(k=5, exp_fld=".", out_fn="best_cnn.onnx", load_file=None, n_epochs=200, batch_size=32, optimizer=["adam"],
        learning_rate=[0.01], momentum=[0.9], patience=5, patience_kick_in=200, 
        check_val_every=10, seed=1, shuffle_seed=2, description="n/a",
        gpu_id=[1], eddl_cs_mem="full_mem", verbose=False, dev=False, remote_log=None, cnn_model="resnet18"):
    img_size = 224



def train_cnn(ds_fn=None, exp_fld=".", out_fn="best_cnn.onnx", load_file=None, n_epochs=200, batch_size=32, optimizer=["adam"], 
        learning_rate=[0.01], momentum=[0.9], patience=5, patience_kick_in=200, 
        check_val_every=5, seed=1, shuffle_seed=2, description="n/a",
        gpu_id=[1], eddl_cs_mem="mid_mem", verbose=False, dev=False, remote_log=None, fine_tune=False):
    
    #>
    img_size = 224  # make this a param when it is possible to select the backbone cnn
    grid = {
        "batch_size": to_list(batch_size),
        "optimizer": to_list(optimizer),
        "learning_rate": to_list(learning_rate),
        "momentum": to_list(momentum),
        "gpu_id": [gpu_id],
        # required by the module 
        "n_epochs": [n_epochs],
        "patience": [patience],
        "patience_kick_in": [patience_kick_in],
        "verbose": [verbose],
        "dev": [dev],
        "eddl_cs": ["gpu" if gpu_id else "cpu"],
        "eddl_cs_mem": [eddl_cs_mem],
        "remote_log": [remote_log],
        "out_fn": [out_fn],
        "check_val_every": [check_val_every],
        "exp_fld": [exp_fld],
        "description": [description],
        "fine_tune": to_list(fine_tune)
    }
    parameters = ParameterGrid(grid)
    print(f"input parameters lead to |runs|= {len(parameters)}")
    #<
    
    drop_last = {"training": True, "validation": False, "test": False}
    augs = ecvl.DatasetAugmentations(augs=[train_augs(img_size), test_augs(img_size), test_augs(img_size)])
    ecvl.AugmentationParam.SetSeed(seed)
    ecvl.DLDataset.SetSplitSeed(shuffle_seed)
    
    #>
    ds_fn = ds_fn or join(exp_fld, "cnn_ds.yml")
    print(f"input dataset: {ds_fn}")
    
    def load_ecvl_dataset(filename, bs, gpu):  
        num_workers = 8 * np.count_nonzero(gpu)
        print(f"using num workers = {num_workers}")
        
        dataset = ecvl.DLDataset(filename, batch_size=bs, augs=augs, 
                ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, 
                num_workers=num_workers, queue_ratio_size= 4, drop_last=drop_last)
        return dataset
    #<

    import os
    from yaml import dump
    run = init_neptune(dev=dev, remote_log=remote_log)
    for i, params in enumerate(parameters):
        print(f"{i+1}/{len(parameters)} runs")
        run_fld = join(exp_fld, f"run_{i}")
        os.makedirs(run_fld, exist_ok=True)
        params["out_fld"] = run_fld
        with open(join(run_fld, "config.yml"), "w") as fout:
            dump(params, fout, default_flow_style=None)
        
        dataset = load_ecvl_dataset(ds_fn, params["batch_size"], gpu=gpu_id)
        
        cnn_module = EddlCnnModule_ecvl(dataset, params, neptune_run=run, name=str(i))
        cnn_module.train()
        cnn_module.delete_nn()
        del cnn_module
    
    print(f" run {i} completed")
    #
    print(f"{len(parameters)} runs completed")
# 
        



def train_cnn_inner(ds_fn="cnn_ds.yml", exp_fld=".", optimizer=["adam"], seed=1, gpu=None, eddl_cs_mem="full_mem", remote_log=None):
    pass

def train_rec():
    pass

if __name__ == "__main__":
    fire.Fire({
        "prepare_training": build_ecvl_ds,
        "train_cnn": train_cnn,
        "train_rec": train_rec
    })