import fire
import json
import pandas as pd
from posixpath import join
import pyecvl.ecvl as ecvl
from sklearn.model_selection import ParameterGrid
from utils.data_partitioning import make_splits
from utils.ecvl_ds_utils import to_img_ecvl_yml_str

from neural_nets.cnn_module import EddlCnnModule_ecvl
from neural_nets.eddl_augmentations import train_augs, test_augs

def to_list(value):
    return [value] if type(value) is not list else value
   

def build_ecvl_ds(exp_fld=".", out_fn="cnn_ds.yml", train_p=0.7, valid_p=0.1, shuffle_seed=1, labels="mesh", descr="n/a", dev=False):
    ds = pd.read_csv( join(exp_fld, "img_reports.tsv"), sep="\t", index_col=0)
    ds["labels"] = ds[labels+"_labels"].apply(lambda x: [int(l) for l in x.split(";")])  # add new column without mesh or auto prefix
    #>
    with open( join(exp_fld, labels + "_lab2index.json")) as fin:
        l2i = json.load(fin)
    if dev:
        with open( join(exp_fld, labels + "_index2lab.json")) as fin:
            i2l = json.load(fin)
            
            for i, lab in i2l.items():
                print(f" - {i}: {lab}")\
    # if
    lab = l2i["normal"]
    print(f"normal label: {lab}")
    #<

    train_idxs, valid_idxs, test_idxs = make_splits(ds, train_p, valid_p, shuffle_seed, lab)
    
    #>
    ecvl_str = to_img_ecvl_yml_str(ds, train_idxs, valid_idxs, test_idxs, descr)
    ecvl_fn = join(exp_fld, out_fn)
    with open(ecvl_fn, "w") as fout:
        fout.write(ecvl_str)
    #<

    print(f"ECVL dataset (cnn) saved at: {ecvl_fn}")
#<

# --------------------------------------------------

def train_cnn(ds_fn=None, exp_fld=".", out_fn="best_cnn.onnx", load_file=None, n_epochs=200, batch_size=32, optimizer=["adam"], 
        learning_rate=[0.01], momentum=[0.9], patience=5, patience_kick_in=200, 
        check_val_every=10, seed=1, shuffle_seed=2, description="n/a",
        gpu_id=[1], eddl_cs_mem="full_mem", verbose=False, dev=False, remote_log=None):
    
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
        "description": [description]
    }
    parameters = ParameterGrid(grid)
    print(f"input parameters lead to |runs|= {len(parameters)}")
    #<
    
    drop_last = {"training": True, "validation": False, "test": False}
    augs = ecvl.DatasetAugmentations(augs=[train_augs(img_size), test_augs(img_size), test_augs(img_size)])
    if seed is not None:
        # set seed for the augmentations
        pass
    
    #>
    ds_fn = ds_fn or join(exp_fld, "cnn_ds.yml")
    print(f"input dataset: {ds_fn}")
    def load_ecvl_dataset(filename, bs, gpu):  
        num_workers = 4
        if gpu:
            num_workers = len(gpu) * num_workers
        dataset = ecvl.DLDataset(ds_fn, batch_size=bs, augs=augs, 
                ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, 
                num_workers=num_workers, queue_ratio_size= 2 * num_workers, drop_last=drop_last)
        return dataset
    #<

    import os
    from yaml import dump
    for i, params in enumerate(parameters):
        print(f"{i+1}/{len(parameters)} runs")
        run_fld = join(exp_fld, f"run_{i}")
        os.makedirs(run_fld, exist_ok=True)
        params["out_fld"] = run_fld
        with open(join(run_fld, "config.yml"), "w") as fout:
            dump(params, fout, default_flow_style=None)
        dataset = load_ecvl_dataset(ds_fn, params["batch_size"], gpu=gpu_id)
        cnn_module = EddlCnnModule_ecvl(dataset, params, name=str(i))
        cnn_module.train()
        cnn_module.delete_nn()
        del cnn_module
    
    
        



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