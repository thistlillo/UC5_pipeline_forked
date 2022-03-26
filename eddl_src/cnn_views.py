import fire
import pandas as pd
from posixpath import join
import numpy as np
import random
from numpy import count_nonzero as nnz
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
import yaml
import os

import pyeddl.eddl as eddl
import pyecvl.ecvl as ecvl
from pyeddl.tensor import Tensor

import neptune.new as neptune
from tqdm import tqdm

from neural_nets.early_stopping import UpEarlyStopping


def create_model(gpu):
    base_cnn = eddl.download_resnet18(top=True)
    # layer_names = [_.name for _ in base_cnn.layers]
    cnn_in = eddl.getLayer(base_cnn, "input")
    cnn_top = eddl.getLayer(base_cnn, "top")
    dense_layer = eddl.HeUniform(eddl.Dense(cnn_top, 2, name="out_dense"))
    dense_layer.initialize()
    cnn_out = eddl.Softmax(dense_layer, name="cnn_out")
    cnn = eddl.Model([cnn_in], [cnn_out])
    optimizer = eddl.adam()
    cs = eddl.CS_GPU(g=gpu, mem="full_mem") if gpu else eddl.CS_CPU(th=2, mem="full_mem")
    eddl.build(cnn, optimizer, ["softmax_cross_entropy"], ["accuracy"], cs, init_weights=False)
    return cnn
#< create_model

def validation(cnn: eddl.Model, ds:ecvl.DLDataset, do_test=False):
    eddl.set_mode(cnn, mode=0)
    ds.SetSplit(ecvl.SplitType.validation if not do_test else ecvl.SplitType.test)
    ds.ResetBatch(shuffle=True)
    n_batches = ds.GetNumBatches()
        
    loss = 0
    acc = 0
    ds.Start()
    for bi in range(n_batches):
        _, X, Y = ds.GetBatch()
        #X = Tensor.fromarray(images)
        #Y = Tensor.fromarray(labels)
        eddl.eval_batch(cnn, [X], [Y])
        
        loss += eddl.get_losses(cnn)[0]
        acc += eddl.get_metrics(cnn)[0]
    ds.Stop()
    loss = loss / n_batches
    acc = acc / n_batches
    return loss, acc
#< validation


def training(cnn: eddl.Model, ds: ecvl.DLDataset, n_epochs: int, out_fld: str, run, name="cnn", patience=10):
    ds.SetSplit(ecvl.SplitType.training)
    n_training_batches = ds.GetNumBatches()
    n_validation_batches = ds.GetNumBatches(ecvl.SplitType.validation)
    
    patience_kick_in = 100
    best_loss = 1_000_000
    best_acc = 0
    best_v_loss = 1_000_000
    best_v_acc = 0
    patience_run = 0

    losses = []
    v_losses = []
    accs = []
    v_accs = []

    print("training starts")
    for ei in range(n_epochs):
        epoch_loss = 0
        epoch_acc = 0
        ds.SetSplit(ecvl.SplitType.training)
        ds.ResetBatch(shuffle=True)
        eddl.set_mode(cnn, mode=1)
        ds.Start()
        eddl.reset_loss(cnn)
        valid_loss, valid_acc = 0, 0
        for bi in tqdm(range(n_training_batches), disable=True):
            _, X, Y = ds.GetBatch()
            eddl.train_batch(cnn, [X], [Y])
                
            loss = eddl.get_losses(cnn)[0]
            acc = eddl.get_metrics(cnn)[0]
            epoch_loss += loss
            epoch_acc += acc
        losses.append(epoch_loss / n_training_batches)
        accs.append(epoch_acc / n_training_batches)
        run[f"{name}/train/loss"].log(losses[-1])
        run[f"{name}/train/acc"].log(accs[-1])

        ds.Stop()

        print("validation")
        v_loss, v_acc = validation(cnn, ds)
        v_losses.append(v_loss)
        v_accs.append(v_acc)
        run[f"{name}/valid/loss"].log(v_losses[-1])
        run[f"{name}/valid/acc"].log(v_accs[-1])
        if v_acc > best_v_acc:
            eddl.save_net_to_onnx_file(cnn, join(out_fld, "best_val_acc_chkp.onnx"))

        print("test")
        t_loss, t_acc = validation(cnn, ds, do_test=True)
        run[f"{name}/test/loss"].log(t_loss)
        run[f"{name}/test/acc"].log(t_acc)


        print(f"{ei+1}/{n_epochs}: loss {losses[-1]:.2f}, acc {accs[-1]:.2f}")
        print(f"{ei+1}/{n_epochs}, validation: loss {v_losses[-1]:.2f}, acc {v_accs[-1]:.2f}")
    #<
#< train
            

def ecvl_yaml(filenames, labels, train_ids, valid_ids, test_ids):
    d = {
        "name"        : "frontal-lateral",
        "description" : "iu frontal-lateral images",
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
    d["classes"] = sorted(list(set(labels)))
    return d

# --------------------------------------------------
def main(base_net="resnet18", seed=2, shuffle_seed=11, valid_p=0.1, dev=False, bs=32, gpu=None):
    ds = pd.read_csv( "/mnt/datasets/uc5/UC5_pipeline_forked/experiments_eddl/reports_raw.tsv", sep="\t", na_filter=False )
    img_fld = "/mnt/datasets/uc5/std-dataset/image"
    out_fld = "/opt/uc5/results/eddl_exp/frontal_lat_view"
    os.makedirs(out_fld, exist_ok=True)
    
    ecvl.AugmentationParam.SetSeed(seed)
    ecvl.DLDataset.SetSplitSeed(shuffle_seed)

    #>
    sub = ds.loc[ds.n_images == 2]
    
    frontal = []
    lateral = []
    def split_images(filenames):
        f = filenames.split(";")
        assert len(f) == 2
        frontal.append(f[0])
        lateral.append(f[1])
    
    for row in sub.itertuples():
        split_images(row.image_filename)
    print(f"frontal images: {len(frontal)}, lateral: {len(lateral)}")
    print(f"total: {sub.shape[0]}")
    assert 2 * sub.shape[0] == len(frontal) + len(frontal)
    #<

    #>
    filenames = frontal + lateral
    labels = [1] * len(frontal) + [0] * len(lateral)
    X = np.array(filenames)
    y = np.array(labels)
    assert X.shape[0] == 2 * sub.shape[0]
    assert y.shape[0] == 2 * sub.shape[0]
    #<
    
    #>
    neptune_mode = "offline" if dev else "async"
    run = neptune.init(project="UC5-DeepHealth", mode = neptune_mode)
    run["description"] = "frontal lateral on iu chest x-ray collection"
    #<

    mean = [0.48197903, 0.48197903, 0.48197903]
    std = [0.26261734, 0.26261734, 0.26261734]
    
    img_size = 224
    train_augs = lambda x: ecvl.SequentialAugmentationContainer([
            ecvl.AugResizeDim([300, 300]),
            ecvl.AugRotate([-5,5]),
            ecvl.AugToFloat32(divisor=255.0),
            ecvl.AugNormalize(mean, std),
            ecvl.AugResizeDim([x, x])
        ])
    train_augs = train_augs(img_size)


    test_augs =  lambda x: ecvl.SequentialAugmentationContainer([
                    ecvl.AugResizeDim([x, x]),
                    # ecvl.AugRandomCrop([size, size]),  # XXX should be parametric, for resnet 18
                    ecvl.AugToFloat32(divisor=255.0),
                    ecvl.AugNormalize(mean, std),
                ])
    test_augs = test_augs(img_size)
    
    #>
    n_splits = 5
    skf= StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=shuffle_seed)
    for i, (others, test) in enumerate(skf.split(X, y)):  # others (train+valid), test are indices
        print(100 * "*")
        kfold_out = join(out_fld, f"fold_{i}")
        os.makedirs(kfold_out, exist_ok=True)
        
        test_p = 1 / skf.n_splits  # below: 1-testp, as valid_p specified as a function of dataset size
        # train, valid are indices
        train, valid = train_test_split(
                    others, 
                    test_size=valid_p/(1-test_p),
                    stratify=y[others],
                    shuffle=True, random_state=shuffle_seed+i)
        print(f"-----/test: label distribution, others -  {np.bincount(y[others])}   |   test -  {np.bincount(y[test])}")
        print(f"train/val:  label distribution, train -  {np.bincount(y[train])}    |   validation -  {np.bincount(y[valid])}")
        assert len(train) + len(valid) == len(others)
        
        # ---
        dataset = ecvl_yaml([join(img_fld, fn) for fn in filenames], labels, train.tolist(), valid.tolist(), test.tolist())
        with open(join(kfold_out, "dataset.yml"), "w") as fout:
            yaml.safe_dump(dataset, fout, default_flow_style=True)
        
        # ---
        num_workers = 8 * nnz(gpu) if gpu else 8
       
        drop_last = {"training": True, "validation": (gpu and nnz(gpu)>1), "test": (gpu and nnz(gpu)>1)}

        augs = ecvl.DatasetAugmentations(augs=[train_augs, test_augs, test_augs])
        multiplier = 1 if not gpu else nnz(gpu)
        mult_bs = bs * multiplier
        print(f"batch size {bs} * {multiplier} = {mult_bs}")
        dataset = ecvl.DLDataset(join(kfold_out, "dataset.yml"), batch_size=mult_bs, augs=augs, 
            ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, 
            num_workers=num_workers, queue_ratio_size= 2 * num_workers, drop_last=drop_last)
        # for name, part, l in zip(["train", "valid", "test"], 
        #                 [ecvl.SplitType.training, ecvl.SplitType.validation, ecvl.SplitType.test], 
        #                 [len(train), len(valid), len(test)]):
        #     print("- {}: {} batches, total ex: {}, originally {} (remember last batch)".format(name,
        #                 dataset.GetNumBatches(split=part),  
        #                 dataset.GetNumBatches(split=part)* bs, 
        #                 l ) )
        cnn = create_model(gpu)
        training(cnn, dataset, n_epochs=300, out_fld=kfold_out, run=run, name="{}/{}_fold".format(i+1, n_splits))
        del cnn
        # ---
    #< for over stratified k fold
#< main


if __name__ == "__main__":
    fire.Fire(main)