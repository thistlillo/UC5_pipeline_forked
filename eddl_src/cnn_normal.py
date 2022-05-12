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


def create_model(gpu, trainable=0):
    base_cnn = eddl.download_resnet18(top=True)
    layer_names = [_.name for _ in base_cnn.layers]
    for name in layer_names:
        eddl.setTrainable(base_cnn, name, trainable)

    cnn_in = eddl.getLayer(base_cnn, "input")
    cnn_top = eddl.getLayer(base_cnn, "top")
    dense_layer = eddl.HeUniform(eddl.Dense(cnn_top, 2, name="out_dense"))
    dense_layer.initialize()
    cnn_out = eddl.Softmax(dense_layer, name="cnn_out")
    cnn = eddl.Model([cnn_in], [cnn_out])
    optimizer = eddl.adam()
    cs = eddl.CS_GPU(g=gpu, mem="full_mem") if gpu else eddl.CS_CPU(th=2, mem="full_mem")
    eddl.build(cnn, optimizer, ["softmax_cross_entropy"], ["accuracy"], cs, init_weights=False)
    return cnn, layer_names
#< create_model

def validation(cnn: eddl.Model, ds:ecvl.DLDataset, do_test=False):
    eddl.set_mode(cnn, mode=0)
    ds.SetSplit(ecvl.SplitType.validation if not do_test else ecvl.SplitType.test)
    ds.ResetBatch(shuffle=True)
    n_batches = ds.GetNumBatches()
    stage = "test" if do_test else "validation"
    print(f"{stage} set: {n_batches} batches")

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


def training(cnn: eddl.Model, ds: ecvl.DLDataset, n_epochs: int, out_fld: str, run, name="cnn", patience=10, fine_tune=0, base_layer_names=[]):
    ds.SetSplit(ecvl.SplitType.training)
    n_training_batches = ds.GetNumBatches()
    n_validation_batches = ds.GetNumBatches(ecvl.SplitType.validation)
    
    patience_kick_in = 20
    best_loss = 1_000_000
    best_acc = 0
    best_v_loss = 1_000_000
    best_v_acc = 0
    patience_run = 0

    losses = []
    v_losses = []
    accs = []
    v_accs = []

    print(f"training starts, epochs {n_epochs}, train batches: {n_training_batches}, validation batches: {n_validation_batches}")
    for epoch in range(n_epochs):
        ds.ResetBatch(shuffle=True)
        ds.Start()
        for bi in tqdm(range(n_training_batches)):
            _, X, Y = ds.GetBatch()
            #X = Tensor.fromarray(images)
            #Y = Tensor.fromarray(labels)
            eddl.train_batch(cnn, [X], [Y])
        ds.Stop()
        loss, acc = validation(cnn, ds, do_test=False)
        v_loss, v_acc = validation(cnn, ds, do_test=True)
        losses.append(loss)
        v_losses.append(v_loss)
        accs.append(acc)
        v_accs.append(v_acc)
        print(f"epoch {epoch}, loss: {loss:.4f}, acc: {acc:.4f}, validation loss: {v_loss:.4f}, validation acc: {v_acc:.4f}")
        if v_loss < best_v_loss:
            best_v_loss = v_loss
            best_v_acc = v_acc
            patience_run = 0
            if fine_tune > 0:
                for layer_name in base_layer_names:
                    eddl.setTrainable(cnn, layer_name, True)
        else:
            patience_run += 1
            if patience_run > patience_kick_in:
                break
    ds.Stop()
    print(f"training finished, best validation loss: {best_v_loss:.4f}, best validation acc: {best_v_acc:.4f}")
   
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
        run[f"{name}/train/loss"].log(losses[-1], step=ei)
        run[f"{name}/train/acc"].log(accs[-1], step=ei)

        ds.Stop()

        print("validation")
        v_loss, v_acc = validation(cnn, ds)
        v_losses.append(v_loss)
        v_accs.append(v_acc)
        run[f"{name}/valid/loss"].log(v_losses[-1], step=ei)
        run[f"{name}/valid/acc"].log(v_accs[-1], step=ei)
        if v_acc > best_v_acc:
            out_fn = join(out_fld, "best_val_acc_chkp.onnx")
            eddl.save_net_to_onnx_file(cnn, out_fn)
            eddl.save(cnn, out_fn.replace(".onnx", ".bin"))
            print(f"ei={ei}, saved: {out_fn} and onnx")

        print("test")
        t_loss, t_acc = validation(cnn, ds, do_test=True)
        run[f"{name}/test/loss"].log(t_loss, step=ei)
        run[f"{name}/test/acc"].log(t_acc, step=ei)


        print(f"{ei+1}/{n_epochs}: loss {losses[-1]:.2f}, acc {accs[-1]:.2f}")
        print(f"{ei+1}/{n_epochs}, validation: loss {v_losses[-1]:.2f}, acc {v_accs[-1]:.2f}")
        # if (ei==2) and (fine_tune==1):
        #     for l in base_layer_names:
        #         eddl.setTrainable(cnn, l, 0)
    #<
    results = dict(
        train_loss = losses,
        valid_loss = v_losses,
        test_loss = t_loss,
        train_acc = accs,
        valid_acc = v_accs,
        test_acc = t_acc
    )
    res_ofn = join(out_fld, "results.tsv")
    df = pd.DataFrame.from_dict(results)
    df.to_csv(res_ofn, sep="\t", index=False)
    print(f"saved results to {res_ofn}")
    print("train ends here. all done.")
#< train
            

def ecvl_yaml(filenames, labels, train_ids, valid_ids, test_ids):
    d = {
        "name"        : "normal-not_normal",
        "description" : "iu frontal-not_normal",
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
def main(base_net="resnet18", seed=2, shuffle_seed=11, valid_p=0.1, dev=False, bs=64, gpu=None, fine_tune=0):
    ds = pd.read_csv( "/mnt/datasets/uc5/std-dataset/img_ds_no_text.tsv", sep="\t", na_filter=False, index_col="image_filename")
    # ds = pd.read_csv( "/opt/uc5/results/eddl_exp/eddl_phi2_exp-eddl_phi2_100_2000/img_reports_phi2_enc.tsv", sep="\t", na_filter=False, index_col="filename")
    
    img_fld = "/mnt/datasets/uc5/std-dataset/image"
    out_fld = "/opt/uc5/results/eddl_exp/iu_normal_vs_rest"
    os.makedirs(out_fld, exist_ok=True)
    
    #>
    normal = ds.loc[ds["normal"]==1] 
    no_normal = ds.loc[ds["normal"]==0].sample(len(normal), random_state=shuffle_seed, axis=0)

    print(f"normal, shape: {normal.shape}")
    print(f"rest, shape: {no_normal.shape}")

    stats = pd.DataFrame()
    label_counts = ds.sum(axis=0)
    stats["original"] = label_counts
    stats["selected"] = no_normal.sum(axis=0)
    stats.to_csv( join(out_fld, "label_stats.tsv"), sep="\t")
    #<

    #>
    normal_filenames = normal.index.to_list()
    rest_filenames = no_normal.index.to_list()
    #<

    #>
    filenames = normal_filenames + rest_filenames
    labels = [0] * len(normal_filenames) + [1] * len(rest_filenames)
    X = np.array(filenames)
    y = np.array(labels)
    assert X.shape[0] == 2 * normal.shape[0]
    assert y.shape[0] == 2 * normal.shape[0]

    print(f"X, shape: {X.shape}")
    print(f"y, shape: {y.shape}")
    print(f"1s in y: {nnz(y)}, 0s in y: {len(y)-nnz(y)}")
    #<
    
    #>
    neptune_mode = "offline" if dev else "async"
    run = neptune.init(project="UC5-DeepHealth", mode = neptune_mode)
    run["description"] = f"normal vs the rest, ecvl dl, fine tune: {fine_tune}"
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
        num_workers = 6 * nnz(gpu) if gpu else 8
       
        drop_last = {"training": True, "validation": (gpu and nnz(gpu)>1), "test": (gpu and nnz(gpu)>1)}

        augs = ecvl.DatasetAugmentations(augs=[train_augs, test_augs, test_augs])
        multiplier = 1 if not gpu else nnz(gpu)
        
        print("using batch size: ", bs)
        
        dataset = ecvl.DLDataset(join(kfold_out, "dataset.yml"), batch_size=bs, augs=augs, 
            ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, 
            num_workers=num_workers, queue_ratio_size= 4, drop_last=drop_last)
        # for name, part, l in zip(["train", "valid", "test"], 
        #                 [ecvl.SplitType.training, ecvl.SplitType.validation, ecvl.SplitType.test], 
        #                 [len(train), len(valid), len(test)]):
        #     print("- {}: {} batches, total ex: {}, originally {} (remember last batch)".format(name,
        #                 dataset.GetNumBatches(split=part),  
        #                 dataset.GetNumBatches(split=part)* bs, 
        #                 l ) )
        cnn, base_layer_names = create_model(gpu, fine_tune)
        training(cnn, dataset, n_epochs=1000, out_fld=kfold_out, run=run, name="{}/{}_fold".format(i+1, n_splits), fine_tune=fine_tune, base_layer_names=base_layer_names)
        del cnn
        # ---
    #< for over stratified k fold
#< main


if __name__ == "__main__":
    fire.Fire(main)