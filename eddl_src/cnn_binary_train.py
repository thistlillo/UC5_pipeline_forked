import fire
import os
from posixpath import join
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
import time
import humanize as H
import pyeddl.eddl as eddl
import pyecvl.ecvl as ecvl
from pyeddl.tensor import Tensor

import neptune.new as neptune
from tqdm import tqdm

from neural_nets.early_stopping import UpEarlyStopping, ProgressEarlyStopping

min_dl_workers = 16



# --------------------------------------------------
def build_cnn(backbone, gpu, mem, config):
    base_cnn = eddl.download_resnet18(top=True)
    # layer_names = [_.name for _ in base_cnn.layers]
    cnn_in = eddl.getLayer(base_cnn, "input")
    cnn_top = eddl.getLayer(base_cnn, "top")
    dense_layer = eddl.HeUniform(eddl.Dense(cnn_top, 2, name="out_dense"))
    dense_layer.initialize()
    cnn_out = eddl.Softmax(dense_layer, name="cnn_out")
    cnn = eddl.Model([cnn_in], [cnn_out])
    
    optimizer = eddl.adam(lr=config["lr"]) if config else eddl.adam(lr=0.0001)
    
    cs = eddl.CS_GPU(g=gpu, mem=mem) if gpu else eddl.CS_CPU(th=16, mem=mem)
    eddl.build(cnn, optimizer, ["softmax_cross_entropy"], ["accuracy"], cs, init_weights=False)
    return cnn

def validation(cnn: eddl.Model, ds:ecvl.DLDataset, do_test=False):
    eddl.set_mode(cnn, mode=0)
    ds.SetSplit(ecvl.SplitType.validation if (not do_test) else ecvl.SplitType.test)
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
    #< for over batches
    ds.Stop()
    loss = loss / n_batches
    acc = acc / n_batches
    return loss, acc
#< validation

# --------------------------------------------------
def full_training(out_fld, cnn, ds, run, config):
    name = os.path.basename(out_fld)
    n_epochs = config["n_epochs"]
    
    ds.SetSplit(ecvl.SplitType.training)
    n_training_batches = ds.GetNumBatches()
    n_validation_batches = ds.GetNumBatches(ecvl.SplitType.validation)
    n_test_batches = ds.GetNumBatches(ecvl.SplitType.test)

    stop_criterion = UpEarlyStopping()
    # progress_criterion = ProgressEarlyStopping()
    ei = 0
    losses, accs = [], []
    v_losses, v_accs = [], []
    t_losses, t_accs = [], []
    train_t, valid_t, test_t, total_t = [], [], [], []
    best_v_acc = 0
    t0 = time.perf_counter()
    if config["dev"]:
        print("!!! dev set, n_epochs set to 3")
        n_epochs = 3
    
    while ei < n_epochs and (not stop_criterion.stop):  # and (not progress_criterion.stop):
        epoch_loss = 0
        epoch_acc = 0
        ds.SetSplit(ecvl.SplitType.training)
        ds.ResetBatch(shuffle=True)
        eddl.set_mode(cnn, mode=1)
        ds.Start()
        eddl.reset_loss(cnn)
        
        print(f"{ei+1}/{n_epochs} - training: {n_training_batches} batches")
        t1 = time.perf_counter()
        for bi in tqdm(range(n_training_batches), disable=True):
            _, X, Y = ds.GetBatch()
            # print(f"got X, Y: {X.shape}, {Y.shape}", flush=True)
            eddl.train_batch(cnn, [X], [Y])
                
            loss = eddl.get_losses(cnn)[0]
            acc = eddl.get_metrics(cnn)[0]
            epoch_loss += loss
            epoch_acc += acc
        #< training
        t2 = time.perf_counter()
        ds.Stop()
        losses.append(epoch_loss / n_training_batches)
        accs.append(epoch_acc / n_training_batches)
        
        # print(f"\t * training, loss {losses[-1]:.3f}, acc {accs[-1]:.3f}")
        run[f"{name}/train/loss"].log(losses[-1])
        run[f"{name}/train/acc"].log(accs[-1])
        
        print(f"{ei+1}/{n_epochs} - validation: {n_validation_batches} batches")
        v_loss, v_acc = validation(cnn, ds)
        t3 = time.perf_counter()
        v_losses.append(v_loss)
        v_accs.append(v_acc)
        run[f"{name}/valid/loss"].log(v_loss)
        run[f"{name}/valid/acc"].log(v_acc)
        if v_acc > best_v_acc:
            eddl.save_net_to_onnx_file(cnn, join(out_fld, "best_val_acc_chkp.onnx"))

        print(f"{ei+1}/{n_epochs} - test: {n_test_batches} batches")
        t_loss, t_acc = validation(cnn, ds, do_test=True)
        t4 = time.perf_counter()
        run[f"{name}/test/loss"].log(t_loss)
        run[f"{name}/test/acc"].log(t_acc)
        t_losses.append(t_loss)
        t_accs.append(t_acc)

        print(f"{ei+1}/{n_epochs}, training loss {losses[-1]:.3f}, acc {accs[-1]:.3f}")
        print(f"{ei+1}/{n_epochs}, validation: loss {v_losses[-1]:.3f}, acc {v_accs[-1]:.3f}")
        print(f"{ei+1}/{n_epochs}, test: loss {t_losses[-1]:.3f}, acc {t_accs[-1]:.3f}")
        
        train_t.append(t2-t1)
        valid_t.append(t3-t2)
        test_t.append(t4-t3)
        total_t.append(t4-t1)
        ei += 1
        eta = np.mean(total_t) * (n_epochs - ei)

        early_stopping = stop_criterion.append(v_acc)
        # progress_stop = progress_criterion.append(accs[-1])
        print(f"early stopping? {early_stopping}")
        # print(f"progress stop? {progress_stop}")
        print(f"{ei+1}/{n_epochs} ends")
        if ei+1 != n_epochs:
            print(f"estimated time to completion without early breaking: {H.precisedelta(eta)} - stopping now? {early_stopping}")
    #< while
    res = {}
    res["training_loss"] = losses
    res["validation_loss"] = v_losses
    res["test_loss"] = t_losses
    res["training_acc"] = accs
    res["validation_acc"] = v_accs
    res["test_acc"] = t_accs
    res["early_breaking"] = stop_criterion.stop # or progress_criterion.stop
    res["stop_criterion"] = stop_criterion.stop
    # res["progress_criterion"] = stop_criterion.stop
    res["n_epochs"] = ei  # last epoch index + 1 (starts from zero)
    res["best_valid_acc"] = best_v_acc
    res["best_model_path"] = join(out_fld, "best_val_acc_chkp.onnx")
    res["time_elapsed"] = time.perf_counter() - t0
    return res
# --------------------------------------------------
def main(in_fld, out_fld, 
        backbone="resnet18", lr=0.0001, 
        shuffle=11, shuffle_seed=20, n_epochs=500,
        bs=32, gpu=None, mem="full_mem", dev=False):
    config = locals()
    sub_flds = [fn for fn in os.listdir(in_fld) if os.path.isdir(join(in_fld,fn))]
    print(f"number of full training iterations (sub-folders): {len(sub_flds)}")
    if len(sub_flds) == 0:
        sub_flds = [in_fld]  # then there is only 1 folder and it is in_fld

    #>
    neptune_mode = "offline" if dev else "async"
    run = neptune.init(project="UC5-DeepHealth", mode = neptune_mode)
    run["description"] = "chest-xrays8 - normal vs others"
    #<

    #> dataloader
    num_workers = min_dl_workers * nnz(gpu) if gpu else min_dl_workers
    # ! ds specific
    # ! chest-xray8
    mean = [0.1534339, 0.19879097, 0.15122881]
    std = [0.26099038, 0.30683932, 0.2560843]
    img_size = 224
    
    train_augs = lambda x: ecvl.SequentialAugmentationContainer([
            ecvl.AugResizeDim([300, 300]),
            ecvl.AugRotate([-5,5]),
            ecvl.AugToFloat32(divisor=255.0),
            ecvl.AugNormalize(mean, std),
            ecvl.AugRandomCrop([x, x])
        ])
    train_augs = train_augs(img_size)
    test_augs =  lambda x: ecvl.SequentialAugmentationContainer([
                    ecvl.AugResizeDim([300, 300]),
                    # ecvl.AugRandomCrop([size, size]),  # XXX should be parametric, for resnet 18
                    ecvl.AugToFloat32(divisor=255.0),
                    ecvl.AugNormalize(mean, std),
                    ecvl.AugCenterCrop([x, x])
                ])
    test_augs = test_augs(img_size)
    augs = ecvl.DatasetAugmentations(augs=[train_augs, test_augs, test_augs])
    drop_last = {"training": True, "validation": (gpu and nnz(gpu)>1), "test": (gpu and nnz(gpu)>1)}
    multiplier = nnz(gpu) if gpu else 1
    mult_bs = bs * multiplier  # larger batch size when using multiple GPUs
    #<

    
    results = {}

    for i, sub_fld in enumerate(sub_flds):
        print(50 * "-")
        print(f"iteration {i+1}/{len(sub_flds)}")
        # expects sub_fld/datset.yml
        print(f"batch size: {mult_bs}")
        dataset = ecvl.DLDataset(join(in_fld, sub_fld, "dataset.yml"), batch_size=mult_bs, augs=augs, 
            ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, 
            num_workers=num_workers, queue_ratio_size= 2 * num_workers, drop_last=drop_last)
        cnn = build_cnn(backbone, gpu, mem, config)
        res = full_training(join(in_fld, sub_fld), cnn, dataset, run, config)
        results["sub_fld"] = res
        # run training iteration
    #< for over subfolders
    df = pd.DataFrame.from_dict(results)
    df.to_csv(join(out_fld, "results.tsv"), sep="\t")
    print(f"results saved in: {join(out_fld, 'results.tsv')}")
    print("done.")

# # --------------------------------------------------
if __name__ == "__main__":
    fire.Fire(main)