# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import humanize as H
import numpy as np
import pandas as pd
from posixpath import join
import time
from tqdm import tqdm

import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from eddl_lib.uc5_dataset import Uc5Dataset
from eddl_lib.jaccard import Jaccard
import text.reports as reports
from text.vocabulary import Vocabulary
from utils.misc import Timer

import neptune.new as neptune

# https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)
    

class EddlCnnModule:
    def __init__(self, config):
        self.conf = Bunch(**config)
        self.verbose = self.conf.verbose
                #section
        self.ds = Uc5Dataset(self.conf, version="simple")
        # version is irrelevant for cnn training
        # "simple" means the text part will not be provided as seq of sentences. 

        self.layer_names = None
        
        if self.conf.load_file:
            print(f"loading model from file {self.conf.load_file}")
            self.cnn = self.load_model()
        else:
            self.cnn = self.build_cnn()

        self.run = self.init_neptune()
        
        self.patience_kick_in = self.conf.patience_kick_in
        self.patience = self.conf.patience
        self.patience_run = 0
        self.best_validation_loss = 1_000_000
        self.best_validation_acc = -1_000_000
    #<

    def init_neptune(self):
        if self.conf["dev"]:
            neptune_mode = "debug"
        elif self.conf["remote_log"]:
            neptune_mode = "async"
        else:
            neptune_mode = "offline"
        run = neptune.init(project="UC5-DeepHealth", mode = neptune_mode)
        run["description"] = "cnn_module"
        return run 
    #<
   
    def comp_serv(self, eddl_cs=None, eddl_mem=None):
        eddl_cs = eddl_cs or self.conf.eddl_cs
        eddl_mem = eddl_mem or self.conf.eddl_cs_mem

        if self.verbose:
            print("creating computing service:")
            print(f"computing service: {eddl_cs}")
            print(f"memory: {eddl_mem}")
        
        return eddl.CS_GPU(g=self.conf.gpu_id, mem=self.conf.eddl_cs_mem) if eddl_cs == 'gpu' else eddl.CS_CPU(th=2, mem=eddl_mem)
    #< 

    def download_base_cnn(self, top=True):
        return eddl.download_resnet18(top=top)  # , input_shape=[3, 256, 256]) 
    #<

    # returns an output layer with the activation specified via cli
    def get_out_layer(self, top_layer, version=None, layer_name="cnn_out"):
        out_layer_act = version or self.conf.cnn_out_layer
        print(f"cnn, output layer: {version}")
        res = None
        print(f"adding classification layer, number of classes {self.ds.n_classes}")
        dense_layer = eddl.HeUniform(eddl.Dense(top_layer, self.ds.n_classes, name="out_dense"))
        dense_layer.initialize()
        if out_layer_act == "sigmoid":
            res = eddl.Sigmoid(dense_layer, name=layer_name)
        elif out_layer_act == "softmax":
            res = eddl.Softmax(dense_layer, name=layer_name)
        else:
            assert False

        return res
    #<

    def create_cnn(self):
        remove_top_layer = True
        base_cnn = self.download_base_cnn(top=remove_top_layer)  # top=True -> remove output layer    
        self.layer_names = [_.name for _ in base_cnn.layers]
        cnn_in = eddl.getLayer(base_cnn, "input")
        cnn_top = eddl.getLayer(base_cnn, "top")
        cnn_out = self.get_out_layer(cnn_top)
        
        cnn = eddl.Model([cnn_in], [cnn_out])
        return cnn
    #<

    def build_cnn(self, cnn=None):
        if cnn is None:
            cnn = self.create_cnn()
        #<
        loss_str = self.get_loss_name()
        optimizer = self.get_optimizer()
        print(f"using loss {loss_str} (output layer: {self.conf.cnn_out_layer})")
        eddl.build(cnn, optimizer, [loss_str], ["accuracy"], self.comp_serv(), init_weights=False)  # losses, metrics, 

        print(f"cnn built: resnet18")
        return cnn
    #<

    def load_model(self, filename=None):
        filename = filename or self.conf.load_file
        cnn = eddl.import_net_from_onnx_file(filename)
        return self.build_cnn(cnn)
    #<

    def get_loss_name(self):
        name = "softmax_cross_entropy"
        print("output layer:", self.conf.cnn_out_layer)
        if self.conf.cnn_out_layer == "sigmoid":
            name = "binary_cross_entropy"
        return name
    #<

    def get_optimizer(self):
        opt_name = self.conf.optimizer
        if opt_name == "adam":
            return eddl.adam(lr=self.conf.lr)
        elif opt_name == "cyclic":
            assert False
            return eddl.sgd(lr=0.001)
        else:
            assert False
    #<

    def get_network(self):
        return self.cnn
    #<

    def train(self):
        ds = self.ds
        cnn = self.get_network()
        
        ds.set_stage("train")
        n_epochs = self.conf.n_epochs
        batch_ids = range(len(ds))
        epoch_loss = 0
        epoch_acc = 0
        best_training_loss = 0
        best_training_acc = 0

        if self.conf["dev"]:
            batch_ids = [0, len(ds)-1]  # first and last batch only
            n_epochs = 2  # 2 epochs only
            check_val_every = 1

        early_stop = False  # set to True if the patience threshold is reached during training
        start_train_t = time.perf_counter()
        self.run["train/start_time"] = start_train_t
        for ei in range(n_epochs):
            print(f"{ei+1} / {n_epochs} starting, patience: {self.patience_run}/{self.patience} [kick-in: {self.patience_kick_in}]")
            ds.set_stage("train")
            ds.shuffle()
            eddl.reset_loss(cnn)

            t1 = time.perf_counter()
            valid_loss, valid_acc = 0, 0
            for bi in batch_ids:
                if (bi + 1) % 50 == 0:
                    print(f"batch {bi+1}/{len(batch_ids)}")
                images, labels, _ = ds[bi]
                X = Tensor.fromarray(images)
                Y = Tensor.fromarray(labels)
                eddl.train_batch(cnn, [X], [Y])

                loss = eddl.get_losses(cnn)[0]
                acc = eddl.get_metrics(cnn)[0]
                epoch_loss += loss
                epoch_acc += acc
                    
                if bi % 20 == 0:
                    self.run["train/batch/loss"].log(loss, step=ei * len(batch_ids) + bi)
                    self.run["train/batch/acc"].log(acc, step=ei * len(batch_ids) + bi)
            #< for over batches (1 epoch)
            
            epoch_end = time.perf_counter()
            # print(f"\t time: {H.precisedelta(t2-t1)}")
            print(f"training epoch completed in {H.precisedelta(epoch_end-t1)}")

            # loss
            epoch_loss = epoch_loss / len(batch_ids)
            epoch_acc = epoch_acc / len(batch_ids)
            self.run["training/epoch/loss"].log(epoch_loss)
            self.run["training/epoch/acc"].log(epoch_acc)
            #<

            self.run["time/training/epoch"].log(epoch_end- t1, step=ei)
            expected_t = (epoch_end - start_train_t) * (n_epochs - ei - 1) / (ei+1)
            print(f"cnn expected training time (without early beaking): {H.precisedelta(expected_t)}")

            if (ei + 1 ) % self.conf["check_val_every"] != 0:
                # SKIP VALIDATION
                continue

            # --------------------------------------------------
            # validation
            val_start = time.perf_counter()
            valid_loss, valid_acc = self.validation()
            val_end = time.perf_counter()
            #<
            print(f"training+val epoch completed in {H.precisedelta(val_end-t1)}")

            self.run["validation/epoch/loss"].log(valid_loss, step=ei)
            self.run["validation/epoch/acc"].log(valid_acc, step=ei)
            self.run["time/validation/epoch"].log(val_end-val_start, step=ei)
            self.run["time/train+val/epoch"].log(val_end - t1, step=ei)
            
            
            #< patience and checkpoint
            if valid_loss < self.best_validation_loss:
                # save checkpoint
                print(f"saving checkpoint, epoch {ei}")
                self.save_checkpoint()
                # if patience is running, reset patience
                if self.patience_run > (self.patience // 2):
                    self.patience = self.patience + int(self.patience * 0.2)
                self.patience_run = 0
                self.best_validation_loss = valid_loss
                self.best_validation_acc = valid_acc
                best_training_loss = epoch_loss
                best_training_acc = epoch_acc
            elif ei > self.patience_kick_in:
                self.patience_run += 1
            #<
            if self.patience_run > self.patience:
                print(f"early breaking, patience {self.patience}")
                early_stop = True
                break 
            #< 
        #< epochs

        # log according to early_stop
        if not early_stop:
            self.run["training/loss"] = epoch_loss
            self.run["training/acc"] = epoch_acc
            self.run["validation/loss"] = valid_loss
            self.run["validation/acc"] = valid_acc
        else:
            self.cnn = self.load_checkpoint()
            self.save()
            self.run["training/loss"] = best_training_loss
            self.run["training/acc"] = best_training_acc
            self.run["validation/loss"] = self.best_validation_loss
            self.run["validation/acc"] = self.best_validation_acc
        #<
        end_train_t = time.perf_counter()
        print(f"training complete: {H.precisedelta(end_train_t - start_train_t)}")
    #< train

    def validation(self):
        ds = self.ds
        cnn = self.get_network()
        ds.set_stage("valid")
        batch_ids = list(range(len(ds)))
        loss = 0
        acc = 0
        for bi in batch_ids:
            images, labels, _ = ds[bi]
            X = Tensor.fromarray(images)
            Y = Tensor.fromarray(labels)
            eddl.eval_batch(cnn, [X], [Y])
            
            loss += eddl.get_losses(cnn)[0]
            acc += eddl.get_metrics(cnn)[0]
        loss = loss / len(ds)
        acc = acc / len(ds)

        return loss, acc
    #<

    def test(self):
        ds = self.ds
        cnn = self.get_network()

        ds.set_stage("test")
        print("unimplemented")
    #<

    def save(self, filename=None):
        filename = join( self.conf.exp_fld) if filename else self.conf.out_fn
        eddl.save_net_to_onnx_file(self.get_network(), filename)
        print(f"model saved, location: {filename}")
        return filename
    #<
    
    def save_checkpoint(self, filename="cnn_checkpoint.onnx"):
        filename = join(self.conf.exp_fld, filename)
        eddl.save_net_to_onnx_file(self.get_network(), filename)
        print(f"saved checkpoint: {filename}")
    #<
    def load_checkpoint(self, filename="cnn_checkpoint.onnx"):
        filename = join(self.conf.exp_fld, filename)
        print("loading last checkpoint")
        return self.build_cnn(self.load_model(filename))
    #<
#< class