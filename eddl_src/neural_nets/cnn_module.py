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

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

import text.reports as reports
from text.vocabulary import Vocabulary

import neptune.new as neptune

from pyeddl._core import Loss, Metric

class Jaccard(Metric):
    def __init__(self):
        Metric.__init__(self, "py_jaccard")

    def value(self, t, y):
        t.info()
        y.info()
        n_labs = t.sum()
        #print(f"n labels {n_labs}", flush=True)
        y_round = y.round()
        #print(f"predicted: {y.round().sum()}", flush=True)
        score = t.mult(y_round).sum()
        #print(f"correctly predicted: {score}", flush=True)
        return score / n_labs


# https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)
    

class EddlCnnModule_ecvl:
    def __init__(self, dataset, config, neptune_run=None, name=""):
        self.ds = dataset
        self.n_classes = len(self.ds.classes_)
       
        print(f"number of classes (output layer): {self.n_classes}")
        self.conf = Bunch(**config)
        self.name = name
        self.out_layer_act = "sigmoid"
        self.verbose = self.conf.verbose
        self.img_size = 224
        # set seed of the augmentation container
        self.layer_names = None
        
        if "load_file" in self.conf:
            print(f"loading model from file {self.conf.load_file}")
            self.cnn = self.load_model()
        else:
            self.cnn = self.build_cnn()

        self.run = self.init_neptune(neptune_run)
        
        self.patience_kick_in = self.conf.patience_kick_in
        self.patience = self.conf.patience
        self.patience_run = 0
        self.best_validation_loss = 1_000_000
        self.best_validation_acc = -1_000_000
    #<

    def delete_nn(self):
        del self.cnn
    #<

    def init_neptune(self, run):
        if not run:
            if self.conf["dev"]:
                neptune_mode = "debug"
            elif self.conf["remote_log"]:
                neptune_mode = "async"
            else:
                neptune_mode = "offline"
            print(f"NEPTUNE REMOTE LOG, mode set to {neptune_mode}")
            run = neptune.init(project="UC5-DeepHealth", mode = neptune_mode)
        
        run["description"] = self.conf.description if "description" in self.conf else "cnn_module"
        run["configuration"] = self.conf
        run["num image classes"] = self.n_classes
        return run 
    #<
   
    def comp_serv(self, eddl_cs=None, eddl_mem=None):
        eddl_cs = eddl_cs or self.conf.eddl_cs
        eddl_mem = eddl_mem or self.conf.eddl_cs_mem

        if self.verbose:
            print("creating computing service:")
            print(f"computing service: {eddl_cs}")
            print(f"memory: {eddl_mem}")

        if self.conf.batch_size == 32:
            lsb = 1
        else:
            lsb = 1
        
        return eddl.CS_GPU(g=self.conf.gpu_id, mem=self.conf.eddl_cs_mem, lsb=lsb) if eddl_cs == 'gpu' else eddl.CS_CPU(th=2, mem=eddl_mem)
    #< 

    def get_loss_name(self):
        name = "softmax_cross_entropy"
        print("output layer:", self.out_layer_act)
        if self.out_layer_act == "sigmoid":
            name = "binary_cross_entropy"
        
        return name
    #<

    def get_optimizer(self):
        opt_name = self.conf.optimizer
        if opt_name == "adam":
            return eddl.adam(lr=self.conf.learning_rate)
        elif opt_name == "cyclic":
            #self.cylic = 
            return eddl.adam(lr=self.conf.learning_rage)
        else:
            assert False
    #<

    def download_base_cnn(self, top=True):
        return eddl.download_resnet101(top=top)  #, input_shape=[1, 224, 224]) 
    #<

    # returns an output layer with the activation specified via cli
    def get_out_layer(self, top_layer, version="sigmoid", layer_name="cnn_out"):
        print(f"cnn, output layer: {version}")
        res = None
        print(f"cnn, number of classes {self.n_classes}")
        dense_layer = eddl.HeUniform(eddl.Dense(top_layer, self.n_classes, name="out_dense"))
        dense_layer.initialize()
        
        if self.out_layer_act == "sigmoid":
            res = eddl.Sigmoid(dense_layer, name=layer_name)
        elif self.out_layer_act == "softmax":
            assert False
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
        loss = eddl.getLoss(loss_str)
        metric = eddl.getMetric("accuracy")  # Jaccard()
        # eddl.build(cnn, optimizer, [loss_str], ["binary_accuracy"], self.comp_serv(), init_weights=False)  # losses, metrics, 
        cnn.build(optimizer, [loss], [metric], self.comp_serv(), initialize=False)  # losses, metrics, 

        print(f"cnn built: resnet18")
        return cnn
    #<

    def load_model(self, filename=None):
        filename = filename or self.conf.load_file
        cnn = eddl.import_net_from_onnx_file(filename)
        return self.build_cnn(cnn)
    #<

    def get_network(self):
        return self.cnn
    #<


    def train(self):
        ds = self.ds
        cnn = self.get_network()
        
        ds.SetSplit(ecvl.SplitType.training)
        n_epochs = self.conf.n_epochs
        n_training_batches = ds.GetNumBatches()
        n_validation_batches = ds.GetNumBatches(ecvl.SplitType.validation)

        print("num batches:")
        print(f" - training: {n_training_batches}")
        print(f" - validation: {n_validation_batches}")

        epoch_loss = 0
        epoch_acc = 0
        best_training_loss = 0
        best_training_acc = 0

        early_stop = False  # set to True if the patience threshold is reached during training
        start_train_t = time.perf_counter()
        self.run[f"{self.name}-train/start_time"] = start_train_t
        for ei in range(n_epochs):
            if self.conf.dev and ei == 2:
                break
            print(f"{ei+1} / {n_epochs} starting, patience: {self.patience_run}/{self.patience} [kick-in: {self.patience_kick_in}]")
            ds.SetSplit(ecvl.SplitType.training)
            ds.ResetBatch(shuffle=True)
            ds.Start()
            
            eddl.reset_loss(cnn)

            t1 = time.perf_counter()
            valid_loss, valid_acc = 0, 0
            for bi in range(n_training_batches):
                if (bi + 1) % 50 == 0:
                    print(f"batch {bi+1}/{n_training_batches}")
                
                _, X, Y = ds.GetBatch()

                #X = Tensor.fromarray(images)
                #Y = Tensor.fromarray(labels)
                eddl.train_batch(cnn, [X], [Y])
                
                loss = eddl.get_losses(cnn)[0]
                acc = eddl.get_metrics(cnn)[0]
                epoch_loss += loss
                epoch_acc += acc
                    
                if bi % 20 == 0:
                    self.run[f"{self.name}-train/batch/loss"].log(loss, step=ei * n_training_batches + bi)
                    self.run[f"{self.name}-train/batch/acc"].log(acc, step=ei * n_training_batches + bi)
            #< for over batches (1 epoch)
            
            ds.Stop()
            epoch_end = time.perf_counter()
            # print(f"\t time: {H.precisedelta(t2-t1)}")
            print(f"training epoch completed in {H.precisedelta(epoch_end-t1)}")

            # loss
            epoch_loss = epoch_loss / n_training_batches
            epoch_acc = epoch_acc / n_training_batches
            self.run[f"{self.name}-training/epoch/loss"].log(epoch_loss)
            self.run[f"{self.name}-training/epoch/acc"].log(epoch_acc)
            #<

            self.run[f"{self.name}-time/training/epoch"].log(epoch_end- t1, step=ei)
            expected_t = (epoch_end - start_train_t) * (n_epochs - ei - 1) / (ei+1)
            print(f"cnn expected training time (without early beaking): {H.precisedelta(expected_t)}")

            if (not self.conf.dev) and ( (ei + 1 ) % self.conf["check_val_every"] != 0 ):
                # SKIP VALIDATION
                continue

            # --------------------------------------------------
            # validation
            val_start = time.perf_counter()
            valid_loss, valid_acc = self.validation()
            val_end = time.perf_counter()
            #<
            print(f"training+val epoch completed in {H.precisedelta(val_end-t1)}")

            self.run[f"{self.name}-validation/epoch/loss"].log(valid_loss, step=ei)
            self.run[f"{self.name}-validation/epoch/acc"].log(valid_acc, step=ei)
            self.run[f"{self.name}-time/validation/epoch"].log(val_end-val_start, step=ei)
            self.run[f"{self.name}-time/train+val/epoch"].log(val_end - t1, step=ei)
            
            
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
            self.run[f"{self.name}-training/loss"] = epoch_loss
            self.run[f"{self.name}-training/acc"] = epoch_acc
            self.run[f"{self.name}-validation/loss"] = valid_loss
            self.run[f"{self.name}-validation/acc"] = valid_acc
        else:
            self.cnn = self.load_checkpoint()
            self.save("best_cnn.onnx")
            self.run[f"{self.name}-training/loss"] = best_training_loss
            self.run[f"{self.name}-training/acc"] = best_training_acc
            self.run[f"{self.name}-validation/loss"] = self.best_validation_loss
            self.run[f"{self.name}-validation/acc"] = self.best_validation_acc
        #<
        end_train_t = time.perf_counter()
        print(f"training complete: {H.precisedelta(end_train_t - start_train_t)}")
    #< train

    def validation(self):
        ds = self.ds
        cnn = self.get_network()
        ds.SetSplit(ecvl.SplitType.validation)
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
    #<

    def test(self):
        ds = self.ds
        cnn = self.get_network()

        ds.set_stage("test")
        print("unimplemented")
    #<

    def save(self, filename):
        filename = join( self.conf.out_fld, filename )
        eddl.save_net_to_onnx_file(self.get_network(), filename)
        print(f"model saved, location: {filename}")
        return filename
    #<
    
    def save_checkpoint(self, filename="cnn_checkpoint.onnx"):
        filename = join(self.conf.out_fld, filename)
        eddl.save_net_to_onnx_file(self.get_network(), filename)
        print(f"saved checkpoint: {filename}")
    #<
    def load_checkpoint(self, filename="cnn_checkpoint.onnx"):
        filename = join(self.conf.out_fld, filename)
        print("loading last checkpoint")
        return self.build_cnn(self.load_model(filename))
    #<
#< class