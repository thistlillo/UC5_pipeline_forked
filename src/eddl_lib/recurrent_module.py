# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import gc
import humanize as H
import numpy as np
import pandas as pd
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor, DEV_CPU, DEV_GPU
from posixpath import join
from tqdm import tqdm
from text.metrics import compute_bleu_edll as compute_bleu
from text.vocabulary import Vocabulary
import time

from eddl_lib.uc5_dataset import Uc5Dataset
from eddl_lib.recurrent_models import recurrent_lstm_model, nonrecurrent_lstm_model
from eddl_lib.recurrent_models import generate_text
import text.reports as reports
from eddl_lib.jaccard import Jaccard

import neptune.new as neptune

# https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)


class EddlRecurrentModule:
    def __init__(self, config):
        self.conf = Bunch(**config)
        self.verbose = self.conf.verbose

        self.cnn = self.load_cnn()
        self.ds = Uc5Dataset(self.conf, version="simple")
        self.voc_size = self.ds.vocab.n_words

        if self.conf.load_file:
            print(f"loading model from file {self.conf.load_file}")
            self.rnn = self.load_model()
        else:
            self.rnn = self.build_model()
        self.rnn2 = None  # non-recurrent version of self.rnn
        
        self.run = self.init_neptune()
    #<
    
    def init_neptune(self):
        if self.conf["dev"]:
            neptune_mode = "debug"
        elif self.conf["remote_log"]:
            neptune_mode = "async"
        else:
            neptune_mode = "offline"
        run = neptune.init(project="UC5-DeepHealth", mode = neptune_mode)
        run["description"] = "rnn_module"
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

    def load_cnn(self):
        filename = self.conf.cnn_file
        cnn = eddl.import_net_from_onnx_file(filename)
        print(f"trained cnn read from: {filename}")
        print(f"cnn input shape {cnn.layers[0].input.shape}")
        print(f"cnn output shape {cnn.layers[-1].output.shape}")
        
        eddl.build(cnn, eddl.adam(0.01), ["softmax_cross_entropy"], ["accuracy"], # not relevant: it is used only in forwarding
            self.comp_serv(), init_weights=False)
        print("cnn model built successfully")
        for name in [_.name for _ in cnn.layers]:
            eddl.setTrainable(cnn, name, False)
        eddl.summary(cnn)
        return cnn
    #<        
        
    def load_model(self, filename=None, for_predictions=False):
        filename = filename or self.conf.load_file
        print(f"loading file from file: {self.conf.load_file}")
        onnx = eddl.import_net_from_onnx_file(self.conf.load_file) 
        return self.build_model(onnx)
    #<

    def create_model(self, visual_dim, semantic_dim, for_predictions=False):
        # cnn
        vs = self.voc_size  # vocabulary size
        if not for_predictions:
            model = recurrent_lstm_model(visual_dim, semantic_dim, vs, self.conf.emb_size, self.conf.lstm_size)
        else:
            assert self.rnn is not None
            model = nonrecurrent_lstm_model(visual_dim, semantic_dim, vs, self.conf.emb_size, self.conf.lstm_size)
        #<
        print(f"recurrent model created, for predictions? {for_predictions}")
        eddl.summary(model)
        return model
    #<

    def get_optimizer(self):
        opt_name = self.conf.optimizer
        if opt_name == "adam":
            return eddl.adam(lr=self.conf.lr)
        elif opt_name == "cyclic":
            return eddl.sgd(lr=0.001)
        else:
            assert False
    #<

    def build_model(self, mdl=None, for_predictions=False):
        do_init_weights = (mdl is None)  # init weights only if mdl has not been read from file
        if mdl is None:
            assert self.cnn is not None
            visual_dim = eddl.getLayer(self.cnn, "top").output.shape[1]
            semantic_dim = eddl.getLayer(self.cnn, "cnn_out").output.shape[1]            
            rnn = self.create_model(visual_dim, semantic_dim, for_predictions=for_predictions)
        
        optimizer = self.get_optimizer()
        eddl.build(rnn, optimizer, ["softmax_cross_entropy"], ["accuracy"], eddl.CS_CPU() if for_predictions else self.comp_serv(), init_weights=do_init_weights)
        
        #for name in [_.name for _ in rnn.layers]:
        #    eddl.initializeLayer(rnn, name)
        
        eddl.summary(rnn)
        print('rnn built')
        return rnn
    #<

    def train(self):
        print("*" * 50)
        conf = self.conf
        
        cnn = self.cnn
        cnn_out = eddl.getLayer(cnn, "cnn_out")
        cnn_top = eddl.getLayer(cnn, "top")
        print(f"cnn, top layer {cnn_top.output.shape}, output layer {cnn_out.output.shape}")
        rnn = self.rnn

        ds = self.ds        
        ds.set_stage("train")
        n_epochs = conf.n_epochs
        batch_ids = range(len(ds))

        if conf.dev:
            batch_ids = [0, len(ds)-1]  # first and last batch only
            n_epochs = 2
        #<
        
        # self.run["params/activation"] = self.activation_name
        start_train_t = time.perf_counter()

        for ei in range(n_epochs):
            print(f"> epoch {ei+1}/{n_epochs}")
            ds.last_batch = conf.last_batch
            ds.set_stage("train")
            ds.shuffle()
            eddl.set_mode(rnn, 1)
            eddl.reset_loss(rnn)
            epoch_loss, epoch_acc = 0, 0
            valid_loss, valid_acc = 0, 0
            
            t1 = time.perf_counter()
            for bi in batch_ids:
                images, _, texts = ds[bi]
                
                X = Tensor.fromarray(images)  # , dev=DEV_GPU)
                
                cnn.forward([X])
                cnn_semantic = eddl.getOutput(cnn_out)
                cnn_visual = eddl.getOutput(cnn_top)

                Y = Tensor.fromarray(texts)
                Y = Tensor.onehot(Y, self.voc_size)
                
                eddl.train_batch(rnn, [cnn_visual, cnn_semantic], [Y])

                loss = eddl.get_losses(rnn)[0]
                acc = eddl.get_metrics(rnn)[0]
                epoch_loss += loss
                epoch_acc += acc
                if bi % 20 == 0:
                    step = ei*len(ds) + bi
                    self.run["train/batch/loss"].log(loss, step=step)
                    self.run["train/batch/acc"].log(acc, step=step)
            #< batch
            epoch_end = time.perf_counter()
            print(f"epoch completed in {H.precisedelta(epoch_end-t1)}")

            epoch_loss = epoch_loss / len(batch_ids)
            epoch_acc = epoch_acc / len(batch_ids)
            self.run["training/epoch/loss"].log(epoch_loss)
            self.run["training/epoch/acc"].log(epoch_acc)
            self.run["time/training/epoch"].log(epoch_end-t1)
            
            expected_t = (epoch_end - start_train_t) * (n_epochs - ei - 1) / (ei+1)
            print(f"rnn expected training time (without early beaking): {H.precisedelta(expected_t)}")


            if (ei+1) % 50 == 0 or conf.dev:
                print("** generating text during training")
                bleu, _ = self.predict(stage="valid")
                self.run["trainin/bleu"].log(bleu, step=ei)
                self.save_checkpoint()

            if (ei+1) % conf.check_val_every != 0:
                continue

            val_start = time.perf_counter()
            valid_loss, valid_acc = self.validation()

            val_end = time.perf_counter()
            self.run["validation/epoch/loss"].log(valid_loss, step=ei)
            self.run["validation/epoch/acc"].log(valid_acc, step=ei)
          
            # times
            self.run["time/validation/epoch"].log(val_end-val_start)
            self.run["time/train+val/epoch"].log(val_end - t1, step=ei)
        #< epoch
        end_train_t = time.perf_counter()
        self.save()
        print(f"rnn training complete: {H.precisedelta(end_train_t - start_train_t)}")
    #<

    def validation(self):
        ds = self.ds
        cnn = self.cnn
        cnn_out = eddl.getLayer(cnn, "cnn_out")
        cnn_top = eddl.getLayer(cnn, "top")
        rnn = self.rnn
        ds.set_stage("valid")
        batch_ids = list(range(len(ds)))
        
        loss = 0
        acc = 0
        for bi in batch_ids:
            images, _, texts = ds[bi]
            X = Tensor.fromarray(images)
            cnn.forward([X])
            cnn_semantic = eddl.getOutput(cnn_out)
            cnn_visual = eddl.getOutput(cnn_top)
            Y = Tensor.fromarray(texts)
            Y = Tensor.onehot(Y, self.voc_size)
            eddl.eval_batch(rnn, [cnn_visual, cnn_semantic], [Y])
            loss += eddl.get_losses(rnn)[0]
            acc += eddl.get_metrics(rnn)[0]
        # for over batches
        
        loss = loss / len(ds)
        acc = acc / len(ds)
        print(f"validation, loss: {loss:.2f}, acc: {acc:.2f}")
        return loss, acc
    #<

    def test(self):
        # NOT IMPLEMENTED, use predict
        pass
    #<

    def get_network(self):
        return self.rnn
    #<

    def save_checkpoint(self, filename="rec_checkpoint.bin"):
        filename = join( self.conf.exp_fld, filename)
        eddl.save(self.get_network(), filename)
        print(f"saved checkpoint for the recurrent model (.bin format): {filename}")

    def save(self, filename=None):
        filename = join(self.conf.exp_fld, filename) if filename else self.conf.out_fn
        eddl.save_net_to_onnx_file(self.get_network(), filename)
        bin_out_fn = self.conf.out_fn.replace(".onnx", ".bin")
        eddl.save(self.get_network(), bin_out_fn)
        print(f"onnx trained recurrent model saved at: {self.conf.out_fn}")
        print(f"\t  binary model saved at: {bin_out_fn}")
        return filename
    #<

    # uses a non-recurrent model for the predictions
    def predict(self, stage="test"):
        self.rnn2 = self.build_model(for_predictions=True)
        rnn = self.rnn2
        
        cnn = self.cnn
        #> test on CPU (ISSUE related to eddl.getStates(.) when running on GPU)
        eddl.toCPU(cnn) 
        eddl.toCPU(rnn)
        eddl.toCPU(self.rnn)
        #<
    
        #> copy parameters from the trained recurrent network (see recurrent_models.py for layer names)
        layers_to_copy = [
            "visual_features", "dense_alpha_v",
            "semantic_features", "dense_alpha_s", "co_attention",
            "lstm_cell", "out_dense", "word_embs"
        ]
        for l in layers_to_copy:
            eddl.copyParam(eddl.getLayer(self.rnn, l), eddl.getLayer(rnn, l))
        #<
        
        #> save the model for predictions
        fn = self.conf.out_fn
        onnx_fn = fn.replace(".onnx", "_pred.onnx")
        bin_fn = onnx_fn.replace(".onnx", ".bin")
        eddl.save_net_to_onnx_file(rnn, onnx_fn)
        eddl.save(rnn, bin_fn)
        print(f"recurrent model used for predictions saved at:")
        print(f"\t - onnx: {onnx_fn}")
        print(f"\t - bin: {bin_fn}")

        #> connection cnn -> rnn
        # image_in = eddl.getLayer(cnn, "input") 
        cnn_out = eddl.getLayer(cnn, "cnn_out")
        cnn_top = eddl.getLayer(cnn, "top")
        #<

        eddl.set_mode(rnn, mode=0)
        ds = self.ds
        ds.set_stage(stage)
        dev = self.conf.dev
        n_tokens = self.conf.n_tokens

        #>
        # batch size, can be set to 1 for clarity
        # print(f"1: {len(ds)}")
        # ds.batch_size = 1
        # print(f"2: {len(ds)}")
        ds.last_batch = "drop"
        bs = ds.batch_size
        print(f"text generation on {stage}, using batches of size: {bs}")
        #< 
        
        #> for over test dataset
        bleu = 0
        generated_word_idxs = np.zeros( (bs * len(ds), n_tokens), dtype=int)
        t1 = time.perf_counter()
        for i in range(len(ds)):
            images, _, texts = ds[i]
            #> cnn forward
            X = Tensor(images)
            eddl.forward(cnn, [X])
            cnn_semantic = eddl.getOutput(cnn_out)
            cnn_visual = eddl.getOutput(cnn_top)
            #<
            if dev: 
                print(f"batch, images: {X.shape}")
                print(f"\t- output. semantic: {cnn_semantic.shape}, visual: {cnn_visual.shape}")
                     
            batch_gen = \
                generate_text(rnn, n_tokens, visual_batch=cnn_visual, semantic_batch=cnn_semantic, dev=False)
            
            generated_word_idxs[i*bs:i*bs+images.shape[0], :] = batch_gen
            # measure bleu
            bleu += compute_bleu(batch_gen, texts)
            # if dev:
            #     for i in range(images.shape[0]):
            #         print(f"*** batch {i+1} / {len(ds)} gen word idxs ***")
            #         print(batch_gen[i, :])
            if dev:
                break
        #< for i over batches in the test dataset
        t2 = time.perf_counter()
        print(f"text generation on {stage} in {H.precisedelta(t2-t1)}")
        bleu = bleu / len(ds)
        self.run[f"{stage}/bleu"] = bleu
        self.run[f"{stage}/time"] = t2 - t1

        rnn = None
        self.rnn2 = None
        gc.collect()

        if self.conf.eddl_cs == "gpu":
            print("moving modules back to GPU")
            eddl.toGPU(self.rnn)
            eddl.toGPU(self.cnn)

        return bleu, generated_word_idxs
    #< predict
#< class