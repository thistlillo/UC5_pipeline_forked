import humanize as H
import numpy as np
import pandas as pd
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from posixpath import join
from tqdm import tqdm
from text.vocabulary import Vocabulary
import time

from eurodll.uc5_dataset import Uc5Dataset
import text.reports as reports
from eurodll.jaccard import Jaccard

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
        
        eddl.build(cnn, eddl.adam(0.01), ["softmax_cross_entropy"], ["accuracy"], # used only in forwarding
            self.comp_serv(), init_weights=False)
        print("cnn model built successfully")
        for name in [_.name for _ in cnn.layers]:
            eddl.setTrainable(cnn, name, False)
        eddl.summary(cnn)
        return cnn
    #<        
        
    def load_model(self, filename=None):
        filename = filename or self.conf.load_file
        print(f"loading file from file: {self.conf.load_file}")
        onnx = eddl.import_net_from_onnx_file(self.conf.load_file) 
        return self.build_model(onnx)
    #<

    def create_model(self, for_training=True):
        # cnn
        cnn = self.cnn
        eddl.summary(cnn)
        cnn_top = eddl.getLayer(cnn, "top")
        cnn_out = eddl.getLayer(cnn, "cnn_out")
        print(f"cnn, top layer - visual features: {cnn_top.output.shape}")
        print(f"cnn, out layer - semantic features: {cnn_out.output.shape}")
        
        visual_dim = cnn_top.output.shape[1]
        semantic_dim = cnn_out.output.shape[1]
        vs = self.voc_size  # vocabulary size

        # INPUT: visual features
        cnn_top_in = eddl.Input([visual_dim], name="in_visual_features")
        visual_features = eddl.RandomUniform(eddl.Dense(cnn_top_in, cnn_top_in.output.shape[1], name="visual_features"), -0.05, 0.05)
        alpha_v = eddl.Softmax(eddl.Dense(eddl.Tanh(visual_features), visual_features.output.shape[1], name="dense_alpha_v"), name="alpha_v")  # missing sentence component
        v_att = eddl.Mult(alpha_v, visual_features)
        print(f"layer visual features: {visual_features.output.shape}")

        # INPUT: semantic features
        cnn_out_in = eddl.Input([semantic_dim], name="in_semantic_features")
        semantic_features = eddl.RandomUniform(eddl.Embedding(eddl.ReduceArgMax(cnn_out_in, [0]), cnn_out_in.output.shape[1], 1, self.conf["emb_size"], name="semantic_features"), -0.05, 0.05)
        alpha_s = eddl.Softmax(eddl.Dense(eddl.Tanh(semantic_features), self.conf["emb_size"], name="dense_alpha_s"), name="alpha_s")  # missing sentence component cnn_out.output.shape[1]
        s_att = eddl.Mult(alpha_s, semantic_features)
        print(f"layer semantic features: {semantic_features.output.shape}")

        # co-attention
        features = eddl.Concat([v_att, s_att], name="co_att_in")
        context = eddl.RandomUniform(eddl.Dense(features, self.conf["emb_size"], name="co_attention"), -0.1, 0.1)
        print(f"layer coattention: {context.output.shape}")

        # lstm
        word_in = eddl.Input([vs])
        to_lstm = eddl.ReduceArgMax(word_in, [0])
        to_lstm = eddl.RandomUniform(eddl.Embedding(to_lstm, vs, 1, self.conf["emb_size"], mask_zeros=True, name="word_embs"), -0.05, +0.05)
        to_lstm = eddl.Concat([to_lstm, context])
        lstm = eddl.LSTM(to_lstm, self.conf["lstm_size"], mask_zeros=True, bidirectional=False, name="lstm_cell")

        # *** *** *** *** *** I M P O R T A N T
        if for_training:
            print("CREATING RECURRENT MODEL FOR TRAINING (eddl.setDecoder(word_in))")
            eddl.setDecoder(word_in)
        else:
            print("CREATING RECURRENT MODEL FOR TESTING (lstm.isrecurrent = False)")
            lstm.isrecurrent = False
        
            
        out_lstm = eddl.Softmax(eddl.Dense(lstm, vs, name="out_dense"), name="rnn_out")
        print(f"layer lstm: {out_lstm.output.shape}")

        # model
        rnn = eddl.Model([cnn_top_in, cnn_out_in], [out_lstm])
        eddl.summary(rnn)
        return rnn
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

    def build_model(self, mdl=None):
        do_init_weights = (mdl is None)  # init weights only if mdl has not been read from file
        if mdl is None:
            rnn = self.create_model()
        
        optimizer = self.get_optimizer()
        eddl.build(rnn, optimizer, ["softmax_cross_entropy"], ["accuracy"], self.comp_serv(), init_weights=do_init_weights)
        
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
        n_epochs = self.conf.n_epochs
        batch_ids = range(len(ds))

        if self.conf["dev"]:
            batch_ids = [0, len(ds)-1]  # first and last batch only
            n_epochs = 1
        #<
        
        # self.run["params/activation"] = self.activation_name
        start_train_t = time.perf_counter()

        for ei in range(n_epochs):
            print(f"epoch {ei+1}/{n_epochs}")
            ds.shuffle()

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
                    self.run["train/batch/loss"].log(loss)
                    self.run["train/batch/acc"].log(acc)
            #< batch
            epoch_end = time.perf_counter()
            print(f"epoch completed in {H.precisedelta(epoch_end-t1)}")

            epoch_loss = epoch_loss / len(batch_ids)
            epoch_acc = epoch_acc / len(batch_ids)
            self.run["training/epoch/loss"].log(epoch_loss)
            self.run["training/epoch/acc"].log(epoch_acc)
            self.run["time/training/epoch"].log(epoch_end-t1)
            
            

            if (ei+1) % self.conf["check_val_every"] != 0:
                continue

            val_start = time.perf_counter()
            valid_loss, valid_acc = self.validation()
            ds.set_stage("train")
            val_end = time.perf_counter()
            self.run["validation/epoch/loss"].log(valid_loss, step=ei)
            self.run["validation/epoch/acc"].log(valid_acc, step=ei)
          
            # times
            self.run["time/validation/epoch"].log(val_end-val_start)
            self.run["time/train+val/epoch"].log(val_end - t1, step=ei)
        #< epoch
        end_train_t = time.perf_counter()
        print(f"rnn training complete: {H.precisedelta(end_train_t - start_train_t)}")
    #<

    def validation(self):
        print("*** VALIDATION ***")
        return 0, 0
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
        # NOT REALLY IMPLEMENTED
        ds = self.ds
        ds.set_stage("test")
        batch_ids = list(range(len(ds)))
        print(f"testing recurrent modules, test batches: {len(batch_ids)}")
        
        cnn = self.cnn
        rnn = self.rnn  # trained, loaded from file

        test_rnn = self.create_model(for_training=False)        
        eddl.build(test_rnn, 
                eddl.adam(0.01), ["softmax_cross_entropy"], ["accuracy"],  # not relevant
                self.comp_serv(), init_weights=False)
        eddl.summary(test_rnn)
        print("test module built")
        print("not implemented.")
    #<

    def get_network(self):
        return self.rnn
    #<

    def save(self, filename=None):
        filename = filename or self.conf.out_fn
        eddl.save_net_to_onnx_file(self.get_network(), filename)
        print(f"model saved, location: {filename}")
        return filename
    #<

    def build_model_prediction(self):
        args = self.conf
        cnn_top = eddl.getLayer(self.cnn, "top")
        cnn_out = eddl.getLayer(self.cnn, "cnn_out")
        visual_dim = cnn_top.output.shape[1]
        semantic_dim = cnn_out.output.shape[1]

        cnn_top_in = eddl.Input([visual_dim], name="in_visual_features")
        visual_features = eddl.Dense(cnn_top_in, cnn_top_in.output.shape[1], name="visual_features")
        alpha_v = eddl.Softmax(eddl.Dense(eddl.Tanh(visual_features), visual_features.output.shape[1], name="dense_alpha_v"), name="alpha_v")  # missing sentence component
        v_att = eddl.Mult(alpha_v, visual_features)

        cnn_out_in = eddl.Input([semantic_dim], name="in_semantic_features")
        semantic_features = eddl.Embedding(eddl.ReduceArgMax(cnn_out_in, [0]), cnn_out_in.output.shape[1], 1, self.conf["emb_size"], name="semantic_features")
        alpha_s = eddl.Softmax(eddl.Dense(eddl.Tanh(semantic_features), self.conf["emb_size"], name="dense_alpha_s"), name="alpha_s")  # missing sentence component cnn_out.output.shape[1]
        s_att = eddl.Mult(alpha_s, semantic_features)

        features = eddl.Concat([v_att, s_att], name="co_att_in")
        context = eddl.Dense(features, self.conf["emb_size"], name="co_attention")

        lstm_in = eddl.Input([self.voc_size])
        lstate = eddl.States([2, args.lstm_size])
        
        to_lstm = eddl.ReduceArgMax(lstm_in, [0])  # word index
        to_lstm = eddl.Embedding(to_lstm, self.voc_size, 1, args.emb_size, name="word_embs")
        to_lstm = eddl.Concat([to_lstm, context])
        lstm = eddl.LSTM([to_lstm, lstate], args.lstm_size, True, name="lstm_cell")
        lstm.isrecurrent = False
        #TODO: rename output for argmax insert axis
        out_lstm = eddl.ReduceArgMax(eddl.Softmax(eddl.Dense(lstm, self.voc_size, name="out_dense"), name="rnn_out"))
        
        # *** model
        model = eddl.Model([cnn_top_in, cnn_out_in, lstm_in, lstate], [out_lstm])
        eddl.build(model, eddl.adam(), ["mse"], ["accuracy"], eddl.CS_CPU(mem="full_mem"))
        
        eddl.summary(model)
        print("model for predictions built")

        # copy parameters from the trained recurrent network
        eddl.copyParam(eddl.getLayer(self.rnn, "visual_features"), # ok
                        eddl.getLayer(model, "visual_features")) # ok
        eddl.copyParam(eddl.getLayer(self.rnn, "dense_alpha_v"), # ok
                        eddl.getLayer(model, "dense_alpha_v")) # ok
        eddl.copyParam(eddl.getLayer(self.rnn, "semantic_features"), # ok
                        eddl.getLayer(model, "semantic_features")) # ok
        eddl.copyParam(eddl.getLayer(self.rnn, "dense_alpha_s"), # ok
                        eddl.getLayer(model, "dense_alpha_s")) # ok
        eddl.copyParam(eddl.getLayer(self.rnn, "co_attention"), # ok
                        eddl.getLayer(model, "co_attention")) # ok
        
        eddl.copyParam(eddl.getLayer(self.rnn, "lstm_cell"), # ok
                        eddl.getLayer(model, "lstm_cell")) # ok
        eddl.copyParam(eddl.getLayer(self.rnn, "out_dense"),
                        eddl.getLayer(model, "out_dense"))
        eddl.copyParam(eddl.getLayer(self.rnn, "word_embs"), # ok
                        eddl.getLayer(model, "word_embs")) # ok
        print("parameters copied from recurrent to non-recurrent model")
        return model
    #<

    # uses a non-recurrent model for the predictions
    def predict(self):
        #>
        cnn = self.cnn
        eddl.toCPU(cnn) 
        rnn = self.build_model_prediction()
        eddl.toCPU(rnn)
        #<

        #>
        # batch size, can be set to 1 for clarity
        # print(f"1: {len(ds)}")
        # ds.batch_size = 1
        # print(f"2: {len(ds)}")
        ds = self.ds
        ds.set_stage("test")
        vocab = self.ds.vocab
        #< 

        #> connection cnn -> rnn
        image_in = eddl.getLayer(cnn, "input")
        cnn_out = eddl.getLayer(cnn, "cnn_out")
        cnn_top = eddl.getLayer(cnn, "top")
        #<

        #> lstm and output layers
        lstm = eddl.getLayer(rnn, "lstm_cell")
        rnn_out = eddl.getLayer(rnn, "rnn_out")
        #<

        dev = self.conf.dev
        for i in range(len(ds)):
            images, _, texts = ds[i]
            bs = images.shape[0] # TODO: bs is constant across batches    
            
            #> cnn forward
            X = Tensor(images)
            eddl.forward(cnn, [X])
            cnn_semantic = eddl.getOutput(cnn_out)
            cnn_visual = eddl.getOutput(cnn_top)
            #<
            if dev: 
                print(f"batch, images: {X.shape}")
                print(f"\t- output. semantic: {cnn_semantic.shape}, visual: {cnn_visual.shape}")
            
            #> target texts are used for convenience to select the BOS token
            Y = Tensor(texts[:, 0])
            Y.reshape_( [bs, 1] )
            if dev:
                print(f"tensor texts, shape: {Y.shape}")
            Y = Tensor.onehot(Y, self.voc_size)
            if dev:
                print(f"1-hot tensor, shape {Y.shape}")
                print(f'should see 1 at the second position: {Y.select([":5", "0", ":4"]).getdata()}')
            
            # lstm cell state
            state = Tensor.zeros([bs, 2, self.conf["lstm_size"]])

            token = Y  # BOS
            for j in range(1, texts.shape[1]):
                eddl.forward(rnn, [cnn_visual, cnn_semantic, token, state])
                states = eddl.getStates(lstm)
                if dev:
                    print(f"|states|= {len(states)}")
                    print(f"states[0]: {states[0].shape}")
                    print(f"|states[1]: {states[1].shape}")
                # save the state for the next token
                for si in range(len(states)):
                    states[si].reshape_([ states[si].shape[0], 1, states[si].shape[1] ])
                    state.set_select( [":", str(si), ":"] , states[si] )
                
                out_token = eddl.getOutput(rnn_out)

                print(f"rnn output: {out_token.shape}")

        #< for over batches
    #< predict
#< class