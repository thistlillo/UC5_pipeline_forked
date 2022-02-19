import humanize as H
import numpy as np
import pandas as pd
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor, DEV_CPU, DEV_GPU
from posixpath import join
from tqdm import tqdm
from text.vocabulary import Vocabulary
import time

from eurodll.uc5_dataset import Uc5Dataset
from eurodll.recurrent_models import recurrent_lstm_model, nonrecurrent_lstm_model
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
            n_epochs = 20
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
        eddl.save_net_to_onnx_file(rnn, self.conf.out_fn)  # cannot be load, error LDense only works over 2D tensors (LDense)
        bin_out = self.conf.out_fn.replace(".onnx", ".bin")
        eddl.save(rnn, bin_out)
        print(f"trained recurrent model saved at {self.conf.out_fn}")
        print(f"\t  binary model saved at {bin_out}")
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

        emb_size_visual = eddl.getLayer(self.rnn, "semantic_features").output.shape[1]
        emb_size_semantic = emb_size_visual
        emb_size_context = emb_size_visual
        voc_size = self.ds.vocab.n_words
        word_emb_size = emb_size_visual
        lstm_size = eddl.getLayer(self.rnn, "lstm_cell").output.shape[1]
        print(f"emb_size_visual: {emb_size_visual}")
        print(f"vocab size: {voc_size}")
        print(f"lstm size: {lstm_size}")

        cnn_top_in = eddl.Input([visual_dim], name="in_visual_features")
        visual_features = eddl.Dense(cnn_top_in, cnn_top_in.output.shape[1], name="visual_features")
        alpha_v = eddl.Softmax(eddl.Dense(eddl.Tanh(visual_features), visual_features.output.shape[1], name="dense_alpha_v"), name="alpha_v")  # missing sentence component
        v_att = eddl.Mult(alpha_v, visual_features)

        cnn_out_in = eddl.Input([semantic_dim], name="in_semantic_features")
        semantic_features = eddl.Embedding(eddl.ReduceArgMax(cnn_out_in, [0]), cnn_out_in.output.shape[1], 1, self.conf["emb_size"], name="semantic_features")
        alpha_s = eddl.Softmax(
                    eddl.Dense(eddl.Tanh(semantic_features), self.conf["emb_size"], name="dense_alpha_s"), 
                name="alpha_s")  # missing sentence component cnn_out.output.shape[1]
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
       
        out_lstm = eddl.Softmax(
                    eddl.Dense(lstm, self.voc_size, name="out_dense"), 
                  name="rnn_out")
        
        # *** model
        model = eddl.Model([cnn_top_in, cnn_out_in, lstm_in, lstate], [out_lstm])
        eddl.build(model, eddl.adam(), ["mse"], ["accuracy"], self.comp_serv())
        eddl.summary(model)
        print("model for predictions built")

        #> copy parameters from the trained recurrent network
        layers_to_copy = [
            "visual_features", "dense_alpha_v",
            "semantic_features", "dense_alpha_s", "co_attention",
            "lstm_cell", "out_dense", "word_embs"
        ]
        for l in layers_to_copy:
            eddl.copyParam(eddl.getLayer(self.rnn, l), eddl.getLayer(model, l))
        #<

        # IMPORTANT
        # NOTICE: saving model with non-recurrent LSTM cell
        # eddl.save_net_to_onnx_file(model, "a2.onnx")  # error:  The layer State1 has no OpType in Onnx.
        pred_out_fn = self.conf.out_fn.replace(".onnx", "_pred.bin")
        eddl.save(model, pred_out_fn)
        print(f"binary non-recurrent model (for predictions) saved at: {pred_out_fn}")
        # if the model is saved in onnx, there is the same error as in the recurrent model when loaded: 
        #       LDense only works over 2D tensors (LDense)
        return model
    #<

    # uses a non-recurrent model for the predictions
    def predict(self):
        rnn = self.build_model(for_predictions=True)
        
        #> copy parameters from the trained recurrent network (see recurrent_models.py for layer names)
        layers_to_copy = [
            "visual_features", "dense_alpha_v",
            "semantic_features", "dense_alpha_s", "co_attention",
            "lstm_cell", "out_dense", "word_embs"
        ]
        for l in layers_to_copy:
            eddl.copyParam(eddl.getLayer(self.rnn, l), eddl.getLayer(rnn, l))
        #<
        eddl.set_mode(rnn, mode=0)
        #> test on CPU (ISSUE related to eddl.getStates(.) when running on GPU)
        test_device = DEV_CPU
        cnn = self.cnn
        eddl.toCPU(cnn) 
        eddl.toCPU(rnn)
        #<

        #>
        # batch size, can be set to 1 for clarity
        # print(f"1: {len(ds)}")
        # ds.batch_size = 1
        # print(f"2: {len(ds)}")
        ds = self.ds
        ds.last_batch = "drop"
        ds.set_stage("test")
        vocab = self.ds.vocab
        bs = ds.batch_size
        print(f"text generation, using batches of size {bs}")
        #< 

        #> connection cnn -> rnn
        image_in = eddl.getLayer(cnn, "input")
        cnn_out = eddl.getLayer(cnn, "cnn_out")
        cnn_top = eddl.getLayer(cnn, "top")
        #<

        #> lstm and output layers
        lstm = eddl.getLayer(rnn, "lstm_cell")
        rnn_out = eddl.getLayer(rnn, "rnn_out")  # softmax
        #<

        dev = self.conf.dev
        n_tokens = self.conf.n_tokens
        last_layer = eddl.getLayer(rnn, "rnn_out")  # rnn.layers[-1]
        
        generated_tokens = np.zeros( (len(ds) * bs, n_tokens), dtype=int)
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
                     
            # lstm cell states
            state_t = Tensor.zeros([bs, 2, self.conf["lstm_size"]], dev=test_device)
            print(f"states tensor: {state_t.shape}")
            # token: input to lstm cell
            token = Tensor.zeros([bs, self.voc_size], dev=test_device)
            print(f"0 token: {token.shape}")
            for j in range(0, n_tokens):
                print(f" *** token {j}/{n_tokens} ***")
                if dev:
                    print(f"cnn_vidual: {cnn_visual.shape}")
                    print(f"cnn_semant: {cnn_semantic.shape}")
                    print(f"token: {token.shape}")
                    print(f"state_t: {state_t.shape}")

                eddl.forward(rnn, [cnn_visual, cnn_semantic, token, state_t])
                print('forward')
                states = eddl.getStates(lstm)  # states = 
                # save the state for the next token
                for si in range(len(states)):
                    states[si].reshape_([ states[si].shape[0], 1, states[si].shape[1] ])
                    state_t.set_select( [":", str(si), ":"] , states[si] )
                
                out_soft = eddl.getOutput(last_layer)
                a = np.array(out_soft)
                print(a[:5, :20])
                # pass control to numpy for argmax
                wis = np.argmax(out_soft, axis=-1)
                print(wis.shape)
                print(f"next_token {wis[0]}")
                generated_tokens[i*bs:i*bs+wis.shape[0], j] = wis
                word_index = Tensor.fromarray(wis.astype(float), dev=test_device)
                word_index.reshape_([bs, 1])  # add dimension for one-hot encoding
                
                token = Tensor.onehot(word_index, self.voc_size)

                token.reshape_([bs, self.voc_size])  # remove singleton dim          
            #< for n_tokens
            
            for i in range(bs):
                print(f"*** {i} ***")
                print(generated_tokens[i, :])
        #< for over batches
    #< predict
#< class