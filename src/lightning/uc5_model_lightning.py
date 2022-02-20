# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import humanize as H
import numpy as np
import pickle
from posixpath import join
import time
from tqdm import tqdm


import torch
import torchvision.models as models
import torch.onnx
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nltk.translate.bleu_score import sentence_bleu

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import NeptuneLogger

import pt.uc5_base_models as base_modules
from text.vocabulary import Vocabulary

# util function for printing shapes - for debugging
def print_shapes(d, header=None):
    header = header or f"print_shapes ({len(d)}):"
    print(f"** {header}:")
    for k, v in d.items():
        print(f"{k}, {type(v)}: ", end="")
        if type(v) is list:
            print(f"{len(v)}")
        elif (type(v) is torch.Tensor) or (type(v) is np.array):
            print(f"{v.shape}")
        else:
            print(" don't know")
    print("---")
#< print_shapes


class Uc5Model(LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.cnn, self.mlc, self.coatt, self.sentence_lstm, self.word_lstm = self._create_modules()
        self.tag_loss, self.stop_loss, self.word_loss = self._configure_losses()
        self.DEBUG = self.conf["debug"]
        self.vocab_size = self.conf["vocab_size"]

        if self.vocab_size == 0:
            print(f"vocabulary size not provided, reading vocabulary from {self.conf['exp_fld']}")
            with open( join(self.conf["exp_fld"], "vocab.pickle"), "rb") as fin:
                self.vocab = pickle.load(fin)
                self.vocab_size = self.vocab.n_words

        self.train_start_t, self.valid_start_t = None, None

    #< init

    def _create_modules(self):
        # feature extraction
        cnn = base_modules.CNN(self.conf)
        # multi-label classifiers, visual to tags
        mlc = base_modules.MultiLabelTags(cnn.n_out_features, self.conf)
        # coattention, merging cnn (visual), mlc (tags) and sentence module (lstm)
        coatt = base_modules.CoAttention(cnn.n_out_features, self.conf)
        # sentence module
        sentence_lstm = base_modules.SentenceModule(self.conf)
        # word module
        word_lstm =  base_modules.WordModule(self.conf)
        return cnn, mlc, coatt, sentence_lstm, word_lstm
    #< create_modules
        
    def _configure_losses(self):
        # one loss per module
        # cnn is frozen (pretrained)
        tag_loss_fn = nn.CrossEntropyLoss(reduction="none")
        stop_loss_fn = nn.CrossEntropyLoss(reduction='none')
        word_loss_fn = nn.CrossEntropyLoss(reduction='none')
        return tag_loss_fn, stop_loss_fn, word_loss_fn
    #< _configure_losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)  # TODO: learning rate from config
        # doc here:
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return {"optimizer": optimizer}
    #< configure_optiizers

    # forward method is calld by training_, validation_, and test_step
    def forward(self, images, labels, sentences, probs):
        if self.DEBUG:
            print("*** NEW BATCH")
            print(f"***\t images: {images.shape}")
            print(f"***\t labels: {labels.shape}")
            print(f"***\t sentences: {sentences.shape}")
            print(f"***\t continue proabilities: {probs.shape}")
        
        # the prob layer outputs two values for two classes: stop and do-not-stop
        #       probs here represent class indexes
        # the comparison between layer outputs and correct labels is done
        #    by CrossEntropyLoss automatically on class indexs (see PyTorch docs)
        probs = probs.long() # class indexes must be "integer"

        #> section: visual features, forward images
        conv_feat, avg_feat, cnn_est = self.cnn.forward(images)
        if self.DEBUG:
            print("--- VISUAL FEATURES:")
            print(f"---\t convolutional features: {conv_feat.shape}")
            print(f"---\t averaged features: {avg_feat.shape}")
            print(f"---\t CNN output (estimate): {cnn_est.shape}")
        #< section: visual features end

        #> section: semantic features, multi-label classifier
        # tags has size equal to the number of classes of the images
        #    its value MUST be specified via a command line argument
        # visual_embs are the k top tags. k is speicifed via a cmd line argument.
        tags, visual_embs = self.mlc.forward(avg_feat)
        
        if self.DEBUG:
            print("/// SEMANTIC FEATURES (MLC):")
            print(f"///\t tags: {tags.shape}")
            print(f"///\t embeddings: {visual_embs.shape}")
        #< section: semantic features, multi-label classifier end
            
        #> section: coattention and sentence module
        previous_hidden_state = torch.zeros( (images.shape[0], 1, self.conf["lstm_sent_h_size"]), device=self.device, requires_grad=False)  # TODO: , requires_grad=False)
        sent_lstm_states = None
        # sentences contain indexes of words in the voabulary
        sentences = sentences.long()
        n_sentences = sentences.shape[1] # sentences has shape (batch, n sents, n words)
        n_words = sentences.shape[2]
        if self.DEBUG:
            print("batch, n_sentences", n_sentences)
        
        # out_prob_stop, out_generated_words contains the generate stop probs and words
        #   for the current batch (used in training_step to compute loss)
        # remember: training_step expects tensors
        # pre allocate tensors
        out_prob_stop = torch.zeros( [images.shape[0], n_sentences, 2], device=self.device )
        out_generated_words = torch.zeros( [images.shape[0], n_sentences, n_words-1, self.vocab_size], device=self.device )
        
        # iterate over sentences: this step aims at learning when to stop
        for sent_idx in range(n_sentences):
            # print(f"** TRAINING ON SENTENCE {sent_idx}")
            coatt, alpha_v, alpha_a = self.coatt.forward(avg_feat, visual_embs, previous_hidden_state)
            # print(f"got co-attention context vector: {coatt.shape}")
            
            # topic generated for the current sentence
            # prob_stop has size n_sentences
            previous_hidden_state, sent_lstm_states, topic, prob_stop = self.sentence_lstm.forward(coatt, previous_hidden_state, sent_lstm_states)            
            # prob_stop will be stored after chaning the type
            # print("got stop prob for current sentence with shape: ", prob_stop.shape)
            
            # iterate over words in the current sentence
            # TODO: first token is bos, out_generated_words[0] corresponds to the word generated after
            # TODO:         seeing the bos token
            for wi in range(1, n_words):  # sentences batch, n sents, n words
                # learning strategy here is normally called "forced teacher": 
                #   at time t+1 the lstm does not receive its input at time t, but the target word
                gen_text = self.word_lstm.forward(topic, sentences[:, sent_idx, :wi])
                
                if sent_idx == 0 and wi ==1:
                    out_prob_stop = out_prob_stop.type_as(prob_stop)  # useful when changing precision
                    out_generated_words = out_generated_words.type_as(gen_text)
                # print("generated text, shape",   gen_text.shape)
                
                out_generated_words[:, sent_idx, wi-1, :] = gen_text
                # >0 => do not consider padding
                
                # nn.CrossEntropyLoss(size_average=False, reduce=False)
                # batch_word_loss += (self.ce_criterion(gen_text, sentences[:, sent_idx, wi]))
            #< for over words
            out_prob_stop[:, sent_idx, :] = prob_stop[:,0,:]  # prob_stop contains the output values of the SentenceModule
        #< for over sentences

        output_values = {"tags": tags, "prob_stop": out_prob_stop, "words": out_generated_words}
        if self.DEBUG:
            print_shapes(output_values, header="UC5MODEL.FORWARD: RETURNING THESE TENSORS")
        return output_values
    #< forward

    def compute_loss(self, batch, tags, prob_stop, words):
        # here we receive the output of the forward step
        images, labels, sentences, probs = batch
        #region
        args = locals()
        if self.DEBUG:
            print_shapes(args, header="COMPUTE LOSS")

        # region: compute tag loss
        tag_loss = self.tag_loss(tags, labels).sum()
        
        # region: stop loss
        stop_loss = 0
        probs = probs[:, 1:]
        # notice: estimated stop probabilities and input probabilities have different shapes
        #  the estimated probabilities have 1 element less: they do not contain an element
        #  for the first element of the sequence <begin of sentence>
        for i in range(sentences.shape[1]):
            stop_loss += self.stop_loss(prob_stop[:, i, :], probs[:, i]).sum() 
        
        # region: word loss
        # notice: generated word tensor DOES NOT contain a begin of sentence token
        word_loss = 0
        #* the first prediction we have in words (generated) is generated from bos
        #*   bos is thus excluded from the target "sentences" (notice 1: in the 3rd dim)
        sentences = sentences[:, :, 1:]
        for i in range(sentences.shape[1]):
            # exclude padded positions from loss computation
            mask = (sentences[:, i, :] > 0).float()
            sen = words[:, i, :, :]
            sen = torch.permute(sen, (0, 2, 1))  # permutation needed by CrossEntropyLOss
            tmp_loss = self.word_loss(sen , sentences[:, i, :])
            word_loss += (tmp_loss * mask).sum()
        # for i over sentences
        # (est) words shape: [batch, sentence, words, embeddings]
        # (target) sentences: [batch, sentence, words]  # indexes in the vocabulary

        # region: merge the three losses
        if self.DEBUG:
            print(f"tag_loss: {tag_loss:.2f}")
            print(f"stop_loss: {type(stop_loss)}, {stop_loss:.2f}")
            print(f"word_loss: {word_loss:.2f}")

        # TODO: evaluate whether or not different weights can be assigned to the three losses
        loss = tag_loss + stop_loss + word_loss
        return loss / images.shape[0]
    #< compute_loss

    #! the following methods are called for each batch
    def training_step(self, batch, batch_idx):
        # print(f"TRAINING STEP, batch {batch_idx}")
        images, labels, sentences, probs = batch
        
        res = self.forward(images, labels, sentences, probs)
        tags = res["tags"]
        prob_stop = res["prob_stop"]
        words = res["words"]
        if self.DEBUG:
            print_shapes({"tags": tags, "prob_stop": prob_stop, "words": words})
 
        loss = self.compute_loss(batch, tags, prob_stop, words)
        
        self.log("train/batch/loss", loss)
        return {"loss": loss}
    #< training_step    

    def validation_step(self, batch, batch_idx):
        images, labels, sentences, probs = batch
        res = self.forward(images, labels, sentences, probs)
        tags = res["tags"]
        prob_stop = res["prob_stop"]
        words = res["words"]
        #print("validation step, received")
        #print("tags",tags.shape)
        #print("prob_stop", prob_stop.shape)
        #print("words", words.shape)
        loss = self.compute_loss(batch, tags, prob_stop, words)
        self.log("val/batch/val_loss", loss)
        return {"val_loss": loss}
    #< validation_step


    def predict(self, images):  # images is batch x 3 x 224 x 224 (if one, batch must be 1)
        import torch.nn.functional as F
        print(f"TESTING ON {self.device}")
        bs = images.shape[0]
        vocab = self.vocab  # this could be None!
        vs = self.vocab_size
        bos = Vocabulary.BOS_I
        eos = Vocabulary.EOS_I

        conv_feat, avg_feat, cnn_est = self.cnn.forward(images)
        tags, visual_embs = self.mlc.forward(avg_feat)
        
        previous_hidden_state = torch.zeros( (images.shape[0], self.conf["lstm_sent_n_layers"], self.conf["lstm_sent_h_size"]), device=self.device, requires_grad=False)  # TODO: , requires_grad=False)
        sent_lstm_states = None

        coatt, alpha_v, alpha_a = self.coatt.forward(avg_feat, visual_embs, previous_hidden_state)
        
        max_sentences = self.conf["n_sentences"]
        max_tokens = self.conf["n_tokens"]

        sent_idx = 0
        tok_idx = 0
        output = torch.zeros( (bs, max_sentences, max_tokens), device=self.device)
        for i in range(bs):
            print(f"predict on batch index {i+1} of {bs}")
            stop_generated = False
            while (sent_idx < max_sentences) and (not stop_generated):
                previous_hidden_state, sent_lstm_states, topic, prob_stop = self.sentence_lstm.forward(coatt, previous_hidden_state, sent_lstm_states)            
                stop_generated = torch.argmax(F.softmax(prob_stop, dim=1)).item() == 0  # see uc_5_dataset,  get_item_... methods
                tok_idx = 0
                eos_generated = False
                toks = torch.zeros( (1, 1), device=self.device).long()
                toks[0,0] = bos
                while (tok_idx < max_tokens) and (not eos_generated):                
                    t = topic[i, :].unsqueeze(0)
                    print(f"current_token.shape:", toks.shape)
                    print(f"sentence_topic.shape: ", t.shape)
                    next_tok = self.word_lstm.forward(t, toks)
                    print(f"next_tok.shape: {next_tok.shape}")
                    next_tok = torch.argmax(F.softmax(next_tok, dim=1), dim=1)
                    if next_tok.item() == eos:
                        eos_generated = True
                    toks = torch.cat( (toks, next_tok.unsqueeze(0)), dim = 1)
                    print(f"next input will have shape: {toks.shape}")
                    print(next_tok)
                    tok_idx += 1
                #< while over tokens
                print(f"output shape: {output.shape}")
                print(f"toks shape: {toks.shape}")
                print(f"output[i, sent_idx, :] shape: {output[i, sent_idx, :].shape}")
                
                output[i, sent_idx, :toks.shape[1]] = toks.squeeze()
                print(f"sentence {sent_idx+1}, generate sequence, length: {toks.shape[1]}, eos generated? {eos_generated}")
                sent_idx += 1
            # while over sentences
            print(f"sentences generated, length , stop generated? {stop_generated}")
        # for over batch
        print(output[0, :, 0])

        return output
    #<

    def compute_bleu(self, predictions, text):
        def build_str(seqs_of_tokens):
            text = []
            for j in range(seqs_of_tokens.shape[0]):
                for k in range(seqs_of_tokens.shape[1]):
                    token = seqs_of_tokens[j,k]  # sentence j, token k
                    text.append(str(token))
                    if token == Vocabulary.EOS_I:
                        break
            return text
        bleu = 0
        for i in range(predictions.shape[0]):
            pred = build_str(predictions[i, :, :].squeeze())
            txt = build_str(text[i, :, :].squeeze())
            bleu += sentence_bleu([txt], pred)
        return bleu / predictions.shape[0]


    # in this step we generate sentences and measure B
    def test_step(self, batch, batch_idx):
        images, labels, sentences, probs = batch
        res = self.forward(images, labels, sentences, probs)
        tags, prob_stop, words = res["tags"], res["prob_stop"], res["words"]
        predictions = self.predict(images)
        bleu = self.compute_bleu(predictions, sentences)
        return {"bleu": torch.tensor(bleu, device=self.device, requires_grad=False)}
    #< test_step

    #! hooks
    def on_train_epoch_start(self):
        self.train_start_t = time.perf_counter()

    def on_train_epoch_end(self):
        t = time.perf_counter() - self.train_start_t
        self.log("time/epoch/train", t, prog_bar=True)
        

    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("train/epoch/avg_loss", avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val/epoch/avg_loss', avg_loss)
        return {'avg_val_loss': avg_loss}

    def test_epoch_end(self, outputs):
        bleu = torch.stack([x['bleu'] for x in outputs]).mean()
        self.log("test/bleu", bleu)
        return {'bleu': bleu}
#< Uc5Model