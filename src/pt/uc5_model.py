
# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pickle
from posixpath import join

import torch
import torchvision.models as models
import torch.onnx
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

class Uc5Model(nn.Module):
    def __init__(self, conf, device="cpu"):
        super(Uc5Model, self).__init__()
        self.conf = conf
        self.vocab_size = self.conf["vocab_size"]
        self.cnn = base_modules.CNN(self.conf)
        # multi-label classifiers, visual to tags
        self.mlc = base_modules.MultiLabelTags(self.cnn.n_out_features, self.conf)
        # coattention, merging cnn (visual), mlc (tags) and sentence module (lstm)
        self.coatt = base_modules.CoAttention(self.cnn.n_out_features, self.conf)
        # sentence module
        self.sentence_lstm = base_modules.SentenceModule(self.conf)
        # word module
        self.word_lstm =  base_modules.WordModule(self.conf)
    #<

    # forward method is calld by training_, validation_, and test_step
    def forward(self, images, labels, sentences, probs, DEBUG=False):
        if DEBUG:
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
        if DEBUG:
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
        
        if DEBUG:
            print("/// SEMANTIC FEATURES (MLC):")
            print(f"///\t tags: {tags.shape}")
            print(f"///\t embeddings: {visual_embs.shape}")
        #< section: semantic features, multi-label classifier end
            
        #> section: coattention and sentence module
        previous_hidden_state = torch.zeros( (images.shape[0], 1, self.conf["lstm_sent_h_size"]), requires_grad=False, device=tags.device)  # TODO: , requires_grad=False)
        sent_lstm_states = None
        # sentences contain indexes of words in the voabulary
        sentences = sentences.long()
        n_sentences = sentences.shape[1] # sentences has shape (batch, n sents, n words)
        n_words = sentences.shape[2]
        if DEBUG:
            print("batch, n_sentences", n_sentences)
        
        # out_prob_stop, out_generated_words contains the generate stop probs and words
        #   for the current batch (used in training_step to compute loss)
        # remember: training_step expects tensors
        # pre allocate tensors
        out_prob_stop = torch.zeros( [images.shape[0], n_sentences, 2], device=sentences.device)
        out_generated_words = torch.zeros( [images.shape[0], n_sentences, n_words-1, self.vocab_size], device=sentences.device)

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
        if DEBUG:
            print_shapes(output_values, header="UC5MODEL.FORWARD: RETURNING THESE TENSORS")
        return tags, out_prob_stop, out_generated_words
    #< forward

    def predict(self, images, DEBUG=False):  # images is batch x 3 x 224 x 224 (if one, batch must be 1)
        import torch.nn.functional as F
        bs = images.shape[0]
        bos = Vocabulary.BOS_I
        eos = Vocabulary.EOS_I
        vs = self. vocab_size
        max_sentences = self.conf["n_sentences"]
        max_tokens = self.conf["n_tokens"]

        conv_feat, avg_feat, cnn_est = self.cnn.forward(images)
        tags, visual_embs = self.mlc.forward(avg_feat)
        previous_hidden_state = torch.zeros( (images.shape[0], self.conf["lstm_sent_n_layers"], self.conf["lstm_sent_h_size"]), device=images.device, requires_grad=False)  # TODO: , requires_grad=False)
        sent_lstm_states = None
        coatt, alpha_v, alpha_a = self.coatt.forward(avg_feat, visual_embs, previous_hidden_state)
                
        sent_idx = 0
        tok_idx = 0
        output = torch.zeros( (bs, max_sentences, max_tokens), device=images.device, requires_grad=False)
        for i in range(bs):
            stop_generated = False
            while (sent_idx < max_sentences) and (not stop_generated):
                previous_hidden_state, sent_lstm_states, topic, prob_stop = self.sentence_lstm.forward(coatt, previous_hidden_state, sent_lstm_states)            
                stop_generated = torch.argmax(F.softmax(prob_stop, dim=1)).item() == 0  # see uc_5_dataset,  get_item_... methods
                tok_idx = 0
                eos_generated = False
                toks = torch.zeros( (1, 1), device=images.device).long()
                toks[0,0] = bos
                while (tok_idx < max_tokens) and (not eos_generated):                
                    t = topic[i, :].unsqueeze(0)
                    next_tok = self.word_lstm.forward(t, toks)
                    next_tok = torch.argmax(F.softmax(next_tok, dim=1), dim=1)
                    if next_tok.item() == eos:
                        eos_generated = True
                    toks = torch.cat( (toks, next_tok.unsqueeze(0)), dim = 1)
                    tok_idx += 1
                #< while over tokens
               
                output[i, sent_idx, :toks.shape[1]] = toks.squeeze()
                # print(f"sentence {sent_idx+1}, generate sequence, length: {toks.shape[1]}, eos generated? {eos_generated}")
                sent_idx += 1
            # while over sentences
            if DEBUG:
                print(f"stop generated? {stop_generated}")
        # for over batch
        return output
    #< predict
#< class Uc5Model

def compute_loss(labels, sentences, probs, # batch 
                tags, words, prob_stop, # forward output
                losses, # loss functions
                DEBUG=False):
    # here we receive the output of the forward step
    #region
    args = locals()
    if DEBUG:
        print_shapes(args, header="COMPUTE LOSS")

    # region: compute tag loss
    tag_loss = losses["tag"](tags, labels).sum()
    
    # region: stop loss
    stop_loss = 0
    probs = probs[:, 1:]
    
    # notice: estimated stop probabilities and input probabilities have different shapes
    #  the estimated probabilities have 1 element less: they do not contain an element
    #  for the first element of the sequence <begin of sentence>
    for i in range(sentences.shape[1]):
        stop_loss += losses["prob"](prob_stop[:, i, :], probs[:, i]).sum()
    
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
        tmp_loss = losses["word"](sen , sentences[:, i, :])
        word_loss += (tmp_loss * mask).sum()
    # for i over sentences
    # (est) words shape: [batch, sentence, words, embeddings]
    # (target) sentences: [batch, sentence, words]  # indexes in the vocabulary

    # region: merge the three losses
    if DEBUG:
        print(f"tag_loss: {tag_loss:.2f}")
        print(f"stop_loss: {type(stop_loss)}, {stop_loss:.2f}")
        print(f"word_loss: {word_loss:.2f}")

    # TODO: evaluate whether or not different weights can be assigned to the three losses
    loss = tag_loss + stop_loss + word_loss
    return loss / labels.shape[0]
#< compute_loss

def compute_bleu(predictions, text):
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