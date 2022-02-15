# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import fire
import humanize as H
import json

import numpy as np
import pandas as pd
import pickle
from posixpath import join
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pt.uc5_dataset import Uc5ImgDataset
import pt.utils as pt_utils
from pt.uc5_model import Uc5Model, compute_loss, compute_bleu

import text.reports as reports
import time

import neptune.new as neptune

def configure_logger(args):
    mode="async"

    if args.debug or args.dev:
        mode = "debug"

    if not args.remote_log:
        mode = "offline"

    #> section: neptune logger, remember to set environment variable
    run = neptune.init(
        project = "thistlillo/UC5-DeepHealth-PyTorch",
        name = "base_version",
        mode = mode
    )
    #< section: neptune logger, remember to set environment variable end
    return run

# https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)

# --------------------------------------------------
def pipeline(config):
    args = Bunch(**config)
    print("torch pipeline starting")

    #> logger
    run = configure_logger(args)
    run['parameters'] = args

    #> D A T A
    tsv = pd.read_csv(args.in_tsv, sep=reports.csv_sep, na_filter=False).set_index("filename", inplace=False, drop=False)
    run["dataset"].track_files(args.in_tsv)
    
    train_ids, val_ids, test_ids = pt_utils.load_data_split(args.exp_fld) if args.load_data_split else pt_utils.create_partitions(args, tsv)
    with open(join(args.exp_fld, "lab2index.json"), "r") as fin:
        label2index = json.load(fin)
        n_classes = len(label2index)

    #with open(join(args.exp_fld, "index2lab.json"), "r") as fin:
    #    index2label = json.load(fin)    
    
    vocab = None
    vocab_size = args.vocab_size
    if vocab_size == 0:
        with open( join(args.exp_fld, "vocab.pickle"), "rb") as fin:
            vocab = pickle.load(fin)
            vocab_size = vocab.n_words
    args["vocab_size"] = vocab_size
    
    print(f"full dataset, shape {tsv.shape}, |tags|= {n_classes}, |vocabulary|= {vocab_size}")
    print(f"|train|= {len(train_ids)}, |val|={len(val_ids)}, |test|= {len(test_ids)}")
    
    # log
    run["vocab_size"] = vocab_size
    run["train_size"] = len(train_ids)
    run["valid_size"] = len(val_ids)
    run["test_size"] = len(test_ids)
    #<

    bs = args.batch_size
    n_epochs = args.n_epochs
    check_val_every = args.check_val_every

    #> D E V: limit data and number of epochs when dev flag is set
    if args.dev:
        print(f"*** dev set, limiting data and epochs")
        train_ids, val_ids, test_ids = train_ids[: int(1.5 * bs) ], val_ids[:bs], test_ids[:5]
        n_epochs = 2
        check_val_every = 1
    #<

    #> ? training, validation, test
    # pandas dataframes
    train_df = tsv[tsv.filename.isin(train_ids)].copy()
    val_df = tsv[tsv.filename.isin(val_ids)].copy()
    test_df = tsv[tsv.filename.isin(test_ids)].copy()
    # torch datasets
    train_ds = Uc5ImgDataset(train_df, n_classes, args, version=None)  # dataset code shared with EDDL, version is important
    val_ds = Uc5ImgDataset(val_df, n_classes, args, version=None)
    test_ds = Uc5ImgDataset(test_df, n_classes, args, version=None)
    # torch dataloaders
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.loader_threads)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)
    #<

    #> device
    device = args.device
    if device == "gpu":
        gpu_id = args.gpu_id or np.random.randint(torch.cuda.device_count())
        device = f"cuda:{gpu_id}"
    run["device"] = device
    device = torch.device(device)
    print(f"*** using device: {device}")
    #<

    #> M O D E L
    model = Uc5Model(args).to(device)
    
    #> loss
    tag_loss_fn = nn.CrossEntropyLoss(reduction="none")
    stop_loss_fn = nn.CrossEntropyLoss(reduction='none')
    word_loss_fn = nn.CrossEntropyLoss(reduction='none')
    losses = {"tag":tag_loss_fn, "prob": stop_loss_fn, "word": word_loss_fn} # will be passed to compute loss
    #<

    #> optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #<
 
    #> T R A I N I N G 
    best_val_loss = 1_000_000
    timings = []
    start_time_t = time.perf_counter()
    print(f"training is starting: {n_epochs} epochs, {bs} batch size, {len(train_dl)} batches, validation every {check_val_every} epochs")
    
    for ei in range(n_epochs):
        loss_e = 0
        
        start_time_e = time.perf_counter()
        model.train()
        for bi, (images, labels, text, probs) in enumerate(tqdm(train_dl)):
            images = images.to(device)
            labels = labels.to(device)
            text = text.to(device)
            probs = probs.to(device)

            optimizer.zero_grad()
            out_labels, out_probs, out_words = model.forward(images, labels, text, probs)
            
            loss = compute_loss(labels, text, probs, out_labels, out_words, out_probs, losses, DEBUG=args.debug)
            loss.backward()
            optimizer.step()
            run["train/batch/loss"].log(loss.item(), step=ei *len(train_dl) + bi)
            loss_e += loss.item()
        #< for over training batches        
        loss_e = loss_e / len(train_dl)
        run["train/epoch/loss"].log(loss_e)

        #> V A L I D A T I O N
        if ei % check_val_every == 0:
            model.eval()
            loss_v = 0
            for bi, (images, labels, text, probs) in enumerate(val_dl):
                images = images.to(device)
                labels = labels.to(device)
                text = text.to(device)
                probs = probs.to(device)
                out_labels, out_probs, out_words = model.forward(images, labels, text, probs)
                loss_v += compute_loss(labels, text, probs, out_labels, out_words, out_probs, losses, DEBUG=args.debug).item()
            #< for over val batches
            
            loss_v = loss_v/len(val_dl)
            if loss_v < best_val_loss:
                best_val_loss = loss_v
                path = join(args.exp_fld, f"checkpoint_e{ei}_l{loss_v:.02f}.pt")
                torch.save(model.state_dict(), path)
                print(f"checkpoint saved at { path } for best validation loss {loss_v:.3f}")

            print(f"epoch {ei+1}/{n_epochs}, validation loss: {loss_v:.3f}")
            run["valid/epoch/loss"].log(loss_v, step=ei)
        #< if do validation

        timings.append(time.perf_counter() - start_time_e)
        
        #> log every 20 epochs
        if ei % 19 == 0:
            print(f"epoch {ei+1}/{n_epochs}, loss: {loss_e:.3f}")
            avg_time = sum(timings)/len(timings)
            print(f"{H.precisedelta(avg_time)} per epoch, remaining { H.precisedelta(avg_time * (n_epochs-ei-1)) }")

        # evaluate BLEU on everything every 500 epochs
        if (ei+1) % 500 == 1:
            print("evaluating bleu on everything")
            model = model.to("cpu")
            bleu1 = evaluate_bleu(model, train_dl)
            bleu2 = evaluate_bleu(model, val_dl)
            bleu3 = evaluate_bleu(model, test_dl)
            run["train/epoch/bleu"].log(bleu1, step=ei)
            run["val/epoch/bleu"].log(bleu2, step=ei)
            run["test/epoch/bleu"].log(bleu3, step=ei)
            print("BLEU evaluated during training:")
            print(f" - training: {bleu1:.3f}")
            print(f" - validation: {bleu2:.3f}")
            print(f" - test: {bleu3:.3f}")
            model = model.to(device)
        #< 
    #< for over epochs
    #< training

    #>
    torch.save(model.state_dict(), args.out_fn)
    print(f"training complete, model saved at: {args.out_fn}")

    #> T E S T / predict
    start_time_p = time.perf_counter()
    print("running predict(), moving model to cpu")
    model = model.to("cpu")
    
    bleu = evaluate_bleu(model, test_dl)
    print(f"predictions, BLEU score: {bleu:.3f}")
    print(f"predictions, time elpased: {H.precisedelta(time.perf_counter() - start_time_p)}")
    run["test/bleu"] = bleu
    #< test
    
    print(f"done, total time elapsed: {H.precisedelta(time.perf_counter() - start_time_t)}")
#< pipeline

def evaluate_bleu(model, dl):
    bleu = 0
    with torch.no_grad():
        for bi, (images, _, text, _) in enumerate(dl):
            predictions = model.predict(images)
            bleu += compute_bleu(predictions, text)
            bleu = bleu / len(dl)
    return bleu



# --------------------------------------------------
def main(in_tsv,
         exp_fld,
         img_fld,
         #gpu_id=[0],
         vocab_size = 0,          # when 0, uc5_model will read the vocab.pickle in exp_fld
         only_images = False,     # for training a CNN without text -- NOT USED IN PYTORCH
         load_data_split = True,  # instead of generating a split on the fly, load ids from files previously built
         img_size = 224,          # input image size
         out_fn = "uc5model_pt_base",  # filename for the best model
         train_p = 0.7,           # train_p, valid_p: training, validation percentage
         valid_p = 0.1,     
         seed = 1,                # random seed
         shuffle_seed = 2,        # seed for shuffling examples
         text_column = "text",    # name of the column containing the text. Encoded text expected at enc_{text_column}
         term_column = "auto_term", # name fo the column containing the tags for the multi-label classified
         batch_size = 32,         # batch size
         last_batch = "random",   # how to fill the last batch (EDDL requires the batches to be all of the same size)
         n_epochs = 50,           # training epochs
         n_sentences = 5,         # number of sentences to generate
         n_tokens = 10,           # maximum number of tokens per sentence
         lr = 1e-3,             # learning rate and momentum
         momentum = 0.9,
         lstm_size = 512,         # lstm_size (for all of the lstm in the model)
         emb_size = 512,          # embedding size for all the embeddings used in the model
         init_linear = 0.1,       # weights will be initialize with torch.nn.init.uniform_(t, -init_linear, init_linear)
         top_k_tags = 5,          # number of semantics tags used for sentence generation
         # ATTENTION HERE (n_tags)!
         n_tags = 94,             # number of tags/classes of the images. This number must be manually passed. It depends on the preprocessing steps.
         tag_emb_size = 512,      # dimension of tag emebddings. Currently all the embeddings dimension MUST have the same value (512).
         init_embs = 0.5,         # embedding matrixes wille be initialized with torch.nn.init.uniform_(t, -init_embs, init_embs)
         attn_emb_size = 512,     # dimension of attention embeddings. Currently all the embeddings dimension MUST have the same value (512).
         word_emb_size = 512,
         lstm_sent_h_size = 512,  # USE SAME DIMENSION OF THE EMBEDDINGS
         lstm_sent_n_layers = 1,  # number of layers of the sentence LSTM
         lstm_sent_dropout = 0.3, # dropout for sentence LSTM
         lstm_word_h_size = 512,
         lstm_word_n_layers = 1, 
         check_val_every = 10,
         loader_threads = 1,
         single_channel_cnn = False,
         device = "cpu",
         gpu_id = None,
         strategy=None,
         amp_level=None,
         amp_backend=None,
         verbose = False,         # 
         debug = False,           # more verbosity
         dev = False,             # dev limit number of epochs and batches,
         dryrun = False,          # currently UNused: if True do not save any file
         remote_log = False):     # remote log on neptune.ai
    
    config = locals()
    pipeline(config)

# --------------------------

if __name__ == "__main__":
    fire.Fire(main)