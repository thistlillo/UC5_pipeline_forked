# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import numpy as np
import pandas as pd
import pickle
from posixpath import join
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import pyecvl.ecvl as ecvl
from tqdm import tqdm

from eddl_lib.recurrent_models import nonrecurrent_lstm_model, generate_text
from text.encoding import SimpleCollator
from text.metrics import compute_bleu_edll
from text.reports import csv_sep, list_sep

# --------------------------------------------------
def gen_image_from_file(filename):
    pass
#< gen_image

# --------------------------------------------------
def gen_tsv():
    pass
    # fot each line
    # gen for image
#< gen_tsv

# https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__.update(kw)
#<

# --------------------------------------------------
# 
augs_f = lambda x: ecvl.SequentialAugmentationContainer([
                ecvl.AugDivBy255(),  #  ecvl.AugToFloat32(divisor=255.0),
                ecvl.AugNormalize(122.75405603 / 255.0, 0.296964375 / 255.0),
                ecvl.AugResizeDim([300, 300]),
                # ecvl.AugCenterCrop([256, 256]),  # XXX should be parametric, for resnet 18
                ecvl.AugRandomCrop([x, x]),  # XXX should be parametric, for resnet 18
                ])

def load_image(path, augs=None):
    img = ecvl.ImRead(path)
    ecvl.RearrangeChannels(img, img, "cxy")
    if augs:
        augs.Apply(img)
    return img
#<

# --------------------------------------------------

# simplified version of EddlRecurrentModule.predict(.)
def annotate(img, cnn, rnn, n_tokens, dev=False):
    if dev:
        print(f"annotating an ECVL image {img.Width()} x {img.Height()} x {img.Channels()} with {n_tokens} words")
    # - 
    cnn_out = eddl.getLayer(cnn, "cnn_out")
    cnn_top = eddl.getLayer(cnn, "top")
    # -
    a = np.expand_dims(np.array(img, copy=False), axis=0)  # add batch dimension
    eddl.forward(cnn, [Tensor.fromarray(a)])
    # - 
    cnn_semantic = eddl.getOutput(cnn_out)
    cnn_visual = eddl.getOutput(cnn_top)
    # - 
    text =  generate_text(rnn, n_tokens, visual_batch=cnn_visual, semantic_batch=cnn_semantic, dev=False)
    return text
#<

# --------------------------------------------------
def main(out_fn,
        exp_fld,
        img_fld,
        cnn_model,
        rnn_model,
        n_tokens,
        img_size = 224,
        emb_size = 512,
        lstm_size = 512,
        tsv_file = None,
        img_file = None,
        dev=False):
    config = locals()
    args = Bunch(**config)
    
    #>
    with open(join(args.exp_fld, "vocab.pickle"), "rb") as fin:
        vocab = pickle.load(fin)
    #<

    #> load models
    cnn = eddl.import_net_from_onnx_file(args.cnn_model)
    print(f"trained cnn read from: {args.cnn_model}")
    print(f" - cnn input shape {cnn.layers[0].input.shape}")
    print(f" - cnn output shape {cnn.layers[-1].output.shape}")
    eddl.build(cnn, eddl.adam(0.01), ["softmax_cross_entropy"], ["accuracy"], eddl.CS_CPU(), init_weights=False)
    eddl.set_mode(cnn, 0)
    print("cnn model built successfully")
     
    visual_dim = eddl.getLayer(cnn, "top").output.shape[1]
    semantic_dim = eddl.getLayer(cnn, "cnn_out").output.shape[1]
    
    rnn = nonrecurrent_lstm_model(visual_dim=visual_dim, semantic_dim=semantic_dim, vs=vocab.n_words, emb_size=args.emb_size, lstm_size=args.lstm_size)
    eddl.build(rnn, eddl.adam(0.01), ["softmax_cross_entropy"], ["accuracy"], eddl.CS_CPU(), init_weights=False)
    print("rnn model built successfully")
    # TODO: remove once onnx can be used
    fn = args.rnn_model.replace(".onnx", ".bin")
    eddl.load(rnn, fn)
    eddl.set_mode(rnn, 0)
    print(f"!!! rnn weights read from: {fn} -- IMPORTANT: BIN FILE WAS USED")
    #<
    
    augs = augs_f(args.img_size)

    #>
    print("> step 1: generate text for a single image")
    path = args.img_file or join(args.img_fld, "CXR1521_IM-0337-1001.png")
    img = load_image(path, augs)
    text = annotate(img, cnn, rnn, args.n_tokens, dev=args.dev)
    print(f"< step 1: generated text: {text}")
    #<
    
    print("")
    
    #> read the tsv and annotate all of the images
    collator = SimpleCollator()

    print(f"> step 2: processing {tsv_file}")
    tsv = pd.read_csv(tsv_file, sep=csv_sep)
    print(f" - shape: {tsv.shape}")
    print(f" - columns: {', '.join(tsv.columns)}")
    # reading partitions, assume they exist
    # NOTE: we might regenerate them providing to this script: shuffle_seed, training and validation percentages
    #       not implemented for the sake of clarity
    train_ids, val_ids, test_ids = [], [], []
    id_lists = []
    partitions = ["train_ids.txt", "valid_ids.txt", "test_ids.txt"]
    for fn in partitions:
        with open(join(args.exp_fld, fn), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            lines = lines[:10] if args.dev else lines
            id_lists.append( lines )
    #<
    
    if not args.dev:
        assert sum([len(l) for l in id_lists]) == tsv.shape[0]
    
    #<
    def tokens_eddl(wis_str, max_tokens):
        wis = wis_str.split(" ")
        out = []
        i = 0
        while i < len(wis) and i < max_tokens-1:
            out.append(wis[i].replace(".", ""))
            i+=1
        out.append(str(vocab.EOS_I))
        return " ".join(out)
    #<

    # all intermediate results saved as strings to avoid mixing lists, nd.array etc
    tsv["enc_text_eddl"] = tsv["enc_text"].apply(lambda x: " ".join(
        [str(y) for y in collator.parse_and_collate(x, n_tokens=args.n_tokens)]
        ))
    tsv["target_text"] = tsv["enc_text_eddl"].apply(lambda x:  # x over rows in column "text"
        vocab.decode_sentence(x))
    
    tsv["gen_text"] = ""
    tsv["gen_wis"] = ""
    tsv["partition"] = ""
    tsv["bleu"] = np.nan
    

    for p, l in zip(partitions, id_lists):
        print(f"processing {p}, len: {len(l)}")
        indexes = tsv.filename.isin(l)
        filenames = tsv.loc[indexes, "filename"].tolist()
        
        gen_texts = []
        gen_word_idxs = []

        for fn in tqdm(filenames, disable=args.dev):
            img = load_image(join(args.img_fld, fn), augs=augs)
            gen_wis = annotate(img, cnn, rnn, args.n_tokens, dev=args.dev)[0]
            gen_wis_str = " ".join([str(n) for n in gen_wis])
            decoded = vocab.decode_sentence(gen_wis_str)
            gen_word_idxs.append(gen_wis_str)
            gen_texts.append(decoded)
        
        tsv.loc[indexes, "gen_text"] = gen_texts
        tsv.loc[indexes, "gen_wis"] = gen_word_idxs
        tsv.loc[indexes, "partition"] = p.split("_", 1)[0]
        tsv.loc[indexes, "bleu"] = tsv.loc[indexes, ["gen_wis", "enc_text_eddl"]].apply(
            lambda x: compute_bleu_edll(x["gen_wis"], x["enc_text_eddl"]), axis=1)
        print(f"mean BLEU on {p}: {tsv.loc[indexes, 'bleu'].mean():.3f}")
    tsv.to_csv(args.out_fn, sep=csv_sep)
    print(f"saved: {args.out_fn}")
    print("done.")
#< main
