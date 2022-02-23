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
def classify(img, cnn, dev=False):
    cnn_out = eddl.getLayer(cnn, "cnn_out")
    cnn_top = eddl.getLayer(cnn, "top")
    # -
    a = np.expand_dims(np.array(img, copy=False), axis=0)  # add batch dimension
    eddl.forward(cnn, [Tensor.fromarray(a)])
    # - 
    cnn_semantic = eddl.getOutput(cnn_out)
    classes = np.array(cnn_semantic)
    c = np.argmax(classes, axis=-1)
    return c
#<

# --------------------------------------------------
def main(out_fn,
        exp_fld,
        img_fld,
        cnn_model,
        img_size = 224,
        emb_size = 512,
        tsv_file = None,
        dev=False):
    config = locals()
    args = Bunch(**config)
    
    #> load models
    cnn = eddl.import_net_from_onnx_file(args.cnn_model)
    print(f"trained cnn read from: {args.cnn_model}")
    print(f" - cnn input shape {cnn.layers[0].input.shape}")
    print(f" - cnn output shape {cnn.layers[-1].output.shape}")
    eddl.build(cnn, eddl.adam(0.01), ["softmax_cross_entropy"], ["accuracy"], eddl.CS_CPU(), init_weights=False)
    eddl.set_mode(cnn, 0)
    print("cnn model built successfully")
    
    augs = augs_f(args.img_size)

    print(f"> processing {tsv_file}")
    tsv = pd.read_csv(tsv_file, sep=csv_sep)
    print(f" - shape: {tsv.shape}")
    print(f" - columns: {', '.join(tsv.columns)}")
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
    
    for p, l in zip(partitions, id_lists):
        print(f"processing {p}, len: {len(l)}")
        indexes = tsv.filename.isin(l)
        filenames = tsv.loc[indexes, "filename"].tolist()
        labels = tsv.loc[indexes, "labels"].tolist()
        gen_texts = []
        gen_word_idxs = []

        for idx, fn in enumerate(filenames):
            img = load_image(join(args.img_fld, fn), augs=augs)
            c = classify(img, cnn)
            print(f"target: {labels[idx]}, predicted: {c}")
            
        # tsv.loc[indexes, "gen_text"] = gen_texts
        
    tsv.to_csv(args.out_fn, sep=csv_sep)
    print(f"saved: {args.out_fn}")
    print("done.")
#< main
