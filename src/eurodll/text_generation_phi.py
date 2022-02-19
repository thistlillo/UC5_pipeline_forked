import pickle
from posixpath import join
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from eurodll.recurrent_models import nonrecurrent_lstm_model, generate_text
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

# --------------------------------------------------
def main(config):
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
    fn = args.rnn_model.replace(".onnx", ".bin")
    eddl.load(rnn, fn)
    print(f"!!! rnn weights read from: {fn} -- IMPORTANT: BIN FILE WAS USED")
    eddl.set_mode(rnn, 0)
    #<
    
    # tsv or image?

    out_tsv = gen_tsv()

    # save tsv

    print("done.")
#< main
