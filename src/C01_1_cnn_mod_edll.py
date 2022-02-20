# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import fire
from eddl_lib.cnn_module import EddlCnnModule


def train(in_tsv,
         exp_fld,
         img_fld,
         cnn_out_layer = "softmax",
         out_fn = "uc5model_default.bin",
         train_p = 0.7,
         valid_p = 0.1,
         seed = 1,
         shuffle_seed = 2,
         # term_column = "auto_term",
         batch_size = 32,
         last_batch = "random",
         n_epochs = 50,
         check_val_every = 1,
         n_sentences = 5,  # ignored when using dataset version "simple"
         n_tokens = 10,
         # max_tokens = 17,  # used by dataset version "simple". In that case, max_sentences and max_sentence_length are ignored
         eddl_cs_mem = "mid_mem",
         eddl_cs = "cpu",
         gpu_id = [1],
         optimizer = "adam",
         lr = 0.002, 
         momentum = 0.9,
         gamma = 0.95,
         lstm_size = 512,
         emb_size = 512,
         img_size = 224,
         text_column = "text",
         out_layer_type = "softmax",
         load_data_split = True,
         preload_images = True,
         preproc_images = None,
         load_file = None,
         verbose = False,
         debug = False,
         dev = False,
         remote_log=False,
         patience = 10,
         patience_kick_in = 50):

    config = locals()
    cnn_module = EddlCnnModule(config)
    cnn_module.train()
    cnn_module.save()
    print("train - done.")
#< train

# --------------------------------------------------
def test(in_tsv,
         exp_fld,
         img_fld,
         train_p = 0.7,
         valid_p = 0.1,
         seed = 1,
         batch_size = 1,
         shuffle_seed = 2,
         last_batch = "random",
         eddl_cs_mem = "mid_mem",
         eddl_cs = "cpu",
         gpu_id =[0],
         img_size = 224,
         text_column = "text",
         load_data_split = True,
         preload_images = True,
         preproc_images = None,
         load_file = None,
         verbose = False,
         debug = False,
         dev = False,
         # required but unused in cnn traiing
         n_sentences = 5,
         n_tokens = 10,
         ):
    config = locals()
    cnn_module = EddlCnnModule(config)
    cnn_module.test() # not implemented
#< test


# --------------------------------------------------
if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "test": test
    })