import fire
from eddl_lib.recurrent_module import EddlRecurrentModule


# ----------------------------------------------
def train(in_tsv,
         exp_fld,
         img_fld,
         cnn_file,
         cnn_out_layer="softmax",
         out_fn = "uc5model_default.bin",
         train_p = 0.7,
         valid_p = 0.1,
         seed = 1,
         shuffle_seed = 2,
         term_column = "auto_term",
         batch_size = 32,
         last_batch = "random",
         n_epochs = 50,
         check_val_every = 10,
         n_sentences = 5,  # ignored when using dataset version "simple"
         n_tokens = 10,
         # max_tokens = 17,  # used by dataset version "simple". In that case, max_sentences and max_sentence_length are ignored
         eddl_cs_mem = "mid_mem",
         eddl_cs = "cpu",
         gpu_id =[3],
         optimizer = "adam",
         lr = 0.09, 
         momentum = 0.9,
         lstm_size = 512,
         emb_size = 512,
         img_size = 224,
         text_column = "text",
         load_data_split = True,
         preload_images = True,
         preproc_images = None,
         load_file = None,
         verbose = False,
         debug = False,
         dev = False,
         remote_log = False):
    config=locals()
    rec_mod = EddlRecurrentModule(config)
    rec_mod.train()
    rec_mod.save()  # save ok, but cannot be loaded (error)
    bleu, gen_wis = rec_mod.predict()
    print(f"BLEU score on test set: {bleu:.3f}")
   
#< train


# ----------------------------------------------
def test(in_tsv,
         exp_fld,
         img_fld,
         load_file,
         cnn_file,
         out_fn = "test.txt",
         train_p = 0.7,
         valid_p = 0.1,
         seed = 1,
         shuffle_seed = 2,
         term_column = "auto_term",
         batch_size = 2,  # needed by the dataloader
         last_batch = "random",
         n_sentences = 5,  # ignored when using dataset version "simple"
         n_tokens = 10,
         # max_tokens = 17,  # used by dataset version "simple". In that case, max_sentences and max_sentence_length are ignored
         eddl_cs_mem = "mid_mem",
         eddl_cs = "cpu",
         gpu_id =[0],
         optimizer = "adam",
         lr = 0.09, 
         momentum = 0.9,
         lstm_size = 512,
         emb_size = 512,
         img_size = 224,
         text_column = "text",
         load_data_split = True,
         preload_images = True,
         preproc_images = None,

         verbose = False,
         debug = False,
         dev = False):
    config=locals()
    print("NOT IMPLEMENTED - USE TRAIN")
    assert False
    rec_mdl = EddlRecurrentModule(config)
    rec_mdl.predict()
#<


# ---------------------------------------
if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "test": test
    })
    
    