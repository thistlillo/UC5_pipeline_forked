import fire
import pt.train_test 

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
    pt.train_test.pipeline(config)

# --------------------------

if __name__ == "__main__":
    fire.Fire(main)