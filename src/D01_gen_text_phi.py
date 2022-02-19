import fire
import eurodll.text_generation_phi

def main(out_fn,
        exp_fld,
        cnn_model,
        rnn_model,
        n_tokens,
        emb_size = 512,
        lstm_size = 512,
        tsv_file = None,
        img_file = None,
        dev=False):
    config = locals()
    eurodll.text_generation_phi.main(config)
#<

# --------------------------------------------------
if __name__ == "__main__":
    fire.Fire(main)