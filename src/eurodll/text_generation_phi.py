import fire


# --------------------------------------------------
def gen_image(filename):
    pass
#< gen_image

# --------------------------------------------------
def gen_tsv():
    pass
    # fot each line
    # gen for image
#< gen_tsv

# --------------------------------------------------
def main(cnn_model,
         rnn_model,
         vocabulary,
         tsv_file,
         out_fn, 
         img_file):
    config = locals()
    # load models

    # read vocabulary

    # tsv or image?

    out_tsv = gen_tsv()

    # save tsv

    print("done.")
#< main

# --------------------------------------------------
if __name__ == "__main__":
    fire.Fire(main)