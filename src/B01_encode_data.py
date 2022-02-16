# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import fire
import json
import nltk
import numpy as np
import pandas as pd
import pickle
from posixpath import join
from tqdm import tqdm

import text.reports as reports
from text.vocabulary import Vocabulary

re_word_tokenizer = nltk.RegexpTokenizer(r"\w+")

# ----------------------

def image_labels_to_indexes(column, verbose=False):
    u_labs = set()
    total_terms = 0
    normal_cnt = 0

    for l in column.apply(lambda x: x.split(reports.list_sep)).to_list():
        total_terms += len(l)
        for t in l:
            u_labs.add(t.strip())
            if t == "normal":
                normal_cnt += 1
        
    print("|'normal' labels|: ", normal_cnt)
    print(f"|labels|: {total_terms}, |unique|: {len(u_labs)}")

    i2l = { i : l for i, l in enumerate(sorted(u_labs)) }
    l2i = { l : i for i, l in i2l.items() }

    # useless check
    for i in range(len(l2i)):
        assert l2i[ i2l[i] ] == i
    for l in l2i.keys():
        assert i2l[ l2i[l] ] == l

    return l2i, i2l

def encode_image_labels(column, verbose):
    lab2i, i2lab = image_labels_to_indexes( column, verbose )

    enc_col = column.apply( lambda x: reports.list_sep.join([ str(lab2i[y]) for y in x.split(reports.list_sep)] ) )
    return enc_col, lab2i, i2lab


# ----------------------
def build_vocabulary(column,
                     vocab_size=0,
                     min_freq=0,
                     verbose=False,
                     debug=False):

    voc = Vocabulary(verbose=verbose, debug=debug)

    for _, text in column.iteritems():
        voc.add_text(text)

    if verbose:
        voc.print_stats()
    if vocab_size > 0 or min_freq > 0:
        voc.set_n_words(max_words=vocab_size, min_freq=min_freq)  #, max_words=1000)

    if debug:
        text = column.loc[100]
        e_text = voc.encode(text, n_sentences=20, sentence_length=30)
        decoded = voc.decode(e_text)
        print(text)
        print(e_text)
        print(decoded)

    return voc


def image_based_ds(df, config):
    image_filenames = []
    report_ids = []
    texts = []
    labels = []

    for (i, row) in df.iterrows():
        filenames_ = row["image_filename"].split(reports.list_sep)

        # next op needed because in previous versions
        # images were repeated if less than a user-specified number
        # so we only keep the names up to the index orig_n_images
        if row["orig_n_images"] < len(filenames_):
            filenames_ = filenames_[:row["orig_n_images"]]
        id_ = row["id"]
        text_ = row["enc_" + config["text_column"]]
        labels_ = row["labels"]
        for f in filenames_:
            image_filenames.append(f)
            report_ids.append(id_)
            texts.append(text_)
            labels.append(labels_)

    df2 = pd.DataFrame({
        "filename": image_filenames,
        "report_id": report_ids,
        "enc_" + config["text_column"]: texts,
        "labels": labels
    })
    df2.set_index(["filename"], inplace=True, drop=True)
    return df2


# -----------------------------------------------------------------------------------------------------------------------

def main(in_file,
        out_tsv,
        out_img_tsv,
        out_fld,
        term_column=reports.k_auto_term,
        text_column="text",
        vocab_size=1000,
        min_freq=0,
        sen_len=0,
        n_sen=0,
        verbose=False,
        debug=False):

    config = locals()

    # 1: read data
    tsv = pd.read_csv(config["in_file"], sep=reports.csv_sep, na_filter=False)
    print(tsv.columns)
    print(f"read {config['in_file']}, lines: {len(tsv)}")
    if len(tsv) == 0:
        print(f"EMPTY INPUT FILE: {config['in_file']}")
        exit(1)
        
    print(f"image labels from column: {config['term_column']}")
    print(f"texts from column: {config['text_column']}")
    print(f"vocabulary size: {config['vocab_size']}")

    # 2: encode labels
    print("encoding image labels...")
    tsv["labels"], lab2i, i2lab = encode_image_labels(tsv[config["term_column"]], verbose )

    # > check label encoding
    rows = [100, 102, 500]
    for row in rows:
        print(tsv.head())
        labels = tsv.loc[row, term_column].split(reports.list_sep)
        enc_labels = [int(x) for x in tsv.loc[row, "labels"].split(reports.list_sep)]
        if debug:
            print(10 * "-")
            print(f"check/ labels: {labels}")
            print(f"check/ enc labels: {enc_labels}")

        for i in range(len(labels)):
            lab = labels[i]
            enc_lab = i2lab[enc_labels[i]]
            if debug: print(f"check/ \t\t{lab}, {enc_lab} <- back from encoding")
            assert labels[i] == i2lab[enc_labels[i]]
    print(f"label encoding ok - stored in column '{'labels'}'")
    # < check label encoding

    # 3: build vocab
    print("building vocabulary")
    vocabulary = build_vocabulary(tsv[config["text_column"]], vocab_size=config["vocab_size"], min_freq=config["min_freq"], verbose=config["verbose"], debug=config["debug"])
    pickle.dump(vocabulary, open(join(out_fld, "vocab.pickle"), "wb"))
    print(f"vocabulary saved in: {join(out_fld, 'vocab.pickle')}")

    # 4: encode text
    print("encoding text")
    from text.encoding import BaseTextEncoder
    txt_encoder = BaseTextEncoder(vocab=vocabulary)
    text_column = config["text_column"]
    tsv["enc_" + text_column] = tsv[text_column].apply(lambda x: txt_encoder.encode(x, as_string=True, insert_dots=True))

    # 5: image-based dataset
    print("building image-based dataset")
    ib_ds = image_based_ds(tsv, config)
    ib_ds.to_csv(config["out_img_tsv"], sep=reports.csv_sep, index=True)  # index set to image_filename
    print(f"saved tsv with the 'image_based' dataset, one image per row: {config['out_img_tsv']}")
    print(ib_ds.columns)
    
    # 6: preprocess images
    # moved
    # print("preprocessing images")
    # d = {}
    # filenames = ib_ds.filename.to_list()
    # augs = ecvl.SequentialAugmentationContainer([
    #             ecvl.AugDivBy255(),  #  ecvl.AugToFloat32(divisor=255.0),
    #             ecvl.AugNormalize(122.75405603 / 255.0, 0.296964375 / 255.0),
    #             ecvl.AugResizeDim([300, 300]),
    #             # ecvl.AugCenterCrop([256, 256]),  # XXX should be parametric, for resnet 18
    #             ecvl.AugCenterCrop([img_size, img_size]),  # XXX should be parametric, for resnet 18
    #             ])
    # for i, fn in enumerate(tqdm(filenames)):
    #     afn = join(img_fld, fn)
    #     img = ecvl.ImRead(afn)  # , flags=ecvl.ImReadMode.GRAYSCALE)
    #     augs.Apply(img)
    #     ecvl.RearrangeChannels(img, img, "cxy")
    #     d[fn] = np.array(img)
        
    # s = pd.Series(d)
    # ff = join(out_fld, "images.pickle")
    # with open( ff, "wb" ) as fout:
    #     pickle.dump(s, fout)
    # print(f"saved preprocessed images in: {ff}")

    # 7: save
    tsv.to_csv(config["out_tsv"], sep=reports.csv_sep, index=False)
    print(f"saved tsv with the new 'labels': {config['out_tsv']} and new encoded text column: 'enc_{text_column}'")
    out_fld = config["out_fld"]
    json.dump(lab2i, open(join(out_fld, "lab2index.json"), "w") )
    json.dump(i2lab, open(join(out_fld, "index2lab.json"), "w") )
    print(f"image label to index saved in {join(out_fld, 'lab2index.json')}")
    print(f"index to image label saved in {join(out_fld, 'index2lab.json')}")

    

    
    print("exiting with success")

    
if __name__ == "__main__":
    fire.Fire(main)
