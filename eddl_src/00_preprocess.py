#
# all preprocessing steps in a single file.
#


from bs4 import BeautifulSoup
from collections import defaultdict
import fire
import json
from numpy import count_nonzero as nnz
import os
import pandas as pd
import pickle
from posixpath import join
from tqdm import tqdm

from text.encoding2 import BaseTextEncoder
import text.cleaning as text_cleaning
import text.reports as reports
from text.vocabulary import Vocabulary


# --------------------------------------------------
def parse_single_report(filepath, verbose=False):
    with open(filepath, "r", encoding="utf-8") as fin:
        xml = fin.read()

    soup = BeautifulSoup(xml, "lxml")
    d = reports.parse_soup(soup)
    d["filename"] = os.path.basename(filepath)
    return d

def parse_reports(txt_fld, ext="xml", verbose=False, dev=False):
    reports = []
    for i, fn in enumerate(tqdm( [ join(txt_fld, fn) for fn in os.listdir(txt_fld) if (ext is None or fn.endswith(ext)) ])):
        reports.append(parse_single_report(fn))
    return reports

def finalize(df):
    df[reports.k_n_major_mesh] = df[reports.k_major_mesh].apply(lambda x: len(x))
    df[reports.k_major_mesh] = df[reports.k_major_mesh].apply(lambda x: reports.list_sep.join(x))
    df[reports.k_n_auto_term] = df[reports.k_auto_term].apply(lambda x: len(x))
    df[reports.k_auto_term] = df[reports.k_auto_term].apply(lambda x: reports.list_sep.join(x))    
    df[reports.k_n_images] = df[reports.k_image_filename].apply(lambda x: len(x))
    df[reports.k_image_filename] = df[reports.k_image_filename].apply(lambda x: reports.list_sep.join(x))
    return df

def build_raw_tsv(txt_fld, out_fld):
    reps = parse_reports(txt_fld, dev=True)
    df = pd.DataFrame.from_records(reps)
    df = finalize(df)
    return df


# --------------------------------------------------
def raw_data_cleansing(df):
    #> drop reports with no images
    iii = df[reports.k_n_images] == 0
    df = df.drop(df[iii].index)
    print(f"dropped reports without images: {nnz(iii)}")
    #< 

    #> drop rows with mesh=="no indexing" and empty auto
    iii = df[reports.k_major_mesh] == "no indexing"
    jjj = df[reports.k_auto_term].str.len() == 0
    #df = df.drop(df[iii|jjj].index)
    #print(f"dropped {nnz(iii&jjj)} rows without any info in tags")

    #> keep only the first term, lowercased
    def clean_terms(terms):
        # x of the form: # Markings/lung/bilateral/interstitial/diffuse/prominent;Fibrosis/diffuse
        groups = [y.strip() for y in terms.split(reports.list_sep)]
        
        # initially split with ; => groups of the form t1/t2/t3
        terms = [x.split("/")[0].lower().strip() for x in groups]  # take first element in group t1/t2/t3, make it lowercase
        # some terms contain a comma: keep only the first
        new_terms = []
        for t in terms:
            new_terms.append(t.split(",")[0])

        return reports.list_sep.join(new_terms)
    #
    cols = [reports.k_auto_term, reports.k_major_mesh]
    for c in cols:
        df[c] = df[c].apply(lambda x: clean_terms(x))
    #<

    #> fill empty normal impression
    iii = df[reports.k_impression].str.len() == 0
    jjj = df[reports.k_major_mesh].str == "normal"
    df.loc[iii & jjj, reports.k_impression] = "normal"
    print(f"filled {nnz(iii&jjj)} empty impressions where major mesh was normal")
    #<

    #> fill empty auto tag
    iii = df[reports.k_major_mesh].str.lower() == "normal"
    jjj = df[reports.k_auto_term].str.len() == 0
    df.loc[iii & jjj, reports.k_auto_term] = "normal"
    df.loc[iii & jjj, reports.k_n_auto_term] = 1
    print(f"filled {nnz(iii&jjj)} auto_term normal rows using mesh normal tag")
    #<

    #>
    iii = df[reports.k_impression].str.len() == 0
    jjj = df[reports.k_findings].str.len() == 0
    print(f"DROPPED incomplete reports: {nnz(iii & jjj)}")
    df = df.drop(df[iii & jjj].index)
    #<

    #> no indexing
    # iii = df[reports.k_major_mesh] == "no indexing"
    # jjj = df[reports.k_auto_term].str.len() == 0
    # kkk = iii & jjj
    # df.drop(df[kkk].index)
    # print(f"dropped {nnz(kkk)} rows with no indexing")

    return df
#<

# --------------------------------------------------

def make_manual_corrections(tsv, dev=False):
    import os
    # TODO read file with term substitutions: make it a cmd line argument
    fn = "text/auto_term_norm.txt"
    
    print("reading 'manual' corrections from:", os.path.realpath(fn))

    lines = []
    d = {}  # (wrong) term to substitute -> correct term
    with open(fn, "r", encoding="utf-8") as fin:
        lines = [line for line in fin.readlines() if len(line.strip()) > 0]
    for line in lines:
        tt = [t.strip() for t in line.split(":") if len(t.strip()) > 0]
        if len(tt) != 2:
            print(f"FAILURE, wrong format. Expected format 't1: t2', found this line: '{line}'")
            exit(1)
        d[ tt[0] ] = tt[1]

    if dev:
        print("auto_term substitutions:")
        for i, (k, v) in enumerate(d.items()):
            print(f"\t {i:03d}) {k} -> {v}")

    def perform_subst(s):
        terms = s.split(reports.list_sep)
        new_terms = []
        subst = False
        for t in terms:
            if t not in d.keys():
                new_terms.append(t)
                continue

            new_term = d.get(t, t)
            if (new_term != t) and (new_term not in terms):  # substitution
                subst=True
                new_terms.append(new_term)

        new_s = reports.list_sep.join(new_terms)

        if subst and dev:
            print(f"SUBSTITUTION {s} -> {new_s}")
        return new_s

    tsv[reports.k_auto_term] = tsv[reports.k_auto_term].apply(lambda x: perform_subst(x))
    return tsv
#

# --------------------------------------------------
def apply_term_freq_thresh(df, col_name, th, dev=False):
    print(f"using threshold {th} on {col_name}")
    rep_ids = df.filename.tolist()
    n_images = df. n_images.tolist()
    labels = df[col_name].tolist()
    cnt = defaultdict(int)
    for l, n in zip(labels, n_images):
        for l in l.split(reports.list_sep):
            cnt[l] += n
    to_remove = []

    for l, c in cnt.items():
        if dev: print(f"{l} occurs {c}")
        if c < th:
            to_remove.append(l)
    
    n_labels_kept = len(cnt) - len(to_remove)
    print(f"{len(to_remove)}/{len(cnt)} labels will be removed, remaining {n_labels_kept}")
    
    if dev:
        for l in to_remove:
            print(f"- removed: {l}")
    
    # for l, c in cnt.items():
    #     if l not in to_remove:
    #         print(f"label {l} kept: {c} occurrences")

    def remove_labels(labs, rem):
        labs2 = []
        for l in labs.split(reports.list_sep):
            if l not in rem:
                labs2.append(l)
        if len(labs2) == 0:
            labs2.append("misc")
        return reports.list_sep.join(labs2)

    #> check
    nc = df[col_name].apply(lambda x: remove_labels(x, to_remove))
    ncl = nc.tolist()
    c = set()
    for ll in ncl:
        for l in ll.split(reports.list_sep):
            c.add(l)
    print(f"verifying label filtering: assert {len(c)} == {n_labels_kept + 1}?")
    assert len(c) == (n_labels_kept + 1)
    return nc

# --------------------------------------------------
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

    return voc


# --------------------------------------------------

def image_labels_to_indexes(column, verbose=False):
    u_labs = set()
    total_terms = 0
    normal_cnt = 0

    for l in column.apply(lambda x: x.split(reports.list_sep)).to_list():
        print(l)
        total_terms += len(l)
        for t in l:
            u_labs.add(t.strip())
            if t == "normal":
                normal_cnt += 1
        
    print("|'normal' labels|: ", normal_cnt)
    print(f"|labels|: {total_terms}, |unique|: {len(u_labs)}")

    i2l = { i : l for i, l in enumerate(sorted(u_labs)) }
    l2i = { l : i for i, l in i2l.items() }
    for i, l in i2l.items():
        print(f"label {i}: {l}")

    # useless check
    for i in range(len(l2i)):
        assert l2i[ i2l[i] ] == i
    for l in l2i.keys():
        assert i2l[ l2i[l] ] == l

    return l2i, i2l

def encode_image_labels(column, verbose):
    lab2i, i2lab = image_labels_to_indexes( column, verbose )

    enc_col = column.apply( lambda x: reports.list_sep.join([ str(lab2i[y]) for y in sorted(list(set(x.split(reports.list_sep))))] ) )
    return enc_col, lab2i, i2lab

# --------------------------------------------------

def image_based_ds(df, img_fld, enc_text_col):
    image_filenames = []
    report_ids = []
    texts = []
    auto_labels = []
    mesh_labels = []
    n_images = []

    for (i, row) in df.iterrows():
        filenames_ = [os.path.abspath(join(img_fld, fn)) for fn in row["image_filename"].split(reports.list_sep)]

        id_ = row["id"]
        text_ = row[enc_text_col]
        auto_labels_ = row["auto_labels"]
        mesh_labels_ = row["mesh_labels"]
        for f in filenames_:
            image_filenames.append(f)
            report_ids.append(id_)
            texts.append(text_)
            auto_labels.append(auto_labels_)
            mesh_labels.append(mesh_labels_)
    print(f"number of images: {len(image_filenames)}")

    df2 = pd.DataFrame({
        "filename": image_filenames,
        "report_id": report_ids,
        enc_text_col: texts,
        "auto_labels": auto_labels,
        "mesh_labels": mesh_labels
    })
    #df2.set_index(["filename"], inplace=True, drop=True)
    return df2



# --------------------------------------------------
def main(txt_fld = "../data/text",  # folder containing the xml reports
        img_fld = "../data/image",  # folder containing the images
        out_fld=".", 
        out_fn=None,  # default is reports.tsv & img_<default>
        apply_min_term_freq = True,
        min_term_freq_mesh = 100,
        min_term_freq_auto = 100,
        vocab_size = 1000,
        min_token_freq = 2,
        force_rebuild=False,
        remove_normal_class=False): # path for the output file):
    
    if force_rebuild or not os.path.exists(join(out_fld, "reports_raw.tsv")):
        df = build_raw_tsv(txt_fld, out_fld)
        df.to_csv(join(out_fld, "reports_raw.tsv"), sep=reports.csv_sep, index=False, )
        print(f"saved: {join(out_fld, 'reports_raw.tsv')}")

    df = pd.read_csv(join(out_fld, 'reports_raw.tsv'), sep=reports.csv_sep, na_filter=False)
    
    #> raw data
    rows1 = df.shape[0]
    df = raw_data_cleansing(df)
    df = make_manual_corrections(df)
    print(f"output df with {len(df)} rows, removed {rows1 - len(df)} reports")

    # select tags with min frequency
    # after all the previous steps, recompute number of terms
    def count_terms(x):
        if len(x) == 0:
            return 0
        terms = x.split(reports.list_sep)
        return len(terms)
    if apply_min_term_freq:
        df[reports.k_auto_term] = apply_term_freq_thresh(df, reports.k_auto_term, min_term_freq_auto)
        df[reports.k_major_mesh] = apply_term_freq_thresh(df, reports.k_major_mesh, min_term_freq_mesh)
    df[reports.k_n_major_mesh] = df[reports.k_major_mesh].apply(lambda x: count_terms(x))  # len(reports.list_sep.split(x)) if len(x)>0 else 0)
    df[reports.k_n_auto_term] = df[reports.k_auto_term].apply(lambda x: len(x.split(reports.list_sep)) if len(x)>0 else 0)
    #<

    # ---------------

    #> text encoding
    text_col = "text"
    enc_text_col = "enc_" + text_col

    df = text_cleaning.clean(df, ['findings','impression'], out_col=text_col)

    vocabulary = build_vocabulary(column=df[text_col], vocab_size=vocab_size, min_freq=min_token_freq, verbose=False, debug=False)
    pickle.dump(vocabulary, open(join(out_fld, "vocab.pickle"), "wb"))

    txt_encoder = BaseTextEncoder(vocab=vocabulary)
    df[enc_text_col] = df[text_col].apply(lambda x: txt_encoder.encode(x, as_string=True, insert_dots=True))
    #<

    #>
    if remove_normal_class:
        normal_iii = df[reports.k_major_mesh] == "normal"
        print("WARNING: removing normal class")
        print("number of normal reports:", nnz(normal_iii))
        print("shape of current dataset:", df.shape)
        df = df[~normal_iii]
        print("\t new shape:", df.shape)

    print("WARNING: removing technical quality of image unsatisfactory")
    iii = df[reports.k_major_mesh].apply( lambda x: "technical quality of image unsatisfactory" in x.split(reports.list_sep) )
    print("number of dropped reports:", nnz(iii))
    df = df[~iii]
    print("\t new shape:", df.shape)

    print("WARNING: removing no indexing")
    iii = df[reports.k_major_mesh].apply( lambda x: "no indexing" in x.split(reports.list_sep) )
    print("number of dropped reports:", nnz(iii))
    df = df[~iii]
    print("\t new shape:", df.shape)
    #<
    

    #>
    print("encoding image labels...")
    print("- auto_term labels:")
    df["auto_labels"], auto_lab2i, auto_i2lab = encode_image_labels(df["auto_term"], verbose=True )
    for i,l in auto_i2lab.items():
        print(f"auto terms {i} -> {l}")

    print("- major_mesh labels:")
    df["mesh_labels"], mesh_lab2i, mesh_i2lab = encode_image_labels(df["major_mesh"], verbose=True )
    json.dump(auto_lab2i, open(join(out_fld, "auto_lab2index.json"), "w") )
    json.dump(auto_i2lab, open(join(out_fld, "auto_index2lab.json"), "w") )
    json.dump(mesh_lab2i, open(join(out_fld, "mesh_lab2index.json"), "w") ) 
    json.dump(mesh_i2lab, open(join(out_fld, "mesh_index2lab.json"), "w") )
    #<

    df.set_index("filename", inplace=True)
    ib_ds = image_based_ds(df, img_fld, enc_text_col=enc_text_col).set_index("filename")

    #>
    out_fn = out_fn or "reports.tsv"
    img_out_fn = "img_" + out_fn
    reports_fn = join(out_fld, out_fn)
    ib_fn = join(out_fld, img_out_fn)
    df.to_csv(reports_fn, index="id", sep=reports.csv_sep)
    ib_ds.to_csv(ib_fn, index="image_filename", sep=reports.csv_sep)
    #< 

    print(f"report-based csv file: {df.shape}: {reports_fn}")
    print(f"image-based csv file: {ib_ds.shape}: {ib_fn}")
    print("done.")
#

# --------------------------------------------------
if __name__ == "__main__":
    fire.Fire(main)

