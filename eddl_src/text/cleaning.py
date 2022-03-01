
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import re
import sys
import string
from tqdm import tqdm
import text.reports as reports


nltk.download('punkt')
re_word_tokenizer = nltk.RegexpTokenizer(r"\w+")

def clean_text_v1(text, verbose=False):
    def subst_numbers(token):
        s = re.sub(r"\A\d+(,|\.)\d+", "_NUM_", token)  # _DEC_ for finer texts
        s = re.sub(r"\A\d+", "_NUM_", s)
        return s

    def subst_meas(text):
        # substitute measures
        e = r"(_NUM_|_DEC_)\s?(cm|mm|in|xxxx)|_NUM_ x _MEAS_|_DEC_ x _MEAS_|_MEAS_ x _MEAS_ x _MEAS|_MEAS_ x _MEAS_"
        t1 = text
        while True:
            t2 = re.sub(e, "_MEAS_", t1)
            if t1 == t2:
                break
            else:
                t1 = t2
        return t1

    text2 = text.replace(" ", " ")
    text2 = text2.replace("..", ".")


    symbols = ",;:?)(!"

    e = "|".join([re.escape(s) for s in symbols])
    text2 = re.sub(e, " ", text2)
    # text2 = " ".join( [t.strip() for t in text2.split(" ")])
    # numbered list items
    text2 = re.sub(r"\s\d+\. ", " ", text2)
    # dash
    text2 = re.sub(r"-", "_", text2)
    # percentages
    text2 = re.sub(r"\d+%\s", "_PERC_ ", text2)
    # XXXX XXXX -> XXXX_XXX
    text2 = re.sub(r"xxxx(\sxxxx)+", "xxxx", text2)
    # ordinals
    text2 = re.sub(r"1st|2nd|3rd|[0-9]+th ", "_N_TH_ ", text2)


    sentences = []
    for sent in sent_tokenize(text2):
        new_tokens = [subst_numbers(token) for token in word_tokenize(sent)[:-1]]  # [:-1] not using last dot
        # for token in word_tokenize(sent):
        #     w = subst_numbers(token)
        #     new_tokens.append(w)
    
        sent = " ".join(new_tokens)
        sent = subst_meas(sent)
        sentences.append(sent)

    text2 = ". ".join(sentences) + "."  # dots, and in particular the last ., were not removed by word_tokenize

    if verbose and text != text2 and "_MEAS_" in text2:
        print("* IN (it has been modified):")
        print(text)
        print("* OUT:")
        print(text2)
        print(20 * "/")

    return text2



def clean_text_column(column, verbose=False, cleaning="v1"):
    # first make sure tokens are separated by a single space
    if cleaning == "v1":
        return column.apply(lambda x: clean_text_v1(x, verbose))
    else:
        print(f"FAILURE, unknown cleaning mode: {cleaning}")
        exit(1)


def fix_sentences_in_text(x):
    out_sents = []
    for sent in sent_tokenize(x):
        se = []
        for word in sent.split(" "):
            se.append(word.strip())
        out_sents.append(" ".join(se))

    return (". ".join(out_sents) + ".")


def fix_sentences_in_column(col):
    col = col.apply(lambda x: fix_sentences_in_text(x))
    return col


# --------------------------------------------------
def clean(df, in_cols, out_col="text", verbose=False):
    # step
    print(f"concat columns: {', '.join(in_cols)} in {out_col}")
    def concat_cols(cols):
        for c in cols:
            if not c.endswith("."):
                c += "."
        out =  " ".join(cols)
        return out

    df[out_col] = ""
    df[out_col] = df[in_cols].apply(lambda x: concat_cols(x), axis=1)
    # df[out_col] = df[in_cols[0]]
    # for i in range(1, len(in_cols)):
    #     df[out_col] += (" " + tsv[in_cols[i]])

    # step
    print("to lowercase")
    df[out_col] = df[out_col].apply(lambda x: x.lower())

    # step
    print(f"cleaning {out_col}...")
    df[out_col] = clean_text_column(df[out_col], verbose=verbose)

    return df
#<