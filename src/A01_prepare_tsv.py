# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import fire
from numpy import count_nonzero as nnz
import pandas as pd

import text.reports as reports

def drop_reports_no_images(tsv, verbose=False):
    iii = tsv[reports.k_n_images] == 0
    if verbose:
        print(f"DROPPED reports without images: {nnz(iii)} ")
    return tsv.drop(tsv[iii].index)
    

def all_terms_to_lowercase(tsv, verbose=False):
    # some reports contain the same terms twice, in two different cased versions. For example: scarring; Scarring
    def remove_duplicates(s):
        terms = [t.strip() for t in s.split(reports.list_sep) if len(t.strip()) > 0]
        new_terms = []
        for t in terms:
            if t not in new_terms:
                new_terms.append(t)
        return reports.list_sep.join(new_terms)

    tsv[reports.k_major_mesh] = tsv[reports.k_major_mesh].apply( lambda x: remove_duplicates(x.lower()) )
    tsv[reports.k_auto_term] = tsv[reports.k_auto_term].apply( lambda x: remove_duplicates(x.lower()) )
    

def drop_incomplete_reports(tsv, verbose=False):
    """
    Drop rows with empty 'impression' and 'findings columns' (both of them).
    """
    iii = tsv[reports.k_impression].str.len() == 0
    jjj = tsv[reports.k_findings].str.len() == 0
    if verbose:
        print(f"DROPPED incomplete reports: {nnz(iii & jjj)}")

    tsv = tsv.drop(tsv[iii & jjj].index)
    return tsv

def fill_empty_impression(tsv, verbose=False):
    iii = tsv[reports.k_impression].str.len() == 0
    jjj = tsv[reports.k_major_mesh].str.lower() == "normal"
    tsv.loc[tsv.index[ iii & jjj ], reports.k_impression] = "normal"

    if verbose:
        print(f"FILLED empty '{reports.k_impression}' fields in normal reports: {nnz(iii & jjj)}")


def fill_auto_tag(tsv, verbose=False):
    iii = tsv[reports.k_major_mesh].str.lower() == "normal"
    jjj = tsv[reports.k_auto_term].str.len() == 0
    tsv.loc[iii & jjj, reports.k_auto_term] = "normal"
    tsv.loc[iii & jjj, reports.k_n_auto_term] = 1
    if verbose:
        print(f"FILLED normal '{reports.k_auto_term}' fields: {nnz(iii & jjj)} ")


def normalize_n_images(tsv, max_images = 0, verbose=False):
    def adjust_n_images(s):  # is is of the form filename1.png;filename2.png[;...]
        splitted = sorted([fn.strip() for fn in s.split(reports.list_sep)])
        if max_images > 0:
            if len(splitted) < max_images:
                first = splitted[0]
                splitted += [first] * (max_images - len(splitted))
            elif len(splitted) > max_images:
                splitted = splitted[:max_images]

        return reports.list_sep.join(splitted)

    col = reports.k_image_filename  # we will look at the content of this column
    tsv[col] = tsv[col].apply(lambda x: adjust_n_images(x))
    
    tsv[reports.k_n_images] = tsv.loc[:, reports.k_image_filename].apply(lambda x: len(x.split(reports.list_sep)))

    
    # check
    # assert(nnz(tsv[reports.k_n_images] != max_images) == 0)
    if verbose:
        print(f"NORMALIZED number of images per report: {max_images}")


def simplify_major_mesh(tsv, n_mesh=3, verbose=False):
    def simplify_list(x):
        # x of the form: # Markings/lung/bilateral/interstitial/diffuse/prominent;Fibrosis/diffuse
        groups = [y.strip() for y in x.split(reports.list_sep)]
        if n_mesh > 0:
            groups = groups[:n_mesh]

        # initially split with ; => groups of the form t1/t2/t3
        terms = [x.split("/")[0].lower().strip() for x in groups]  # take first element in group t1/t2/t3, make it lowercase
        out = reports.list_sep.join(terms)
        return out

    tsv.loc[:, reports.k_major_mesh] = tsv.loc[:, reports.k_major_mesh].apply(lambda x: simplify_list(x))
    tsv.loc[:, reports.k_n_major_mesh] = tsv.loc[:, reports.k_major_mesh].apply(lambda x: len(reports.list_sep.split(x)) if len(x)>0 else 0)

    # check
    if n_mesh > 0:
        assert max(tsv[reports.k_n_major_mesh]) <= n_mesh
        if verbose:
            print(f"SIMPLIFIED '{reports.k_major_mesh}' terms. Max kept: {n_mesh}")
    return tsv


def simplify_auto_terms(tsv, n_terms=3, verbose=False):
    def simplify_list(x):
        # x of the form: t1;t2;t3
        terms = x.split(reports.list_sep)
        if n_terms > 0:
            terms = terms[:n_terms]  # split with ; -> groups of the form t1/t2/t3

        terms = [x.lower().strip() for x in terms]
        out = reports.list_sep.join(terms)
        return out

    tsv.loc[:, reports.k_auto_term] = tsv.loc[:, reports.k_auto_term].apply(lambda x: simplify_list(x))
    tsv.loc[:, reports.k_n_auto_term] = tsv.loc[:, reports.k_auto_term].apply(lambda x: len(reports.list_sep.split(x)) if len(x)>0 else 0)

    if n_terms > 0:
        assert max(tsv[reports.k_n_auto_term]) <= n_terms
        if verbose:
            print(f"SIMPLIFIED '{reports.k_auto_term}', max kept: {n_terms}")
    return tsv


def simplify_groups_of_tags(tsv):
    """
    Many terms (MeSH or auto, mostly MeSH) have the form term[/term]+;term[/term]+... groups of terms separated by /
    The simplification done here keeps only the first term per group, removing all the / and seconday terms
    """
    def simplify(terms):
        values = terms.split(reports.list_sep)
        out = []
        for v in values:
            out.append(v.split("/")[0])
        out = reports.list_sep.join(out)
        # print(f"{terms} ---> {out}")
        return out

    tsv[reports.k_auto_term] = tsv[reports.k_auto_term].apply(lambda x: simplify(x))
    tsv[reports.k_major_mesh] = tsv[reports.k_major_mesh].apply(lambda x: simplify(x))
    # end


    
# latest available zip file contains PNG images - this function is not needed any more
def png2jpg_extensions(tsv, verbose=False):
    tsv[reports.k_image_filename] = tsv[reports.k_image_filename].apply(lambda x: x.replace(".png", ".jpg"))
    if verbose:
        print("EXTENSIONS of image filenames changed from jpg to png")
    return tsv


def remove_commas_in_terms(tsv, verbose=False):
    # remove commas from terms, mesh and auto
    def remove_comma(x, debug=False):
        x = x.strip()
        if len(x) == 0:
            return x

        terms = x.split(reports.list_sep)
        new_terms = []
        comma_found = False
        for g in terms:
            tokens = g.split(",")
            if len(tokens) > 1:
                comma_found = True
                new_terms.append(tokens[0])
            else:
                new_terms.append(g)

        out = reports.list_sep.join([nt.strip() for nt in new_terms])
        if debug and comma_found:
            print(f"COMMA removed: {x} -> {out}")
        return out

    tsv[reports.k_major_mesh] = tsv[reports.k_major_mesh].apply(lambda x: remove_comma(x, debug=False))
    tsv[reports.k_auto_term] = tsv[reports.k_auto_term].apply(lambda x: remove_comma(x, debug=False))

    if verbose:
        print(f"COMMA removed from columns '{reports.k_major_mesh}' and '{reports.k_auto_term}' (comma separates minor terms: only main kept)")
    # end

def fix_no_indexing(tsv, keep_lines, verbose=False):
    iii = tsv[reports.k_major_mesh] == "no indexing"
    jjj = tsv[reports.k_auto_term].str.len() == 0
    kkk = iii & jjj  # reports with major_mesh=="no indexing" and with empty auto_term
    
    if not keep_lines:
        tsv = tsv.drop(tsv[kkk].index)
        if verbose:
            print(f"DROPPED rows with 1) 'no indexing' as '{reports.k_major_mesh}'; and 2) empty '{reports.k_auto_term}': {nnz(kkk)}")
    else:
        # XXX may be change no indexing to unknown
        pass
    # end

def fill_empty_auto_terms(tsv, verbose=False, debug=False):
    iii = tsv[reports.k_auto_term].str.len() == 0
    jjj = tsv[reports.k_major_mesh].str.len() > 0
    kkk = iii & jjj  # indexes of lines with empty auto terms and non-empty major mesh terms

    # use first major_mesh as fill value, for each row
    values = tsv.loc[kkk, reports.k_major_mesh].apply(lambda x: x.split(reports.list_sep)[0])
    
    if debug:
        u_terms = set(values)
        for ut in u_terms:
            print(ut)

    tsv.loc[kkk, reports.k_auto_term] = values
    tsv.loc[kkk, reports.k_n_auto_term] = 1

    # check
    jjj = tsv[reports.k_auto_term].str.len() == 0
    assert nnz(jjj) == ( nnz(iii) - nnz(kkk))
    if verbose:
        print(f"FILLED empty '{reports.k_auto_term}' with first '{reports.k_major_mesh}' term: {nnz(kkk)} ")
        print(f"\t remaining empty '{reports.k_auto_term}': {nnz(jjj)}")
    # end

def make_manual_corrections(tsv, verbose=False, debug=False):
    import os
    # TODO read file with term substitutions: make it a cmd line argument
    fn = "aux_files/auto_term_norm.txt"
    
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

    if debug:
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

        if subst and debug:
            print(f"SUBSTITUTION {s} -> {new_s}")
        return new_s

    tsv[reports.k_auto_term] = tsv[reports.k_auto_term].apply(lambda x: perform_subst(x))


def fix_empty_and_noindexing(tsv, verbose):
    def process_column(s):
        if len(s) == 0 or ( s == "no indexing"):
            return "empty"
        else:
            return s
        
    tsv[reports.k_auto_term] = tsv[reports.k_auto_term].apply(lambda x: process_column(x))
    tsv[reports.k_major_mesh] = tsv[reports.k_major_mesh].apply(lambda x: process_column(x))


def frequency_in_column(col):
    from collections import defaultdict
    counter = defaultdict(int)
    for _, values in col.iteritems():
        for tag in values.split(reports.list_sep):
            counter[tag] = counter[tag] + 1
    counter = dict(
        sorted(counter.items(), key=lambda item: item[1], reverse=True))
    new_counter = dict()
    min_frequency = 10
    for key, value in counter.items():
        if value >= min_frequency:
            new_counter[key] = value
        else:
            break  # ordered, so all the other frequencies are below the threshold

    
    return new_counter

def filter_terms(column, max_terms, verbose=False): # max_terms_per_rep
    def filter_terms_in_col(value, terms_to_accept, term_count, verbose):  # max_terms_per_rep removed
        terms = value.split(reports.list_sep)
        new_terms = []
        append_misc = False
        for t in terms:
            if t in terms_to_accept:
                new_terms.append(t)
            else:
                append_misc = True

        if append_misc:
            new_terms.append("misc")
        
        return reports.list_sep.join(new_terms)
        
    term_count = frequency_in_column(column)
    accept_terms = list(term_count.keys())  # ordered on frequency
    rem = []
    print("** terms:")
    print(accept_terms)
    if len(accept_terms) > max_terms:
        rem = accept_terms[max_terms:]
        accept_terms = accept_terms[:max_terms]
    print(f"terms to remove {len(rem)} ({len(accept_terms)}, max terms: {max_terms})")
    column = column.apply(lambda x: filter_terms_in_col(x, accept_terms, term_count, verbose))
    return column
    
    # tsv[reports.k_auto_term] = tsv[reports.k_auto_term].apply(lambda x: sel_terms(x, auto_count))
from collections import defaultdict

def apply_term_freq_thresh(df, col_name, th):
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
        print(f"{l} occurs {c}")
        if c < th:
            to_remove.append(l)
    
    n_labels_kept = len(cnt) - len(to_remove)
    print(f"{len(to_remove)}/{len(cnt)} labels will be removed, remaining {n_labels_kept}")
    
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
    print("verifying label filtering")
    print(f" {len(c)} == {n_labels_kept + 1}")
    assert len(c) == (n_labels_kept + 1)
    return nc


#********************************************
def main(raw_tsv, out_file, 
        keep_n_imgs=0, 
        n_terms=2, 
        n_terms_per_rep=3,
        fill_normal_impression=True, 
        fill_normal_auto_tag=True, 
        keep_no_indexing=False, 
        manual_corrections=True,
        min_term_frequency=50,
        verbose=False, 
        dev=False):
    """Script for cleaning the values extracted from the xml reports and stored in the file 'raw_tv'
        Process the raw tsv file and creates and new tsv with clean values to be used in further processing stages:
        - remove reports that have no associated images
        - fill as many empty fields as possible
        - simplify both major mesh and auto terms (keep only main tags, remove commas, lowercase text, ...)
        - ...

        The sequence of steps is executed in such an order to minimize the number of reports that are to be dropped. 
        Incomplete reports are dropped after filling as many values as possible and not as soon as the input raw tsv file is read.

        Args:
            raw_tsv (string): path to the raw tsv, built by A00_prepare_raw_tsv.py
            out_fld (string): output folder
            keep_n_imgs (int): number of images to keep in a report. If a report has less images, the first one is repeated. Default: 1.
            n_terms (int): max number of major_mesh and auto_term terms per report to keep. Use 0 to keep all of them. Default: 2.
            fill_normal_impression (boolean): write 'normal' in empty 'impression' fields in reports whose major_mesh is 'normal'. Default: True.
            fill_normal_auto_tag (boolean): write 'normal' in 'auto_tag' for reports whose major mesh is 'normal'. Default: True.
            manual_corrections (boolean): make corrections based on heuristics and 'human' inspection of raw data. For example, some plural forms of the tags are turned into singular. Default: True.
            verbose (boolean): print information summarizing each processing step. Default: False.
            dev (boolean): flag used while developing - DO NOT USE. Default: False.
            
    """
    config = locals()
    tsv = pd.read_csv(config["raw_tsv"], sep=reports.csv_sep, na_filter=False)
    n_input_rows = len(tsv)
    print(f"number of RAW rows in {config['raw_tsv']}: {n_input_rows}")
    
    print(f"input raw tsv file: {raw_tsv}, |rows|: {n_input_rows}")
    # step
    tsv = drop_reports_no_images(tsv, verbose=config["verbose"])
    # step, to lowercase and remove dupliactes (e.g. Term, term -> when lowercased: term, term)
    all_terms_to_lowercase(tsv, verbose=config["verbose"])

    # step
    if config["fill_normal_impression"]:
        fill_empty_impression(tsv, verbose=config["verbose"])

    # step: remove reports with both indication and findinds empty
    tsv = drop_incomplete_reports(tsv, verbose=config["verbose"])

    
    
    # step
    tsv["orig_n_images"] = tsv[reports.k_n_images]
    if False:
        normalize_n_images(tsv, int(config["keep_n_imgs"]), verbose=config["verbose"])

    # step
    if config["fill_normal_auto_tag"]:
        fill_auto_tag(tsv, verbose=config["verbose"])
    
    
    # step
    simplify_groups_of_tags(tsv)
    
    # step
    fill_empty_auto_terms(tsv, verbose=config["verbose"])

    # step:
    remove_commas_in_terms(tsv, verbose=config["verbose"])
    
    # step
    fix_empty_and_noindexing(tsv, verbose=config["verbose"])

    # step
    if config["manual_corrections"]:
        make_manual_corrections(tsv, verbose=config["verbose"])

    
    print("parameters n_terms_per_rep IGNORED")   # config["n_terms_per_rep"], 
    # tsv[reports.k_auto_term] = filter_terms(tsv[reports.k_auto_term], config["n_terms"], verbose=config["verbose"])
    # tsv[reports.k_major_mesh] = filter_terms(tsv[reports.k_major_mesh], config["n_terms"], verbose=config["verbose"])
    



    # after all the previous steps, recompute number of terms
    def count_terms(x):
        if len(x) == 0:
            return 0
        terms = x.split(reports.list_sep)
        return len(terms)

    

    th = config["min_term_frequency"]
    tsv[reports.k_auto_term] = apply_term_freq_thresh(tsv, reports.k_auto_term, th)
    tsv[reports.k_major_mesh] = apply_term_freq_thresh(tsv, reports.k_major_mesh, th)
    tsv[reports.k_n_major_mesh] = tsv[reports.k_major_mesh].apply(lambda x: count_terms(x))  # len(reports.list_sep.split(x)) if len(x)>0 else 0)
    tsv[reports.k_n_auto_term] = tsv[reports.k_auto_term].apply(lambda x: len(x.split(reports.list_sep)) if len(x)>0 else 0)
    

    # assert( max(tsv[reports.k_n_major_mesh] <= config["n_terms_per_rep"]) )
    # assert( max(tsv[reports.k_n_auto_term] <= config["n_terms_per_rep"]) )
    # # step
    # if config["n_terms"] > 0:
    #     simplify_major_mesh(tsv, n_mesh=config["n_terms"], verbose=config["verbose"])

    # # step
    # if config["n_terms"] > 0:
    #     simplify_auto_terms(tsv, n_terms=config["n_terms"], verbose=config["verbose"])

    # step - not needed any more, current zip files contain png images
    # tsv2 = png2jpg_extensions(tsv2, verbose=config["verbose"])

    tsv.to_csv(config["out_file"], sep=reports.csv_sep, index=False)
    print(f"DROPPED rows: {n_input_rows - len(tsv)} (in {n_input_rows} > {len(tsv)} out)")
    print(f"dataset saved at: {config['out_file']}")
#< main

#********************************************
if __name__ == "__main__":
    fire.Fire(main)
