# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
from bs4 import BeautifulSoup
from collections import defaultdict
import fire
import glob
import os
from posixpath import join
from tqdm import tqdm

import utils.misc as mu
import text.reports as reports


def parse_single_report(filepath, verbose=False):
    with open(filepath, "r", encoding="utf-8") as fin:
        xml = fin.read()

    soup = BeautifulSoup(xml, "lxml")
    d = reports.parse_soup(soup)
    d["filename"] = mu.filename_from_path(filepath)
    if verbose:
        print(f"File {fn} parsed, results:")
        for k in d.keys():
            print(f"Key {k}: {d[k]}")
            if k == reports.k_images:
                for le in d[k]:
                    print(le)
    return d

def parse_reports(txt_fld, ext="xml", verbose=False, dev=False):
    file_list = sorted(mu.list_files(txt_fld, ext))
    for i, fn in enumerate(tqdm(file_list, desc="xml -> report")):
        if dev and i == 20:
            break
        yield parse_single_report( join(txt_fld, fn) )


def build_report_header():
    # see build_report_line for the correct order of column names
    header = [
              reports.k_filename,
              reports.k_identifier,
              reports.k_n_major_mesh,
              reports.k_major_mesh,
              reports.k_n_auto_term,
              reports.k_auto_term,
              reports.k_n_images,
              reports.k_image_filename,
              reports.k_indication,
              reports.k_findings,
              reports.k_impression,
              ]
    header = reports.csv_sep.join(header)
    return header

def build_report_line(parsed):
    cols = []
    p = parsed # just a shortcut
    cols.append(p["filename"])
    cols.append(p[reports.k_identifier])
    cols.append(str(len(p[reports.k_major_mesh])))
    cols.append(reports.list_sep.join(p[reports.k_major_mesh]))
    cols.append(str(len(p[reports.k_auto_term])))
    cols.append(reports.list_sep.join(p[reports.k_auto_term]))
    img_filenames = [d[reports.k_image_filename] for d in p[reports.k_images]]
    cols.append(str(len(img_filenames)))
    cols.append(reports.list_sep.join(img_filenames))
    cols.append(p[reports.k_indication])
    cols.append(p[reports.k_findings])
    cols.append(p[reports.k_impression])
    # build_report_line.lineno += 1
    return reports.csv_sep.join(cols)

def main(
    txt_fld,  # folder containing the xml reports
    img_fld,  # folder containing the images
    out_file, # path for the output file
    verbose=False, 
    stats=False,  # set to print some stats in csv format on the stdout
    dev=False):  # set to reduce the number of reports processed
    config = locals()
    
    reps = parse_reports(config["txt_fld"], verbose=config["verbose"], dev=config["dev"])
    counter = defaultdict(int)
    with mu.Timer("parsing of xml reports") as _:
        with open( config["out_file"], "w", encoding="utf-8") as fout_r:
            fout_r.write(build_report_header() + "\n")
            for r in reps:
                counter[len(r[reports.k_images])] += 1
                fout_r.write(build_report_line(r) + "\n")
    if counter[0] > 0:
        print(f"\nWARNING! {counter[0]} reports are without images\n")
    print(f"Saved: {config['out_file']}")

    
    if config["stats"]:
        print("Printing some stats (csv format)\n")
        #for k, v in sorted(counter.items()):
        #    print(f"reports with {k} images: {v}")
        print("n_images, count")
        cs_lines = "\n".join([f"{k}, {v}" for k,v in sorted(counter.items())])
        print(cs_lines)


# ************************
if __name__ == "__main__":
    fire.Fire(main)

#< end of file