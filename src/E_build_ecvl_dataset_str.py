import fire
import os
import pandas as pd
from posixpath import join
import pyecvl.ecvl as ecvl

from eddl_lib.eddl_augmentations import train_augs, test_augs
from text.reports import list_sep, csv_sep
from text.vocabulary import Vocabulary
from text.encoding import SimpleCollator
from utils.data_partitioning import load_data_split, DataPartitioner

collator = SimpleCollator()

def extract_lists(ds, ex_ids):
    df = ds.loc[ds.filename.isin(ex_ids)]
    return df.filename.tolist(), df.labels.tolist(), df.enc_text.tolist()

def labs_repr(ll):
    ll = ll.split(list_sep)
    ll = [int(l.strip()) for l in ll]
    return repr(ll)
# <

def txt_repr(txt, n_tokens):
    txt = collator.parse_and_collate(txt, n_tokens)
    return repr(txt)

def to_yaml(filenames, labels, texts, img_fld, n_tokens):   
    out = ""
    for fn, labs, txt in zip(filenames, labels, texts):
        labs = labs_repr(labs)
        txt = txt_repr(txt, n_tokens)
        out += f"- location: {join(img_fld, fn)}\n"  # os.path.abspath( 
        out += f"  label: {labs}\n"
        out += f"  values: {txt}\n"
    return out


def main(out_fn, in_tsv, exp_fld, img_fld, n_tokens, img_size=224, description="none provided", seed=None, shuffle_seed=None, add_text=True, dev=False):
    config = locals()
    if dev:
        print("build ecvl dataset, actual parameters:")
        for k, v in config.items():
            print(f" - {k}: {v}")
    
    ds = pd.read_csv(in_tsv, sep=csv_sep)
    if dev:
        print(f"data read, shape: {ds.shape}")
        print(f"  columns: {ds.columns}")

    train_ids, valid_ids, test_ids = load_data_split(exp_fld)

    if dev:
        keep = 3
        train_ids = train_ids[:keep]
        valid_ids = valid_ids[:keep]
        test_ids = test_ids[:keep]
        ds = ds[ds.filename.isin(train_ids) | ds.filename.isin(valid_ids) | ds.filename.isin(test_ids)]


    with open(out_fn, "w") as fout:
        fout.write("name: ECVL dataset for UC5\n")
        fout.write("\n")
        fout.write(f"description: {description}\n")
        fout.write("\n")
        classes = repr(list( set([int(l) for y in ds.labels.tolist() for l in y.split(list_sep)]) ))
        fout.write(f"classes: {classes}\n")
        fout.write("\nimages:\n")

        for ids in [train_ids, valid_ids, test_ids]:
            filenames, labels, texts = extract_lists(ds, ids)
            yml_str = to_yaml(filenames, labels, texts, img_fld, n_tokens)
            if dev:
                print(yml_str)
            fout.write(yml_str)
        n_train = len(train_ids)
        n_valid = len(valid_ids)
        n_test  = len(test_ids)
        train_split = list(range(0, n_train))
        valid_split = list(range(n_train, n_train + n_valid))
        test_split = list(range(n_train + n_valid, len(ds)))
        split_str = ""
        split_str += "split:\n"
        split_str += f"  training: {repr(train_split)}\n"
        split_str += f"  validation: {repr(valid_split)}\n"
        split_str += f"  test: {repr(test_split)}\n"
        if dev:
            print(split_str)
        fout.write(split_str)
    
    print(f"dataset in: {out_fn}")
    
    
    # augs = ecvl.DatasetAugmentations(augs=[train_augs(img_size), test_augs(img_size), test_augs(img_size)])
    augs = ecvl.DatasetAugmentations(augs=[train_augs, test_augs, test_augs])
    print(type(train_augs))
    drop_last = {"training": False, "validation": False, "test": False}

    dataset = ecvl.DLDataset(out_fn, batch_size=1, augs=augs, ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, num_workers=2, queue_ratio_size=2, drop_last=drop_last)
    dataset.SetSplit(ecvl.SplitType.training)
    print(f"training, n_ex: {len(dataset.GetSplit())}")
    assert len(train_ids) == len(dataset.GetSplit())
    dataset.SetSplit(ecvl.SplitType.test)
    print(f"test, {len(dataset.GetSplit())}")
    assert len(test_ids) == len(dataset.GetSplit())
    dataset.SetSplit(ecvl.SplitType.validation)
    print(f"validation, {len(dataset.GetSplit())}")
    assert len(valid_ids) == len(dataset.GetSplit())

    dataset.SetSplit(ecvl.SplitType.training)
    dataset.ResetAllBatches(shuffle=True)
    dataset.Start()
    a, b, c = dataset.GetBatch()
    dataset.Stop()
    print("done.")

# --------------------------------------------------
if __name__ == "__main__":
    fire.Fire(main)