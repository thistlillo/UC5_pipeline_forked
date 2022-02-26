import fire
from collections import OrderedDict
import os
import pandas as pd
from posixpath import join
import pyecvl.ecvl as ecvl
import yaml

from eddl_lib.eddl_augmentations import train_augs, test_augs
from text.reports import list_sep, csv_sep
from text.vocabulary import Vocabulary
from text.encoding import SimpleCollator
from utils.data_partitioning import load_data_split, DataPartitioner

collator = SimpleCollator()


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
        train_ids = train_ids[:10]
        valid_ids = valid_ids[:10]
        test_ids = test_ids[:10]
        ds = ds[ds.filename.isin(train_ids) | ds.filename.isin(valid_ids) | ds.filename.isin(test_ids)]
        print(len(train_ids))

    n_train = len(train_ids)
    n_valid = len(valid_ids)
    n_test = len(test_ids)

    d = {
        "name"        : "ECVL dataset for UC5",
        "description" : description,
        "classes"     : list( set([int(l) for y in ds.labels.tolist() for l in y.split(list_sep)]) ),
        "images"      : [],
        "split"       : dict(training = list(range(n_train)), 
                            validation = list(range(n_train, n_train + n_valid)), 
                            test=list(range(n_train + n_valid, len(ds))))
    }

    imgs = []
    for ids in [train_ids, valid_ids, test_ids]:
        imgs += [{
                'location': os.path.abspath(join(img_fld, x[0])),
                'label': [int(l.strip()) for l in x[1].split(list_sep)],
                'values': collator.parse_and_collate(x[2], n_tokens),
                } 
                for x in ds.loc[ds.filename.isin(ids), ["filename", "labels", "enc_text"]].values
            ]
    d["images"] = imgs
    print(yaml.dump(d, default_flow_style=None))
    
    with open(out_fn, "w") as fout:
        yaml.safe_dump(d, fout, default_flow_style=None)

    print(f"dataset in: {out_fn}")

    # augs = ecvl.DatasetAugmentations(augs=[train_augs(img_size), test_augs(img_size), test_augs(img_size)])
    augs = ecvl.DatasetAugmentations(augs=[train_augs, test_augs, test_augs])
    drop_last = {"training": False, "validation": False, "test": False}
    dataset = ecvl.DLDataset(out_fn, batch_size=32, augs=augs, ctype=ecvl.ColorType.RGB, ctype_gt=ecvl.ColorType.GRAY, num_workers=2, queue_ratio_size=2, drop_last=drop_last)
    
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
    _, b, c = dataset.GetBatch()
    dataset.Stop()
    print("done.")

# --------------------------------------------------
if __name__ == "__main__":
    fire.Fire(main)