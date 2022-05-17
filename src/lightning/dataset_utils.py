
import numpy as np
import pandas as pd

from pt.dataset import ImageDataset, DataLoader, ImageTransforms, MultiModalDataset, MultiModalDatasetForGeneration
from utils.text_collation import collate_fn_one_s, collate_fn_n_sents

def build_datasets(ds, splits, ds_name, img_fld):
    parts = ["train", "valid", "test"]

    res = {}
    img_transforms = ImageTransforms(dataset=ds_name)
    for i, p in enumerate(parts):
        subset = ds.loc[splits.filename[splits.split == p]]
        trans = img_transforms.test_transforms
        if p == "train":
            trans = trans = img_transforms.train_transforms
        dataset = ImageDataset(subset, img_fld, trans)
        res[p] = dataset
    #<
    return res["train"], res["valid"], res["test"]

def build_dataloaders(train, valid, test, bs=32, n_workers=16, shuffle=True, pin_memory=False):
    splits = [train, valid, test]
    droplast = [False, True, True]
    res = []
    for split, dl in zip(splits, droplast):
        res.append( DataLoader(split, batch_size=bs, shuffle=shuffle, num_workers=n_workers, drop_last=dl, pin_memory=False) )
    # train_dl = DataLoader(train, batch_size=bs, shuffle=shuffle, num_workers=n_workers)
    # valid_dl = DataLoader(valid, batch_size=bs, shuffle=False, num_workers=n_workers)
    # test_dl = DataLoader(test, batch_size=bs, shuffle=False, num_workers=n_workers)
    # return train_dl, valid_dl, test_dl
    return res[0], res[1], res[2]


def build_mm_datasets(ds, text_ds, splits, ds_name, img_fld, n_sentences=1, max_tokens=12, collate_fn=collate_fn_one_s, img_size=224, for_generation=False):
    parts = ["train", "valid", "test"]

    res = {}
    img_transforms = ImageTransforms(dataset=ds_name)
    for i, p in enumerate(parts):
        subset = ds.loc[splits.filename[splits.split == p]]
        trans = img_transforms.test_transforms
        if p == "train":
            trans = trans = img_transforms.train_transforms
        if not for_generation:
            dataset = MultiModalDataset(subset, text_ds, img_fld, img_transforms=trans,
                n_classes=None, img_size=img_size, n_sentences=n_sentences, n_tokens=max_tokens, collate_fn=collate_fn, verbose=False)
        else:
            dataset = MultiModalDatasetForGeneration(subset, text_ds, img_fld, img_transforms=trans,
                n_classes=None, img_size=img_size, n_sentences=n_sentences, n_tokens=max_tokens, collate_fn=collate_fn, verbose=False)
        res[p] = dataset
    #<
    return res["train"], res["valid"], res["test"]

# def build_mm_dataloaders(train, valid, test, bs=32, n_workers=16, shuffle=True):
#     splits = [train, valid, test]
#     droplast = [False, True, True]
#     res = []
#     for split, dl in zip(splits, droplast):
#         res.append( DataLoader(split, batch_size=bs, shuffle=shuffle, num_workers=n_workers, drop_last=dl) )
#     # train_dl = DataLoader(train, batch_size=bs, shuffle=shuffle, num_workers=n_workers)
#     # valid_dl = DataLoader(valid, batch_size=bs, shuffle=False, num_workers=n_workers)
#     # test_dl = DataLoader(test, batch_size=bs, shuffle=False, num_workers=n_workers)
#     # return train_dl, valid_dl, test_dl
#     return res[0], res[1], res[2]
