import json
import numpy as np
import os
import pandas as pd
import pickle
from posixpath import join
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from utils.data_partitioning import DataPartitioner
import text.reports as reports
from text.encoding import SimpleCollator, StandardCollator
from pt.uc5_dataset import Uc5ImgDataset
import utils.misc as mu

class Uc5DataModule(LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        
        self.l1normalization = True

        self.in_tsv = conf["in_tsv"]
        self.img_fld = conf["img_fld"]
        self.exp_fld = conf["exp_fld"]
        self.img_size = conf["img_size"]

        self.tsv = pd.read_csv(self.in_tsv, sep=reports.csv_sep, na_filter=False)
        self.tsv.set_index("filename", inplace=True, drop=False)
        # NOTICE: set_index to filename, with drop=True
        
        self.train_dl, self.val_dl, self.test_dl = None, None, None   # cache of the dataloaders, not used - code commented
        self.train_ids, self.val_ids, self.test_ids = self._set_partitions()
                
        # section: load files
        #   read all the files here to reduce memory footprint
        with open(join(self.exp_fld, "lab2index.json"), "r") as fin:
            self.label2index = json.load(fin)
        self.n_classes = len(self.label2index)

        with open(join(self.exp_fld, "index2lab.json"), "r") as fin:
            self.index2label = json.load(fin)
        
        # add this column, that will be read and served by the uc5_dataset receiving the view of this tsv
        self.tsv["one_hot_labels"] = self.tsv["labels"].apply(
            lambda x: mu.encode_labels_one_hot(
                [int(l) for l in x.split(reports.list_sep)], 
                n_classes = self.n_classes,
                l1normalization=self.l1normalization)
                )

        #   this new column contains the encoding of the labels, ready for training
        # self.tsv["int_labels"] = self.tsv["labels"].apply(lambda x: self.encode_labels([int(l) for l in x.split(reports.list_sep)]))

        # self.vocab = None
        # with open(join(self.exp_fld, "vocab.pickle"), "rb") as fin:
        #     self.vocab = pickle.load(fin)
        # # < section: load files

        # # section: set text encoding
        # self.n_sentences = int(conf["n_sentences"])
        # self.sentence_length = int(conf["n_tokens"])
        # if self.version == "simple":
        #     self.n_sentences = 1
        #     self.collator = SimpleCollator()
        # else:
        #     self.collator = StandardCollator()       
        # # < section: set text encoding


        # < init
    
    

    # section: partitions
    def _set_partitions(self):
        if self.conf["load_data_split"]:
            return self._load_data_split()
        else:
            return self._create_partitions()
        
    def _load_data_split(self):
        with open(join(self.exp_fld, "train_ids.txt"), "r", encoding="utf-8") as fin:
            train_ids = [line.strip() for line in fin.readlines()]
        with open(join(self.exp_fld, "valid_ids.txt"), "r", encoding="utf-8") as fin:
            valid_ids = [line.strip() for line in fin.readlines()]
        with open(join(self.exp_fld, "test_ids.txt"), "r", encoding="utf-8") as fin:
            test_ids = [line.strip() for line in fin.readlines()]
        print(f"data split read from disk. |train|={len(train_ids)}, |valid|={len(valid_ids)}, |test|={len(test_ids)}")
        return train_ids, valid_ids, test_ids

    def _create_partitions(self):
        c = {}
        c["in_tsv"] = self.in_tsv
        c["exp_fld"] = self.exp_fld
        c["train_p"] = self.conf["train_p"]
        c["valid_p"] = self.conf["valid_p"]
        c["shuffle_seed"] = self.conf["shuffle_seed"]
        c["term_column"] = self.conf["term_column"]
        c["verbose"] = self.conf["verbose"]
        partitioner = DataPartitioner(c, self.tsv)   
        return partitioner.partition()
    # < section: partitions

    # section: uc5 datasets
    def _filter_tsv_for_split(self, ids):
        #print("loading split")
        #print(f"length of ids: {len(ids)}")
        subdf = self.tsv[self.tsv.filename.isin(ids)]  # .reset_index(drop=True)
        #print(subdf.columns)
        #print(subdf.shape)
        return subdf
        
    # < section: uc5 datasets

    # section: pt-lightning methods
    def train_dataloader(self):
        if self.conf["verbose"]:
            print("returning train_dataloader")

        train_dataset = Uc5ImgDataset(
            tsv=self._filter_tsv_for_split(self.train_ids),
            n_classes=self.n_classes,
            conf=self.conf,
            version=None)
        print(f"train dataloader using {self.conf['loader_threads']} loader threads")
        return DataLoader(train_dataset, batch_size=self.conf["batch_size"], num_workers=self.conf["loader_threads"]) # , 

    # def train_dataloader(self):
    #     if self.conf["verbose"]:
    #         print("returning train_dataloader")

    #     if self.train_dl is not None:
    #         return self.train_dl
        
    #     train_dataset = Uc5ImgDataset(
    #         tsv=self._filter_tsv_for_split(self.train_ids),
    #         conf=self.conf,
    #         version=None)
    #     self.train_dl = DataLoader(train_dataset, batch_size=self.conf["batch_size"]) # , num_workers=self.conf["loader_threads"]
    #     return self.train_dl
        
    def val_dataloader(self):
        if self.conf["verbose"]:
            print("returning val_dataloader")
        
        val_dataset = Uc5ImgDataset(
            tsv=self._filter_tsv_for_split(self.val_ids),
            n_classes=self.n_classes,
            conf=self.conf,
            version=None)
        print(f"val dataloader using {self.conf['loader_threads']} loader threads")
        return DataLoader(val_dataset, batch_size=self.conf["batch_size"], num_workers=self.conf["loader_threads"]) # , num_workers=self.conf["loader_threads"]
    

    # def val_dataloader(self):
    #     if self.conf["verbose"]:
    #         print("returning val_dataloader")
        
    #     if self.val_dl is not None:
    #         return self.val_dl

    #     val_dataset = Uc5ImgDataset(
    #         tsv=self._filter_tsv_for_split(self.val_ids),
    #         conf=self.conf,
    #         version=None)
    #     self.val_dl = DataLoader(val_dataset, batch_size=self.conf["batch_size"]) # , num_workers=self.conf["loader_threads"]
    #     return self.val_dl

    def test_dataloader(self):
        if self.conf["verbose"]:
            print("returning test_dataloader")
                
        test_dataset = Uc5ImgDataset(
            tsv=self._filter_tsv_for_split(self.test_ids),
            n_classes=self.n_classes,
            conf=self.conf,
            version=None)
        print(f"test dataloader using {self.conf['loader_threads']} loader threads")
        return DataLoader(test_dataset, batch_size=self.conf["batch_size"], num_workers=self.conf["loader_threads"]) # , num_workers=self.conf["loader_threads"]
    
    # def test_dataloader(self):
    #     if self.conf["verbose"]:
    #         print("returning test_dataloader")
        
    #     if self.test_dl is not None:
    #         return self.test_dl
        
    #     test_dataset = Uc5ImgDataset(
    #         tsv=self._filter_tsv_for_split(self.test_ids),
    #         conf=self.conf,
    #         version=None)
    #     self.test_dl = DataLoader(test_dataset, batch_size=self.conf["batch_size"]) # , num_workers=self.conf["loader_threads"]
    #     return self.test_dl

    # section: pt-lightning methods

# USED ONLY FOR TESTING
def main(in_tsv,
         exp_fld,
         img_fld,
         out_fn="uc5model_default.bin",
         only_images=False,
         train_p=0.7,
         valid_p=0.1,
         seed=1,
         shuffle_seed=2,
         term_column="auto_term",
         text_column="text",
         img_size = 224,
         batch_size=32,
         last_batch="random",
         n_epochs=50,
         n_sentences=5,  # ignored by simple
         n_tokens=10,
         eddl_cs_mem="mid_mem",
         eddl_cs="cpu",
         sgl_lr=0.09,
         sgd_momentum=0.9,
         lstm_h_size=512,
         emb_size=512,
         load_data_split=True,
         preload_images = True,
         verbose=False,
         debug=False,
         dev=False):
    config = locals()
    datamod = Uc5DataModule(config)
    train_dl = datamod.train_dataloader()
    val_dl = datamod.val_dataloader()
    test_dl = datamod.test_dataloader()

    print(f"|train_dataloader|= {len(train_dl)}")
    print(f"|val_dataloader|= {len(val_dl)}")
    print(f"|test_dataloader|= {len(test_dl)}")
    