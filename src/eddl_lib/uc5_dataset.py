# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import fire
import json
import math
import numpy as np
import pandas as pd
import pickle
from posixpath import join
import random

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl

import text.reports as reports
from text.encoding import SimpleCollator, StandardCollator
from C00_split import DataPartitioner

from utils.misc import Timer

class Uc5Dataset:
    VERBOSE_NONE = 0
    VERBOSE_INFO = 1
    VERBOSE_DEBUG = 2

    def __init__(self, conf, version="simple"):
        self.conf = conf
        self.version = version
        self.only_images = False # conf["only_images"]  # return only images, without text
        
        if "cnn_out_layer" in self.conf.keys():
            self.l1normalization = (self.conf["cnn_out_layer"] == "softmax")
        else:
            print("cnn out type not found in configuration, l1 normalization set to False")
            self.l1normalization = False
        
        self.dev_grayscale = False
        # verbose level
        self.verbose = Uc5Dataset.VERBOSE_NONE
        if conf["debug"]:
            self.verbose = Uc5Dataset.VERBOSE_DEBUG
        elif conf["verbose"]:
            self.verbose = Uc5Dataset.VERBOSE_INFO

        # folders
        self.in_tsv = conf["in_tsv"]
        self.img_fld = conf["img_fld"]
        self.exp_fld = conf["exp_fld"]

        self.img_size = conf["img_size"]
        

        # parameters
        self.load_data_split = conf["load_data_split"]

        self.seed = int(conf["shuffle_seed"])        
        self.random = random.Random()
        self.random.seed(self.seed)
        if self.verbose > Uc5Dataset.VERBOSE_NONE:
            print(f"Using shuffling seed: {self.seed}")
        
        self.train_p = float(conf["train_p"])
        self.valid_p = float(conf["valid_p"])
        self.test_p = 1 - ( self.train_p + self.valid_p)
        assert self.test_p > 0

        # input files
        self.tsv = pd.read_csv(self.in_tsv, sep=reports.csv_sep, na_filter=False)
        self.tsv.set_index("filename", inplace=True, drop=False)
        print(f"input dataset read. index: {self.tsv.index.name}")

        with open(join(self.exp_fld, "lab2index.json"), "r") as fin:
            self.label2index = json.load(fin)
        self.n_classes = len(self.label2index)

        with open(join(self.exp_fld, "index2lab.json"), "r") as fin:
            self.index2label = json.load(fin)

        # this new column hosts the encoding of the labels, ready for training
        self.tsv["int_labels"] = self.tsv["labels"].apply(lambda x: self.encode_labels([int(l) for l in x.split(reports.list_sep)]))

        self.vocab = None
        with open(join(self.exp_fld, "vocab.pickle"), "rb") as fin:
            self.vocab = pickle.load(fin)

        if self.verbose > Uc5Dataset.VERBOSE_NONE:
            print(f"number of image labels: {self.n_classes}")

        # text collation
        self.n_sentences = int(conf["n_sentences"])
        self.sentence_length = int(conf["n_tokens"])
        if self.version == "simple":
            self.n_sentences = 1
            self.collator = SimpleCollator()
        else:
            self.collator = StandardCollator()        

        # training related        
        self.stage = None
        self.n_batches = None
        self.stage_ids = None
        self.batch_size = conf["batch_size"]

        self.last_batch = conf["last_batch"]
        assert self.last_batch in ["drop", "random"]
        
        # partitions
        self.train_ids, self.valid_ids, self.test_ids = self.set_partitions_()

        # image preprocessing
        self.augs = ecvl.SequentialAugmentationContainer([
                ecvl.AugDivBy255(),  #  ecvl.AugToFloat32(divisor=255.0),
                ecvl.AugNormalize(122.75405603 / 255.0, 0.296964375 / 255.0),
                ecvl.AugResizeDim([320, 320]),
                # ecvl.AugCenterCrop([256, 256]),  # XXX should be parametric, for resnet 18
                # ecvl.AugCenterCrop([self.img_size, self.img_size]),  # XXX should be parametric, for resnet 18
                ecvl.AugRandomCrop([self.img_size, self.img_size]),  # XXX should be parametric, for resnet 18
                ])
        ecvl.AugmentationParam.SetSeed(self.seed)

        self.preproc_images = None
        if self.conf["preload_images"]:
            self.preproc_images = self.preload_images() 
    # >init

    def preload_images(self):
        print("preloading images... ")
        with Timer("full image dataset loaded") as _:
            with open( self.conf["preproc_images"] , "rb") as fin:
                print(f"loading preprocessed images from: {self.conf['preproc_images']}")
                return pickle.load(fin)

    def set_partitions_(self):
        if self.load_data_split:
            print("pre loading partitions of examples")
            return self.load_partitions_()
        else:
            return self.create_partitions_()
    
    def load_partitions_(self):
        with open(join(self.exp_fld, "train_ids.txt"), "r", encoding="utf-8") as fin:
            train_ids = [line.strip() for line in fin.readlines()]
        with open(join(self.exp_fld, "valid_ids.txt"), "r", encoding="utf-8") as fin:
            valid_ids = [line.strip() for line in fin.readlines()]
        with open(join(self.exp_fld, "test_ids.txt"), "r", encoding="utf-8") as fin:
            test_ids = [line.strip() for line in fin.readlines()]
        print("train-valid-test partitions have been loaded")
        print(f"|training|= {len(train_ids)}")
        print(f"|validation|= {len(valid_ids)}")
        print(f"|test|= {len(test_ids)}")

        return train_ids, valid_ids, test_ids

    def create_partitions_(self):
        c = {}
        c["in_tsv"] = self.in_tsv
        c["exp_fld"] = self.exp_fld
        c["train_p"] = self.train_p
        c["valid_p"] = self.valid_p
        c["shuffle_seed"] = self.seed
        c["term_column"] = self.conf["term_column"]
        c["verbose"] = self.verbose
        self.partitioner = DataPartitioner(c, self.tsv)
        
        return self.partitioner.partition()

    def set_stage(self, s):
        assert s in ["train", "valid", "test"]

        if s == "train":
            self.stage = s
            self.stage_ids = self.train_ids
        elif s == "valid":
            self.stage = "valid"
            self.stage_ids = self.valid_ids
        else:
            self.stage = "test"
            self.stage_ids = self.test_ids
            print(f"Test stage. id list, length: {len(self.stage_ids)}")

    def shuffle(self):
        self.random.shuffle(self.stage_ids)

    def __len__(self):
        idxs = self.test_ids
        if self.stage == "train":
            idxs = self.train_ids
        elif self.stage == "valid":
            idxs = self.valid_ids

        self.n_batches = len(idxs) // self.batch_size
        #print(f"{self.stage}:{len(idxs)} examples - number of batches: {self.n_batches} - bs: {self.batch_size}")
        #print(f"\t\t last batch [{ (self.n_batches-1) * self.batch_size} - {self.n_batches * self.batch_size - 1}]")

        #self.n_batches += min(1, self.n_batches % self.batch_size)

        # the code below is NOT USED: the first images in the dataset are used for filling
        #    the last batch
        print(f"last batch policy: {self.last_batch}")
        if (self.last_batch == "random") and ( (len(idxs) % self.batch_size) > 0 ):
            self.n_batches += 1
            if self.verbose > Uc5Dataset.VERBOSE_INFO:
                print(f"{self.stage} stage: last batch with {self.n_batches % self.batch_size} items (less than batch size {self.batch_size})")
        
        #print(f"{self.stage}:{len(idxs)} examples - number of batches: {self.n_batches} - bs: {self.batch_size}, mul: {self.n_batches*self.batch_size}")
        #print(f"\t\t last batch [{ (self.n_batches-1) * self.batch_size} - {self.n_batches * self.batch_size - 1}]")

        return self.n_batches


    def __getitem__(self, batch_idx):
        if self.only_images:
            return self.getitem_only_images(batch_idx)
        elif self.version == "simple":
            return self.getitem_simple(batch_idx)
        else:
            return self.getitem_complex(batch_idx)

    def load_image_from_disk(self, img_id):
        fn = join(self.img_fld, img_id)
        flags = None
        if self.dev_grayscale:
            flags = flags=ecvl.ImReadMode.GRAYSCALE
        img = ecvl.ImRead(fn, flags=flags)  # , flags=ecvl.ImReadMode.GRAYSCALE)
        self.augs.Apply(img)
        ecvl.RearrangeChannels(img, img, "cxy")
        return img

        
    def get_image(self, img_id):
        if self.preproc_images is not None:
            img = self.preproc_images[img_id]
            if self.dev_grayscale:
                img = np.expand_dims(img[0, :, :], axis=0)
                # print(f"GRAYSCALE? {self.dev_grayscale}, {img.shape}")
        else:
            img = self.load_image_from_disk(img_id) 
        return img   

    # XXX rename all get_image to get_batch
    def getitem_only_images(self, batch_idx):
        n_channels = 3
        if self.dev_grayscale:
            n_channels = 1
        images = np.zeros( (self.batch_size, n_channels, self.img_size, self.img_size), dtype=float)
        labels = np.zeros( (self.batch_size, self.n_classes), dtype=float )
        
        for i in range(self.batch_size):
            idx = (self.batch_size * batch_idx + i) % len(self.stage_ids)
            img_id = self.stage_ids[idx]

            img = self.get_image(img_id) 
            
            # UNCOMMENT THIS CODE FOR RANDOM CHOICE IN LAST BATCH
            # if idx < len(self.stage_ids):
            #     img_id = self.stage_ids[idx]
            # elif self.last_batch == "random":
            #     img_id = self.random.choice(self.stage_ids)
            # else:
            #     print("FAILURE - see dataset_v_img get_image_item")
            #     exit(1)
            

            # REMOVED TO AVOID A METHOD CALL
            # img, lab = self.load_id_only_images(img_id)
        
            # numpy_image = np.array(img, copy=False)
            images[i, :, :, :] = img
            labels[i, :] = self.tsv.loc[img_id, "int_labels"]
            
        return images, labels

    # ONE SENTENCE WITH MAX_TOKENS TOKENS
    def getitem_simple(self, batch_idx):
        n_channels = 3
        if self.dev_grayscale:
            n_channels = 1

        images = np.zeros( (self.batch_size, n_channels, self.img_size, self.img_size), dtype=float)
        labels = np.zeros( (self.batch_size, self.n_classes), dtype=float )
        texts = np.zeros( (self.batch_size, self.sentence_length), dtype=float ) # simple vs complex: changes dimension of this
        
        for i in range(self.batch_size):
            idx = self.batch_size * batch_idx + i
            
            # useless control
            if idx < len(self.stage_ids):
                r_id = self.stage_ids[idx]
            elif self.last_batch == "random":
                r_id = self.random.choice(self.stage_ids)
                if self.verbose > self.VERBOSE_INFO:
                    print(f"in {self.stage} stage, last batch: reusing id {r_id}")
            else:
                # NEVER HERE
                print(f"index in batch: {i}, index in entire split: {idx}, current split size: {len(self.stage_ids)} - last batch policy: {self.last_batch}")
                print("FAILURE - see dataset __get_item__*")
                assert False   

            # load data
            
            img, lab, e_text = self.load_id(r_id)
            
            numpy_image = np.array(img, copy=False)
            images[i, :, :, :] = numpy_image
            labels[i, :] = lab
            
            texts[i, :] = np.squeeze(e_text)

        return images, labels, texts


    def getitem_complex(self, batch_idx):
        if batch_idx == len(self)-1:
            bs = len(self) % self.batch_size
        else:
            bs = self.batch_size

        n_channels = 3
        if self.dev_grayscale:
            n_channels = 1
        images = np.zeros( (bs, n_channels, self.img_size, self.img_size), dtype=float)
        labels = np.zeros( (bs, self.n_classes), dtype=float )
        texts = np.zeros( (bs, self.n_sentences, self.sentence_length))  # +2 because of bos, eos

        for i in range(bs):
            idx = self.batch_size * batch_idx + i
            if idx < len(self.stage_ids):
                r_id = self.stage_ids[idx]
            elif self.last_batch == "random":
                r_id = self.random.choice(self.stage_ids)
                if self.verbose > self.VERBOSE_INFO:
                    print(f"in {self.stage} stage, last batch: reusing id {r_id}")
            else:
                # NEVER HERE
                print("FAILURE - see dataset __get_item__*")
                assert False

            # load data
            r_id = self.stage_ids[idx]
            img, lab, text = self.load_id(r_id)
            numpy_image = np.array(img, copy=False)
            images[i, :, :, :] = numpy_image
            labels[i, :] = lab
            texts[i, :, :] = text
        return images, labels, texts


    def load_id(self, img_fn):
        # image names
        text_col = "enc_" + self.conf["text_column"]
        data = self.tsv.loc[img_fn, ["labels", text_col]]
        labels = data["labels"]  # column was label
        labels = [int(l) for l in labels.split(reports.list_sep)]
        text = data["enc_" + self.conf["text_column"]]

        # if self.verbose > Uc5Dataset.VERBOSE_INFO:
        #     print("loading data, id:", img_fn)
        #     # label_names = " / ".join([self.index2label[i] for i in labels.split(reports.list_sep)])
        #     label_names = " / ".join([self.index2label[str(i)] for i in labels])
        #     print(f"\t - labels: {labels} ({label_names})")
        #     print(f"\t - text (length: {len(text)}): {text[:min(50, len(text))]}...")

        img = self.get_image(img_fn)
        # encoding
        labels = self.encode_labels(labels)
        text = self.encode_text(text)

        return img, labels, text

    # NOT USED IN CURRENT IMPLEMENTATION
    def load_id_only_images(self, img_fn):
        # image names
        labels = self.tsv.loc[img_fn, "int_labels"]
        # labels = [int(l) for l in labels.split(reports.list_sep)]

        if False and self.verbose > Uc5Dataset.VERBOSE_INFO:
            print("loading data, id:", img_fn)
            # label_names = " / ".join([self.index2label[i] for i in labels.split(reports.list_sep)])
            label_names = " / ".join([self.index2label[str(i)] for i in labels])
            print(f"\t - labels: {labels} ({label_names})")
        
        fn = join(self.img_fld, img_fn)
        img = ecvl.ImRead(fn)  # , flags=ecvl.ImReadMode.GRAYSCALE)
        self.augs.Apply(img)
        ecvl.RearrangeChannels(img, img, "cxy")
        # img = self.load_images([img_fn])[0]  # load_images expects a list of filenames, here we are using only one image (see above: img_fn=...)
        # encoding
        # labels = self.encode_labels(labels)
        return img, labels


    def load_images(self, filenames):
        out = []
        for fn in filenames:
            out.append(self.get_image(fn))
        return out


    def encode_labels(self, labels):
        out = np.zeros((1, self.n_classes), dtype=float)
        n_labels = len(labels)

        out[0, labels] = 1.0
        # for l in labels:
        #     out[0, l] = 1.0

        # not used with sigmoid output
        if self.l1normalization:
            out = out / n_labels
        return out


    def encode_text(self, text):
        c = self.conf

        n_sentences = self.n_sentences        
        sentence_length = self.sentence_length

        # +2 for begin-of-sentence and end-of-sentence tokens
        out = np.zeros((1, n_sentences, sentence_length), dtype=float)
        # result will differ depending on the specific collator
        collated_text = self.collator.parse_and_collate(text, sentence_length, n_sentences, pad=True)
        enc = np.array(collated_text)
        out[0, :, :] = enc
        return out


    # def encode_text_OLD(self, text):
    #     if self.version == "simple":
    #         return self.encode_text_simple(text)

    #     c = self.conf
    #     n_sentences = c["n_sentences"]
    #     sentence_length = c["n_tokens"]
    #     out = np.zeros( (1, n_sentences, sentence_length + 2), dtype=float )  # +2 for begin-of-sentence and end-of-sentence tokens
    #     enc = self.vocab.encode(text, n_sentences=n_sentences, sentence_length=sentence_length)
    #     out[0, :, :] = enc
    #     return out

    # def encode_text_simple(self, text):
    #     # c = self.conf
    #     enc = self.vocab.simple_encode(text, self.sentence_length)
    #     return np.asarray(enc, dtype=float)


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
         lstm_size=512,
         emb_size=512,
         load_data_split=True,
         preload_images = True,
         verbose=False,
         debug=False,
         dev=False):
    config = locals()

    config["only_images"] = True
    print(" === TEST ONLY_IMAGES DATASET === ")
    ds = Uc5Dataset(config, version=None)
    ds.set_stage("train")
    print(len(ds))
    i = 0
    while i < len(ds):
        print(f"batch {i+1:02d}/{len(ds)}")
        batch = ds[i]
        if i == 0:
            print(batch)
        elif i == 1:
            i = len(ds)-3
        i += 1
    
    config["only_images"] = False
    print(" === TEST STANDARD DATASET === ")
    ds = Uc5Dataset(config, version=None)
    ds.set_stage("train")
    print(len(ds))
    i = 0
    while i < len(ds):
        print(f"{i+1:02d}/{len(ds)}")
        batch = ds[i]
        if i == 0:
            print(batch)
        elif i == 1:
            i = len(ds)-3
        i += 1
    
    print("\n === TEST SIMPLE DATASET === ")
    ds = Uc5Dataset(config, version="simple")
    ds.set_stage("train")
    print(len(ds))
    i = 0
    while i < len(ds):
        print(f"{i+1:02d}/{len(ds)}")
        batch = ds[i]
        if i == 0:
            print(batch)
        elif i == 1:
            i = len(ds)-3 
        i += 1

    

if __name__ == "__main__":
    fire.Fire(main)