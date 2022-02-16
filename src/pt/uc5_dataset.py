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
from PIL import Image
from posixpath import join
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from text.vocabulary import Vocabulary
import text.reports as reports
from text.encoding import SimpleCollator, StandardCollator

from utils.misc import Timer
import utils.misc as mu

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""

#     def __call__(self, image):
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         image = image.transpose((2, 0, 1))
#         return image

class Uc5ImgDataset(Dataset):
    VERBOSE_NONE = 0
    VERBOSE_INFO = 1
    VERBOSE_DEBUG = 2


    def __init__(self, tsv, n_classes, conf, l1normalization=True, version="simple"):
        super().__init__()
        self.tsv = tsv
        self.n_classes = n_classes
        self.conf = conf
        self.l1normalization = l1normalization
        self.version = version
        self.only_images = self.conf["only_images"]
        self.img_fld = self.conf["img_fld"]
        self.n_sentences = self.conf["n_sentences"]
        self.sentence_length = self.conf["n_tokens"]
        
        # verbose level
        self.verbose = Uc5ImgDataset.VERBOSE_NONE
        if conf["debug"]:
            self.verbose = Uc5ImgDataset.VERBOSE_DEBUG
        elif conf["verbose"]:
            self.verbose = Uc5ImgDataset.VERBOSE_INFO

        if self.version == "simple":
            self.n_sentences = 1
            self.collator = SimpleCollator()
        else:
            # print(f"StandardCollator, n_sentences: {self.n_sentences}")
            self.collator = StandardCollator()     

        self.from_disk_transform = self._get_from_disk_transform()
        
    #< init

    def _get_from_disk_transform(self):

        m, s = ([0.485], [0.229]) if self.conf["single_channel_cnn"] else \
                ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s),
            transforms.Resize(300),
            transforms.RandomCrop(224)
        ])
        return T
    #<

    def shuffle(self):
        self.random.shuffle(self.stage_ids)
    #<

    def __len__(self):
        return len(self.tsv)
    #<

    def __getitem__(self, idx):  # idx: int position
        if torch.is_tensor(idx):  # REMOVE
            idx = idx.tolist()
        filename = self.tsv.iloc[idx].name
        
        if self.only_images:
            return self.getitem_only_images(filename)
        # getitem_simple / getitem_complex differ for a single SQUEEZE on the text:
        #   in the "simple" case, the sentence dimension is removed
        # the two methods are kept separated for convenience, but they could be easily merged
        elif self.version == "simple":
            return self.getitem_simple(filename)
        else:
            return self.getitem_complex(filename)
    #<    
        
    def get_image(self, img_id):
        fn = join(self.img_fld, img_id)
        img = Image.open(fn)
        if self.conf["single_channel_cnn"]:
            img = img.getchannel(0)
        if self.from_disk_transform:
            img = self.from_disk_transform(img)
        return img
    #<
         
    # XXX rename all get_image to get_batch
    def getitem_only_images(self, img_id):
        img = self.get_image(img_id)
        labels = self.tsv.loc[img_id, "one_hot_labels"]  # self.encode_labels(self.tsv.loc[img_id, "labels"])
        return img, torch.tensor(labels)
    #<

    # REMOVE THIS METHOD
    # ONE SENTENCE WITH MAX_TOKENS TOKENS 
    def getitem_simple(self, img_id):
        img, lab, e_text = self.load_id(img_id)
        img = np.array(img, copy=False)
        e_text = np.squeeze(e_text)
        probs = torch.tensor([1, 0])
        return img, lab, e_text, probs
    #<

    def getitem_complex(self, img_id):
        img, lab, text = self.load_id(img_id)
        probs = []
        # print(text)
        
        # TODO: make more efficient
        for i in range(text.shape[0]):
            if text[i, 0] == Vocabulary.PAD:
                probs.append(0)
            else:
                probs.append(1)
        # values 0 and 1 represent CLASS INDEXES: 0 means stop 1 means continue
        # the output layer of the classified will have dimension 2, one per class, with outputs 1 0 -> STOP; 0 1 -> CONTINUE
        probs.append(0)  # last element means stop
        probs = torch.tensor(probs)
        # img = np.array(img, copy=False)
        return img, lab, text, probs
    #<

    def load_id(self, img_fn):
        # image names
        text_col = "enc_" + self.conf["text_column"]
        data = self.tsv.loc[img_fn, ["labels", text_col]]
     
        labels = mu.encode_labels_one_hot(
                [int(l) for l in data["labels"].split(reports.list_sep)], 
                n_classes = self.n_classes,
                l1normalization=self.l1normalization)
          
        # labels = [int(l) for l in labels.split(reports.list_sep)]
        text = data["enc_" + self.conf["text_column"]]
        
        img = self.get_image(img_fn)
        # labels = self.encode_labels(labels)
        text = self.encode_text(text)
        
        return img, torch.tensor(labels), torch.tensor(text)
    #<

    def encode_text(self, text):
        c = self.conf

        n_sentences = self.n_sentences        
        sentence_length = self.sentence_length

        out = np.zeros((n_sentences, sentence_length), dtype=float)
        # result will differ depending on the specific collator
        collated_text = self.collator.parse_and_collate(text, sentence_length, n_sentences, pad=True)
        # out = torch.tensor(collated_text)
        
        return collated_text
    #<

# --------------------------------------------------
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
         lstm_size=512,
         emb_size=512,
         load_data_split=True,
         preload_images = True,
         verbose=False,
         debug=False,
         dev=False):
    config = locals()
    print(f"tsv filename: {in_tsv}")
    tsv = pd.read_csv(config["in_tsv"], sep=reports.csv_sep, na_filter=False)
    tsv.set_index("filename", inplace=True, drop=False)


    # this column is filled by the datamodule, providing a view of the input tsv to the three datasets (train, valid, test)
    n_classes = 100
    l1normalization = True
    tsv["one_hot_labels"] = tsv["labels"].apply(
            lambda x: mu.encode_labels_one_hot(
                [int(l) for l in x.split(reports.list_sep)], 
                n_classes,
                l1normalization=l1normalization)
                )
    #print(tsv.columns)
    #print(tsv.head())
    #print(tsv.index)
    
    config["only_images"] = True
    print(" === TEST ONLY_IMAGES DATASET === ")
    ds_img_only = Uc5ImgDataset(tsv, conf=config, version=None)
    print(f"|dataset| = {len(ds_img_only)}")
    i = 0
    while i < len(ds_img_only):
        print(f"accessing image {i}")
        img, labels = ds_img_only[i]
        print(f"example {i+1:02d}/{len(ds_img_only)}, img: {img.shape}, {labels.shape}")
        
        if i == 10:
            i = len(ds_img_only)-3
        i += 1

    print("only images, test passed\n\n")


    config["only_images"] = False
    print(" === TEST STANDARD DATASET === ")
    ds_standard = Uc5ImgDataset(tsv, conf=config, version=None)
    print(f"|dataset| = {len(ds_standard)}")
    i = 0
    while i < len(ds_standard):
        print(f"{i+1:02d}/{len(ds_standard)}")
        img, labels, text, probs = ds_standard[i]
        print(f"img: {img.shape}, labels: {labels.shape}, text: {text.shape}, probs: {probs}")
        if i == 10:
            i = len(ds_standard)-3
        i += 1

    print("\n === TEST SIMPLE DATASET === ")
    ds_simple = Uc5ImgDataset(tsv, conf=config, version="simple")
    print(f"|dataset| = {len(ds_simple)}")
    i = 0
    while i < len(ds_simple):
        print(f"{i+1:02d}/{len(ds_simple)}")
        img, labels, text, probs = ds_simple[i]
        print(f"img: {img.shape}, labels: {labels.shape}, text: {text.shape}, probs: {probs}")
        if i == 10:
            i = len(ds_simple)-3
        i += 1

    print("\n === NOW TESTING STANDARD DATALOADER ===")
    dataloader = DataLoader(ds_standard, batch_size=10, shuffle=True)
    batch = next(iter(dataloader))
    for v in batch:
        print(v.shape)

    print("\n === NOW TESTING SIMPLE DATALOADER ===")
    dataloader = DataLoader(ds_simple, batch_size=15, shuffle=True)
    batch = next(iter(dataloader))
    for v in batch:
        print(v.shape)

    print("\n === NOW TESTING IMAGE_ONLY DATALOADER ===")
    dataloader = DataLoader(ds_img_only, batch_size=9, shuffle=True)
    batch = next(iter(dataloader))
    for v in batch:
        print(v.shape)

if __name__ == "__main__":
    fire.Fire(main)