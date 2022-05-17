import numpy as np
import pandas as pd
import pickle
from posixpath import join
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from lightning.vocabulary_light import Vocabulary

from lightning.text_collation import collate_fn_one_s, collate_fn_n_sents

CHEST_IU = "chest-iu"
MIMIC_CXR = "mimic-cxr"

# *** I M A G E   T R A N S F O R M S
class ImageTransforms:
    def __init__(self, dataset:str, img_size=224):
        if dataset not in [CHEST_IU, MIMIC_CXR]:
            raise f"Uknown dataset: {dataset}"
        self.dataset = dataset
        self.img_size = img_size
        if self.dataset == CHEST_IU:
            self.configure_chest_iu()
        elif self.dataset == MIMIC_CXR:
            self.configure_mimic_cxr()
    #< constructor

    def configure_chest_iu(self):
        # m = [0.48158933, 0.48158933, 0.48158933]
        # s = [0.26255564, 0.26255564, 0.26255564]
        m = [ 0.4818, 0.4818, 0.4818 ]
        s = [ 0.2627, 0.2627, 0.2627 ]
        self.train_transforms = self.get_train_transforms(m, s)
        self.test_transforms = self.get_test_transforms(m, s)
    #< configure_chest_iu

    def configure_mimic_cxr(self):
        m = [ 0.4722, 0.4722, 0.4722 ]
        s = [ 0.3024, 0.3024, 0.3024 ] 
        self.train_transforms = self.get_mimic_train_transforms(m, s)
        self.test_transforms = self.get_mimic_test_transforms(m, s)
        
    #< configure_mimic_cxr

    def get_train_transforms(self, means, stds):
        return transforms.Compose([
            transforms.Resize([300, 300]),
            transforms.RandomRotation(degrees=[-5,5]),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds) ])
    #< 

    def get_test_transforms(self, means, stds):
        return transforms.Compose([
            transforms.Resize([300, 300]),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds) ])
    #< 

    def get_mimic_train_transforms(self, means, stds):
        return transforms.Compose([
            transforms.Resize([300, 300]),
            transforms.RandomRotation(degrees=[-5,5]),
            transforms.RandomCrop(self.img_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(means, stds) ])
    #< 

    def get_mimic_test_transforms(self, means, stds):
        return transforms.Compose([
            transforms.Resize([300, 300]),
            transforms.CenterCrop(self.img_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(means, stds) ])
    #< 

#< ImageTransforms
#--------------------------------------------------------------------------------------------------------------

# I M A G E   D A T A S E T  (only images, no texts)
class ImageDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, img_fld: str, img_transforms=None, n_classes=None, img_size=224):
        super().__init__()
        self.n_classes = n_classes or dataset.shape[1]
        assert self.n_classes == dataset.shape[1]
        
        self.ds = dataset
        self.img_fld = img_fld
        self.transforms = img_transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.ds)
    #< len

    def __getitem__(self, idx):
        assert (idx >=0) and (idx < len(self.ds))
        item = self.ds.iloc[idx]
        filename = item.name
        labels = item.values
        return self.load_image(filename), torch.tensor(labels.astype(np.float32))
    #< getitem

    def load_image(self, img_filename):
        fn = join(self.img_fld, img_filename)
        img = Image.open(fn)
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    #< load_image
#< ImageDataset
#--------------------------------------------------------------------------------------------------------------

class MultiModalDataset(Dataset):
    def __init__(self, img_dataset: pd.DataFrame, text_dataset: pd.DataFrame, 
            img_fld: str, img_transforms=None, n_classes=None, img_size=224,
            n_sentences=1, n_tokens=12, collate_fn=None, verbose=False, l1normalization=True):
        super().__init__()
        self.n_classes = n_classes or img_dataset.shape[1]
        assert self.n_classes == img_dataset.shape[1]
        self.img_ds = img_dataset
        # print(text_dataset.head())
        self.text_ds = text_dataset.set_index("image_filename")
        self.img_fld = img_fld
        self.transforms = img_transforms
        self.img_size = img_size
        self.n_sentences = n_sentences
        self.n_tokens = n_tokens
        self.collate_fn = collate_fn
        self.verbose = verbose
        self.l1normalization = l1normalization

    def __len__(self):
        return len(self.img_ds)
    
    def __getitem__(self, idx):
        assert (idx >=0) and (idx < len(self.img_ds))
        item = self.img_ds.iloc[idx]
        filename = item.name
        labels = item.values
        text = self.text_ds.loc[filename, "enc_text"]

        if self.collate_fn is not None:
            padded_text = self.collate_fn(text, n_sents=self.n_sentences, max_tokens=self.n_tokens, verbose=self.verbose)
        else:
            padded_text = text
        
        labels = torch.tensor(labels.astype(np.float32))
        if self.l1normalization:
            if (labels.sum() > 1):
                labels = labels / labels.sum()
        
         # TODO: make more efficient
        probs = []
        for i in range(padded_text.shape[0]):
            if padded_text[i, 0] == Vocabulary.PAD:
                probs.append(0)
            else:
                probs.append(1)
        # values 0 and 1 represent CLASS INDEXES: 0 means stop 1 means continue
        # the output layer of the classified will have dimension 2, one per class, with outputs 1 0 -> STOP; 0 1 -> CONTINUE
        probs.append(0)  # last element means stop
        probs = torch.tensor(probs)

        return self.load_image(filename), labels, torch.tensor(padded_text), probs

    def load_image(self, img_filename):
        fn = join(self.img_fld, img_filename)
        img = Image.open(fn)
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class MultiModalDatasetForGeneration(Dataset):
    def __init__(self, img_dataset: pd.DataFrame, text_dataset: pd.DataFrame, 
            img_fld: str, img_transforms=None, n_classes=None, img_size=224,
            n_sentences=1, n_tokens=12, collate_fn=None, verbose=False):
        super().__init__()
        self.n_classes = n_classes or img_dataset.shape[1]
        assert self.n_classes == img_dataset.shape[1]
        self.img_ds = img_dataset
        # print(text_dataset.head())
        self.text_ds = text_dataset.set_index("image_filename")
        self.img_fld = img_fld
        self.transforms = img_transforms
        self.img_size = img_size
        self.n_sentences = n_sentences
        self.n_tokens = n_tokens
        self.collate_fn = collate_fn
        self.verbose = verbose

    def __len__(self):
        return len(self.img_ds)
    
    def __getitem__(self, idx):
        assert (idx >=0) and (idx < len(self.img_ds))
        item = self.img_ds.iloc[idx]
        filename = item.name
        labels = item.values
        text = self.text_ds.loc[filename, "enc_text"]

        if self.collate_fn is not None:
            padded_text = self.collate_fn(text, n_sents=self.n_sentences, max_tokens=self.n_tokens, verbose=self.verbose)
        else:
            padded_text = text
        if self.l1normalization:
            labels = labels
            print(labels)
            assert False
        return filename, self.load_image(filename), torch.tensor(labels.astype(np.float32)), torch.tensor(padded_text)

    def load_image(self, img_filename):
        fn = join(self.img_fld, img_filename)
        img = Image.open(fn)
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def main():
    # in_fld = "/opt/uc5/results/sicaai/base_all_labels"
    in_fld = "/opt/uc5/results/sicaai/autoterms_th_130"
    print("folder:", in_fld)
    splits = pd.read_pickle( join(in_fld, "split_0.pkl"))
    # display(splits.head())
    dataset = pd.read_pickle( join(in_fld, "img_dataset.pkl"))
    # display(dataset.head())

    train_set = dataset.loc[splits.filename[splits.split == "train"]]
    print("train set: ", train_set.shape)
    img_transforms = ImageTransforms(dataset=CHEST_IU)
    train_trans =  img_transforms.train_transforms
    train_ds = ImageDataset(train_set, "../data/image", img_transforms=train_trans)
    img, labels = train_ds[3]
    print(img)
    print("len labels:", len(labels))
    print("labels:", labels)

    valid_set = dataset.loc[splits.filename[splits.split == "valid"]]
    test_trans = img_transforms.test_transforms
    valid_ds = ImageDataset(valid_set, "../data/image", img_transforms=test_trans)
    print("test set:", valid_set.shape)
    img, labels = valid_ds[1]
    print(img)
    print("len labels:", len(labels))
    print("labels:", labels)

    test_set = dataset.loc[splits.filename[splits.split == "test"]]
    test_trans = img_transforms.test_transforms
    test_ds = ImageDataset(test_set, "../data/image", img_transforms=test_trans)
    print("test set:", test_set.shape)
    img, labels = train_ds[1]
    print(img)
    print("len labels:", len(labels))
    print("labels:", labels)

def main_text():
    in_fld = "/opt/uc5/results/sicaai/autoterms_th_130"
    print("folder:", in_fld)
    splits = pd.read_pickle( join(in_fld, "split_0.pkl"))
    img_dataset = pd.read_pickle( join(in_fld, "img_dataset.pkl"))
    text_dataset = pd.read_pickle( join(in_fld, "img_text_dataset.pkl"))

    train_set = img_dataset.loc[splits.filename[splits.split == "train"]]
    # print("train set: ", train_set.shape)
    img_transforms = ImageTransforms(dataset=CHEST_IU)
    train_trans =  img_transforms.train_transforms
    train_ds = MultiModalDataset(train_set, text_dataset, "../data/image", img_transforms=train_trans, 
        n_sentences=1, n_tokens=100, collate_fn=collate_fn_one_s, verbose=True)
    img, labels, text = train_ds[3]
    print(img)
    print("len labels:", len(labels))
    print("labels:", labels)
    print(f"text, type {type(text)}: {text}")
    

if __name__ == "__main__":
    main()