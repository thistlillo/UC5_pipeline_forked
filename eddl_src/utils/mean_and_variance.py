from textwrap import indent
import fire
import numpy as np
import pandas as pd
import os
import pyecvl.ecvl as ecvl
from pyeddl.tensor import Tensor
from posixpath import join
from tqdm import tqdm

#
# output
#
# images: 7470, pixels:672300000
# mean: [[0.48197903 0.48197903 0.48197903]]
# var:  [[0.06896787 0.06896787 0.06896787]]
# std: [[0.26261734 0.26261734 0.26261734]]
#

# NOTE: 
img_size = 300
augs = ecvl.SequentialAugmentationContainer([
           ecvl.AugResizeDim([img_size, img_size]),
           ecvl.AugToFloat32(divisor=255.0),
])

def from_filenames(filenames):
    print(f"|images| = {len(filenames)}")
    m = Tensor.zeros( [1, 3] )
    s = Tensor.zeros( [1, 3] )

    right = 30
    with tqdm(total=len(filenames), ascii=True) as pbar:
        for i, fn in enumerate(filenames):
            spaces = (right - len(os.path.basename(fn))) * " "
            pbar.set_description(f"{os.path.basename(fn)}{spaces}")
            img = ecvl.ImRead(fn)
            augs.Apply(img)
            ecvl.RearrangeChannels(img, img, "cxy")
            a = np.array(img, dtype=float)
            a = a.reshape([3, -1])  # img_size*img_size])
            m += np.sum(a, axis=1)
            s += np.sum(a**2, axis=1) 
            pbar.update(1)
        #<
    #< pbar
    n_pixels = len(filenames) * img_size * img_size
    
    m = m / n_pixels
    s = (s / n_pixels) - (m ** 2)
    std = np.sqrt(s)
    
    print(f"images: {len(filenames)}, pixels:{n_pixels}")
    return m, s, std
#<

def flat_layout(img_fld, ext=".png"):
    filenames = [join(img_fld, fn) for fn in os.listdir(img_fld) if fn.endswith(ext)]
    return from_filenames(filenames)
#<

def mimic_cxr(df):
    img_fld = "/mnt/datasets/mimic-cxr/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0"

    def adjust_path(path):
        path = path[:-len(".dcm")]
        path = join(img_fld, path + ".jpg")
        return path
    df["path"] = df["path"].apply(lambda path: adjust_path(path))

    filenames = df["path"].tolist()
    return from_filenames(filenames)


def main(dataset, in_tsv=None, img_fld=None):

    m, s, std = 0, 0, 0
    if dataset == "iu-chest":
        m, s, std = flat_layout(img_fld, ".png")
    elif dataset == "chest-xray8":
        m, s, std = flat_layout(img_fld, ".png")
    elif dataset == "mimic_cxr":
        m, s, std = mimic_cxr(pd.read_csv(in_tsv, sep="\t"))
    else:
        assert False, f"unknown dataset {dataset}"

    print(f"mean: {m}")
    print(f"var:  {s}")
    print(f"std: {std}")

if __name__ == "__main__":
    fire.Fire(main)