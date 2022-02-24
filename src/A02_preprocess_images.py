import fire
import glob
import numpy as np
import os
import pandas as pd
import pickle
from posixpath import join
import pyecvl.ecvl as ecvl
from tqdm import tqdm

from utils.misc import filename_from_path, Timer


def compute_AugResize(img, new_size=300):
    w, h = img.Width(), img.Height()
    m = min(w, h)
    s = new_size / m
    w2, h2 = int(w * s), int(h * s)

    return ecvl.AugResizeDim([new_size, new_size])  # ([w2, h2])
    

#
# images: 7470, pixels:672300000
# mean: [[0.48197903 0.48197903 0.48197903]]
# var:  [[0.06896787 0.06896787 0.06896787]]
# std: [[0.26261734 0.26261734 0.26261734]]
#

def main(img_fld, img_size, out_fn="images.pickle", verbose=False):
    # augs = ecvl.SequentialAugmentationContainer([
    #         ecvl.AugDivBy255(),  #  ecvl.AugToFloat32(divisor=255.0),
    #         ecvl.AugNormalize(122.75405603 / 255.0, 0.296964375 / 255.0),
    #         ecvl.AugResizeDim([img_size, img_size]),
    #         ])

    augs = lambda img: ecvl.SequentialAugmentationContainer([
        ecvl.AugToFloat32(divisor=255.0),  #  ecvl.AugToFloat32(divisor=255.0),
        ecvl.AugNormalize( [0.48197903, 0.48197903, 0.48197903], [0.26261734, 0.26261734, 0.26261734] ),
        compute_AugResize(img)
        #ecvl.AugCenterCrop([img_size, img_size])
        ])
    print(os.getcwd())

    # files
    files = glob.glob( f"{img_fld}/*.png")
    print(f" preprocessing |images| = {len(files)}")
    # encode
    d = {}
    for i, file in enumerate(tqdm(files)):
        img = ecvl.ImRead(file)
        #print(img.channels_)
        ecvl.RearrangeChannels(img, img, "xyc")        
        augs(img).Apply(img)  # normalization needs xyc images            
        ecvl.RearrangeChannels(img, img, "cxy")
        #print(img.channels_)
        a = np.array(img)
        #print(a.shape)
        d[ os.path.basename(file) ] = np.array(img)
        if verbose:
            tqdm.write(f"{i:04d} filename: {file}")
            tqdm.write(f"{i:04d} key: {os.path.basename(file)}")
            tqdm.write(f"{i:04d} image: {img.channels_}, {img.dims_}")
        

    print("saving dictionary as pandas.Series...")
    s = pd.Series(d)
    with open( out_fn, "wb" ) as fout:
        pickle.dump(s, fout)

    img = next(iter(d.values()))
    print(f"images saved with the following shape: {img.shape}")
    print(f"preprocessed (scaled, resized, channel order) images saved at: {out_fn}")
    print("done, exiting with success")


if __name__ == "__main__":
    fire.Fire(main)