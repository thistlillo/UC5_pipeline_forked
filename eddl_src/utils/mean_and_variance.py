import fire
import numpy as np
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


def flat_layout(img_fld, ext=".png"):
    filenames = [join(img_fld, fn) for fn in os.listdir(img_fld) if fn.endswith(ext)]
    print(f"|images| = {len(filenames)}")
    m = Tensor.zeros( [1, 3] )
    s = Tensor.zeros( [1, 3] )

    right = 30
    with tqdm(total=len(filenames)) as pbar:
        for i, fn in enumerate(filenames):
            spaces = (right-len(os.path.basename(fn))) * " "
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


def main(img_fld, dataset):

    m, s, std = 0, 0, 0
    if dataset == "iu-chest":
        m, s, std = flat_layout(img_fld, ".png")
    elif dataset == "chest-xray8":
        m, s, std = flat_layout(img_fld, ".png")
    else:
        assert False, f"unknown dataset {dataset}"

    print(f"mean: {m}")
    print(f"var:  {s}")
    print(f"std: {std}")

if __name__ == "__main__":
    fire.Fire(main)