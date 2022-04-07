# this file should be used from the root of the repository
import pyeddl.eddl as eddl
import pyecvl.ecvl as ecvl
from pyeddl.tensor import Tensor

import pandas as pd
import numpy as np
from posixpath import join
import yaml
import os

from utils.data_partitioning import load_data_split
from eddl_lib.uc5_dataset import Uc5Dataset

# paths
exp_fld = "/mnt/datasets/uc5/UC5_pipeline_forked/experiments_eddl/wp6"
cnn_fn = "cnn_84val_neptune179.onnx"
ds_fn = "img_reports_phi2_enc.tsv"
img_fld = "/mnt/datasets/uc5/std-dataset/image"

# read files from exp_fld
train_ids, valid_ids, test_ids = load_data_split(exp_fld)

cnn = eddl.import_net_from_onnx_file(join(exp_fld, cnn_fn))
eddl.build(
    cnn,
    eddl.rmsprop(0.01),
    ["soft_cross_entropy"],
    ["categorical_accuracy"],
    eddl.CS_GPU(mem="full_mem"),  # if args.gpu else eddl.CS_CPU(mem=args.mem),
    False  # do not initialize weights to random values
)
eddl.summary(cnn)
eddl.set_mode(cnn, 0)

ds = pd.read_csv(join(exp_fld, ds_fn), sep="\t").set_index("filename")  # .set_index("image_filename")
print(ds.shape)
print(ds.head().T)

# aux functions
def load_image(filename):
    augs = ecvl.SequentialAugmentationContainer([
                ecvl.AugToFloat32(divisor=255.0),
                ecvl.AugNormalize([0.48197903, 0.48197903, 0.48197903], [0.26261734, 0.26261734, 0.26261734]),
                ecvl.AugResizeDim([300, 300]),
                ecvl.AugCenterCrop([224, 224]),  # to do: test random crop also in prediction
                ])
    img = ecvl.ImRead(filename, flags=None)  # , flags=ecvl.ImReadMode.GRAYSCALE)
    ecvl.RearrangeChannels(img, img, "xyc")
    augs.Apply(img)
    ecvl.RearrangeChannels(img, img, "cxy")
    return img

split_ids = train_ids

for id in split_ids:
    img = load_image(join(img_fld, id))
    img = ecvl.ImageToTensor(img)
    print(img.info())
    a = np.zeros((2, 3,224,224), dtype=float)
    a = Tensor.fromarray(a)
    cnn.forward([a])
    #layer = eddl.getLayer(cnn, "cnn_out")

    # cnn_out_in = eddl.Input([semantic_dim], name="in_semantic_features")
    break
# for i, split in enumerate([train_ids, valid_ids, test_ids]):