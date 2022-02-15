import fire
import json
import math
import numpy as np
import pandas as pd
import pickle
from posixpath import join
import random

# import pyecvl.ecvl as ecvl
# import pyeddl.eddl as eddl

import text.reports as reports
# from text.encoding import SimpleCollator, StandardCollator

from utils.data_partitioning import DataPartitioner


def main(in_tsv,
         exp_fld,
         train_p=0.7,
         valid_p=0.1,
         shuffle_seed=2,
         term_column="auto_term",
         verbose=False,
         ):
    config = locals()
    p = DataPartitioner(config)
    p.partition_and_save()    

if __name__ == "__main__":
    fire.Fire(main)