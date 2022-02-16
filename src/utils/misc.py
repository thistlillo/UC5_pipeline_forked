#
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import humanize
import os
from posixpath import join
import time

import numpy as np
from posixpath import join

def filename_from_path(path, keep_extension=True):
    base = os.path.basename(path)
    if keep_extension:
        return base

    pre, _ = os.path.splitext(base)
    return pre

def list_files(fld, ext=None):
    files = os.listdir(fld)
    for f in files:
        if os.path.isfile(join(fld,f)):
            if ext is None or f.endswith(ext):
                yield f

def get_time_string():
    return time.strftime("%Y-%m-%d %H:%M:%S")

class Timer:
    def __init__(self, msg=None):
        self.msg = msg
        self.start = 0
        self.end = 0
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        # print("--- timer start")
        self.start_time = get_time_string()
        if self.msg is not None:
            print(f"'{self.msg}'", end=" ")
        print(f"timer starting: {self.start_time}")
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.end_time = get_time_string()
        if self.msg is not None:
            print(f"'{self.msg}'", end=" ")
        print(f"timer ending: {self.end_time}")

        self.interval = self.end - self.start
        if self.msg is not None:
            s = f"'{self.msg}' took {humanize.precisedelta(self.interval)}, secs: {self.interval}"
            print(s)


def encode_labels_one_hot(labels, n_classes, l1normalization=False):
    out = np.zeros((n_classes,), dtype=float)
    n_labels = len(labels)
    out[labels] = 1.0
    # l1normalization should be applied with a CrossEntropyLoss
    if l1normalization:
        out = out / n_labels
    return out
#<

    