# 
# UC5, DeepHealth
# Franco Alberto Cardillo (francoalberto.cardillo@ilc.cnr.it)
#
import numpy as np
from pyeddl._core import Metric
import pyeddl.eddl as eddl


class Jaccard(Metric):
    def __init__(self, max_labels=5, debug=False):
        Metric.__init__(self, "py_jaccard")
        self.max_labels = max_labels
        self.debug = debug

    def value(self, t, y):
        tt = t.getdata()
        yy = y.getdata()
        values = []
        # for every row in tt & yy
        for i in range(tt.shape[0]):
            tr = tt[i, :]
            yr = yy[i, :]
            # sort values in descending order: highest scores first
            ti = np.argsort(-tr)[:self.max_labels]
            yi = np.argsort(-yr)[:self.max_labels]

            # indexes of elements with a zero (or negative for y) value
            tzi = np.where(tr == 0)[0]
            yzi = np.where(yr <= 0)[0]
            # remove indexes of elements with a zero value: this cannot be done earlier
            #   since we need to keep the correct indexes for argsort
            tlabs = set(list(ti)).difference(tzi)
            ylabs = set(list(yi)).difference(yzi)
            if self.debug:
                print('True labels:', tlabs)
                print('Pred labels:', ylabs)
                print('-')
            # jaccard computed as cardinality of the intersection divided by the cardinality of the union
            intersection = ylabs.intersection(tlabs)
            jacc = len(intersection) / (len(ylabs) + len(tlabs) - len(intersection))
            values.append(jacc)
        jaccard = np.mean(values)
        return jaccard
    #<
#<
