import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
from pyeddl._core import Loss, Metric

from sklearn.metrics import balanced_accuracy_score

class BalancedAccuracy(Metric):

    def __init__(self):
        Metric.__init__(self, "py_balanced_accuracy")

    def value(self, y, y_est):
        # size = y.size / y.shape[0]
        
        target = y.getdata()
        est = y_est.getdata()
        return balanced_accuracy_score(target, est)

        