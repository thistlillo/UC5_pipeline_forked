import numpy as np

eps = 1e-8
class PatienceEarlyStopping:
    def __init__(self, patience=10, min_epochs=20):
        self.v = []  # values of the watched metric
        self.stop = False
        self.iter_stop = 0
        self.min_epochs = min_epochs
        self.k = patience

    def append(self, value):
        self.v.append(value)
        # this is useful when comparing multiple EarlyStopping criterion
        # once True, stays True
        should_stop = self.evaluate_criterion()
        if (not self.stop) and should_stop:  # do not update iter_stop if stop flag is alredy True
            self.iter_stop = len(self.v)
        self.stop = self.stop or should_stop
        return self.stop

    def evaluate_criterion(self):
        k = self.k
        assert k < self.min_epochs, "wrong choice for k in evaluate_patience"

        if len(self.v) < self.min_epochs:
            return False # do not stop        

        a = np.array(self.v[-k:])
        assert len(a) == k
        a = a[1:] - a[:-1]
        return np.all( a <=0 )

    def __repr__(self):
        rep = f"Patience stopping criterion, patience={self.k} epochs"
        return rep
#< class PatienceEarlyStopping


# criterio Up_i with strip k (Early Stopping — But When? Prechelt 2012)
class UpEarlyStopping:
    def __init__(self, i=4, k=5):
        self.v = []  # values of the watched metric
        self.stop = False
        self.iter_stop = 0
        self.min_epochs = i * k
        self.i = i  # number of consecutive epoch strips, k-epoch long
        self.k = k 
        assert self.min_epochs >= self.i * self.k, "min_epochs too small"
        


    def append(self, metric):  # expects a metric as a percentage 
        assert (0 <= metric) and (metric <= 1), f"wrong metric with value {metric}: it should be 0<= metric <= 1"
        log_value = 1 - metric  # turn into error
        self.v.append(log_value)
        # this is useful when comparing multiple EarlyStopping criterion
        # once True, stays True
        should_stop = self.evaluate_criterion()
        if (not self.stop) and should_stop:  # do not update iter_stop if stop flag is alredy True
            self.iter_stop = len(self.v)
        self.stop = self.stop or should_stop
        return self.stop

    def evaluate_criterion(self):
        if len(self.v) < self.min_epochs:
            return False # go
        elif len(self.v) % self.k != 0:
            return False
        i = self.i
        k = self.k

        # when tested optimize the following: intermediate array generalization_errors is not needed
        indexes = reversed(range(0, i*k, k))
        generalization_errors = [self.v[j-1] for j in indexes]
        assert len(generalization_errors) == i
        stop = False
        j = 1
        while (j < i) and (not stop):
            print("j:", generalization_errors[j])
            print("j-1:", generalization_errors[j-1])
            stop = generalization_errors[j] < generalization_errors[j-1]
            j += 1
        self.stop = stop
        return stop
    
    def __repr__(self):
        rep = f"UpStoppingCriterion, strip={self.k}, i consecutive strips={self.i}"
        return rep
#< class UpEarlyStopping


# criterio Up_i with strip k (Early Stopping — But When? Prechelt 2012)
class ProgressEarlyStopping:
    def __init__(self, k=5, theta=0.1):
        self.v = []  # values of the watched metric
        self.stop = False
        self.iter_stop = 0
        self.min_epochs = 20
        self.theta = theta  # number of consecutive epoch strips, k-epoch long
        self.k = k 

    def append(self, metric):  # expects a metric as a percentage 
        assert (0 <= metric) and (metric <= 1), f"wrong metric with value {metric}: it should be 0<= metric <= 1"
        log_value = 1 - metric  # turn into error
        self.v.append(log_value)
        # this is useful when comparing multiple EarlyStopping criterion
        # once True, stays True
        should_stop = self.evaluate_criterion()
        if (not self.stop) and should_stop:  # do not update iter_stop if stop flag is alredy True
            self.iter_stop = len(self.v)
        self.stop = self.stop or should_stop
        return self.stop

    def evaluate_criterion(self):
        if len(self.v) < self.min_epochs:
            return False # go
        elif len(self.v) % self.k != 0:
            return False
        theta = self.theta
        k = self.k
        values = self.v[-k:]
        assert len(values) == k
        
        mean = sum(values) / len(values) + eps
        m = min(values) * k + eps
        crit = 1000 * (mean / m - 1)
        
        self.stop = crit < theta
        return self.stop
        
    
    def __repr__(self):
        rep = f"UpStoppingCriterion, strip={self.k}, i consecutive strips={self.i}"
        return rep
#< class UpEarlyStopping
