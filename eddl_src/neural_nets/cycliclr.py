# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 18:54:21 2021

@author: cardillo
"""

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
#sns.set()

class CyclicLR:
    policy_TRIANGULAR = "triangular"
    policy_TRIANGULAR2 = "triangular2"
    policy_EXP_RANGE = "exp_range"
    
    SCALE_EVERY_CYCLE = "cycle"
    SCALE_EVERY_ITERATION = "iteration"
    
    def __init__(self,
                 lr_min=0.001, lr_max=0.006, stepsize=50,
                 policy=policy_TRIANGULAR,
                 gamma=1,
                 scale_fn=None,
                 log_fn=print):
        self.log_fn = log_fn

        if lr_max < lr_min:
            raise Exception(f"min={lr_min} > max={lr_max}: max must be >= min")
        
        self.policy = None
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.stepsize = stepsize
        print('init: stepsize', self.stepsize)
        self.cyclesize = 2 * stepsize
        self.gamma = gamma
                
        self.log("Initialized")
        
        self.lr = None
        self.scale_fn = scale_fn
        if scale_fn is None:
            self.policy = policy            
            self.scale_fn = self.set_scale_fn()

    def log(self, msg):
        if self.log_fn:
            self.log_fn("CyclicLR: " + msg)

    def lr_from_epoch(self, epoch):
        return self.get_lr(epoch)
    
    def lr_from_iteration(self, iteration):
        return self.get_lr(iteration)
    
    def get_lr_min(self):
        return self.lr_min

    def get_lr_max(self):
        return self.lr_max
    
    def set_scale_fn(self):
        if self.policy == CyclicLR.policy_TRIANGULAR:
            return lambda cycle: 1.0
        elif self.policy == CyclicLR.policy_TRIANGULAR2:
            # return lambda cycle: (0.5) ** (cycle-1)
            return lambda xx: 1/(2.** (xx-1))
        elif self.policy == CyclicLR.policy_EXP_RANGE:
            return lambda iteration: self.gamma ** iteration 

    def get_lr(self, epoch):
        # cycle = math.floor(1 + value / (2 * self.stepsize))
        cycle = np.floor(1 + value / ( 2* self.stepsize))
        # x = abs(value / self.stepsize - 2 * cycle + 1)
        x = np.abs(value / self.stepsize - 2 * cycle + 1)
        
        x2 = np.maximum(0, 1-x)
        lr = self.lr_min + (self.lr_max - self.lr_min) * x2 * self.scale_fn(cycle)
        return lr 

    def plot(self, n_epochs=10000, n_iters_in_epoch=500, fig=None):
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes
            ax = ax[0]
        
        n_iters = 0
        
        if self.policy is None:
            self.log("No plot")
            return
        
        if self.policy == CyclicLR.policy_EXP_RANGE:
            x = np.arange(n_epochs*n_iters_in_epoch)
            self.log(f"Policy set to exp_range, plotting over {n_epochs}*{n_iters_in_epoch}={len(x)} iterations, not epochs")
            ax.set_xlabel('(mini-batch) iteration')
        else:
            x = np.arange(n_epochs)
            ax.set_xlabel('epoch')
        
        # y = list(map(lambda y: self.get_lr(y), x))
        y = self.get_lr(x)
        ax.set_ylabel('learning rate')
        plt.plot(x,y)
        plt.title('Cyclic Learning Rate')
        plt.grid(True, alpha=0.3)
        
        plt.show()
        
#< class ends here

def main():
    clr = CyclicLR(lr_min=0.01, lr_max=0.05)
    clr.plot()


def main2():
    # values of `CIFAR-10
    #   50000 training images, 100 batch size -> 500 iterations per epoch
    #   stepsize (2--10)*iterations, this case  2 * iteration -> 1000, cycle=2000
    fig = plt.figure(100)
    fig.add_subplot()

    cyclic = CyclicLR(lr_min=0.001, lr_max=0.003,
                    stepsize=50, policy=CyclicLR.policy_TRIANGULAR2,
                    gamma=0.95, log_fn=print)

    print('0:', cyclic.get_lr(0))
    print('500:', cyclic.get_lr(500))

    print('1000:', cyclic.get_lr(1000))
    print('2000:', cyclic.get_lr(2000))
    print('3999:', cyclic.get_lr(3999))

    print('4000:', cyclic.get_lr(4000))
    #print('1:', cyclic.get_lr(4000))

    cyclic.plot(n_epochs=600, fig=fig)
    fig.savefig("cyclic.png")
    #print('-end-')
    # %% this plot is fine for tri, tri2

    # =============================================================================
    #     def get_lr(self, value):
    #         cycle = math.floor(1 + value / (2 * self.stepsize))
    #         x = abs(value / self.stepsize - 2 * cycle + 1)
    #         lr = self.lr_min + (self.lr_max - self.lr_min) * max(0, 1-x)
    #         return lr * self.scale_fn(value)
    # 
    # =============================================================================
    import collections
    n_epochs = 10000
    batchsize = 100
    n_batches_in_epoch = 50000 / batchsize
    lr_min = 0.01
    lr_max = 0.05  # 0.006

    cycle_size = 2 * n_batches_in_epoch
    stepsize = cycle_size / 2
    iterations = np.arange(n_epochs * n_batches_in_epoch)
    epoch_counter = np.arange(n_epochs)

    cycle = np.floor(1 + np.true_divide(epoch_counter, 2*stepsize))
    x = np.abs(np.divide(epoch_counter, stepsize) - 2 * cycle + 1)
    x2 = np.maximum(0, 1-x)

    def fn_lr_max_exp(x):
        return 0.998 ** (x-1)

    def fn_lr_max(x):
        return (0.5 ** (x-1))

    lr = lr_min + (lr_max - lr_min) * x2  * fn_lr_max(cycle)
    print(lr)
    plt.plot(epoch_counter, lr)


if __name__ == "__main__":
    main()