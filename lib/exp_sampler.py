import numpy as np
from random import random
from math import log

class ExpSampler():
    
    def get_next_sample(self, lamda):
        return -log(1. - random()) / lamda

class ExpSamplerPresample(dict):
    
    def __init__(self, size=1000):
        self.size = int(size)
            
    def get_next_sample(self, lamda):
        try:
            sample = next(self[lamda])
        except (KeyError, StopIteration):
            self[lamda] = iter(np.random.exponential(1/lamda, self.size))
            sample = next(self[lamda])
        finally:
            return sample
        
class ExpSamplerPresampleScaleOne():
    
    def __init__(self, size=1000):
        self.size = int(size)
        # keep an iterator over samples of exp(1)
        self.iter = iter(np.random.exponential(1, self.size))
            
    def get_next_sample(self, lamda):
        try:
            sample = next(self.iter)
        except StopIteration:
            self.iter = iter(np.random.exponential(1, self.size))
            sample = next(self.iter)
        finally:
            return (1 / lamda) * sample
        
        
def get_sampler(presample_size=1000, scale_one=False):
    if presample_size:
        if scale_one:
            return ExpSamplerPresampleScaleOne(presample_size)
        else:
            return ExpSamplerPresample(presample_size)
    else:
        return ExpSampler()