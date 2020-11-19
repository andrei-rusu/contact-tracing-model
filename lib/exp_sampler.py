import numpy as np

class ExpSampler(dict):
    
    def __init__(self, size=1000):
        self.size = int(size)
            
    def get_next_sample(self, lamda):
        try:
            return next(self[lamda])
        except (KeyError, StopIteration):
            self[lamda] = iter(np.random.exponential(1/lamda, self.size))
            return next(self[lamda])
        
class ExpSamplerScaleOne():
    
    def __init__(self, size=1000):
        self.size = int(size)
        # keep an iterator over samples of exp(1)
        self.iter = iter(np.random.exponential(1, self.size))
            
    def get_next_sample(self, lamda):
        try:
            return (1 / lamda) * next(self.iter)
        except StopIteration:
            self.iter = iter(np.random.exponential(1, self.size))
            return next(self.iter)
        
        
def get_sampler(size=1000, scale_one=False):
    if scale_one:
        return ExpSamplerScaleOne(size)
    else:
        return ExpSampler(size)