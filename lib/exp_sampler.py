import numpy as np

class ExpSampler(dict):
    
    def __init__(self, size=1000):
        self.size = size
            
    def get_next_sample(self, lamda):
        try:
            return next(self[lamda])
        except (KeyError, StopIteration):
            self[lamda] = iter(np.random.exponential(1/lamda, self.size))
            return next(self[lamda])        