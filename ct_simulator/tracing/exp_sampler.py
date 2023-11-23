import numpy as np
import itertools
from random import random
from math import log


class NumpySampler():
    """
    A template class for generating random samples using NumPy's random number generator. 
    It contains method definitions for generating single samples and multiple samples. Some methods expect a `lamda` rate parameter.

    Attributes:
        rng (np.random.RandomState): The random number generator.
    """
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
    
    def get_next_sample(self):
        raise NotImplementedError
    
    def get_next_sample(self, lamda):
        raise NotImplementedError
    
    def get_next_samples(self, size=10):
        for _ in range(size):
            yield self.get_next_sample()

    def get_next_samples(self, lamda, size=10):
        for _ in range(size):
            yield self.get_next_sample(lamda)
        

class ExpSampler():
    """
    Sampler based on random.random number generator that can be used to generate samples from an exponential distribution.

    """
    def get_next_sample(self, lamda):
        """
        Returns the next sample from the exponential distribution with the given rate parameter.

        Args:
            lamda (float): The rate parameter of the exponential distribution.

        Returns:
            float: The next sample from the exponential distribution.
        """
        return -log(1. - random()) / lamda

    
class ExpSamplerPresample(NumpySampler):
    """
    Sampler based on NumpySampler that can be used to generate samples from an exponential distribution. This class presamples `size` samples for each input `lamda` value, 
    and then adds an entry to `self.dict` mapping each `lamda` value to a lazily initialized iterator over that sample list.

    Attributes:
        size (int): The number of samples to presample.
        dict (dict): A dictionary mapping each `lamda` value to a lazily initialized iterator over the presampled list of samples.
    """
    def __init__(self, seed, size=1000):
        super().__init__(seed)
        self.size = int(size)
        # holds iterators for each lamda value (lazily initialized)
        self.dict = {}
            
    def get_next_sample(self, lamda):
        """
        Returns the next sample from the exponential distribution with the given rate parameter.

        Args:
            lamda (float): The rate parameter of the exponential distribution.

        Returns:
            float: The next sample from the exponential distribution.
        """
        try:
            sample = next(self.dict[lamda])
        except (KeyError, StopIteration):
            self.dict[lamda] = iter(self.rng.exponential(1/lamda, self.size))
            sample = next(self.dict[lamda])
        return sample

        
class ExpSamplerPresampleScaleOne(NumpySampler):
    """
    Sampler based on NumpySampler that can be used to generate samples from an exponential distribution. This presamples `size` samples from the exponential distribution 
    with rate parameter 1, and scales them by `1/lamda` to generate samples from the exponential distribution with rate parameter `lamda`.

    Attributes:
        size (int): Number of samples to presample from the exponential distribution with rate parameter 1.
        iter (iterator): Iterator over samples of exp(1).

    Methods:
        get_next_sample(lamda): Returns the next sample from the exponential distribution with rate parameter `lamda`.

    """
    def __init__(self, seed, size=1000):
        super().__init__(seed)
        self.size = int(size)
        # keep an iterator over samples of exp(1)
        self.iter = iter(self.rng.exponential(1, self.size))
            
    def get_next_sample(self, lamda):
        """
        Returns the next sample from the exponential distribution with rate parameter `lamda`.

        Args:
            lamda (float): Rate parameter of the exponential distribution.

        Returns:
            float: A sample from the exponential distribution with rate parameter `lamda`.
        """
        try:
            sample = next(self.iter)
        except StopIteration:
            self.iter = iter(self.rng.exponential(1, self.size))
            sample = next(self.iter)
        return (1 / lamda) * sample


class UniformSampler(NumpySampler):
    """
    A class for generating uniformly distributed random samples using NumPy's random number generator.
    """
    def get_next_sample(self):
        """
        Returns a single random sample from a uniform distribution.

        Returns:
            float: The next sample from the uniform distribution.
        """
        return self.rng.random()
    
    def get_next_samples(self, size=10):
        """
        Returns an array of random samples from a uniform distribution.
        
        Args:
            size (int): The number of samples to generate. Default is 10.

        Returns:
            np.array: An array of random samples from the uniform distribution.
        """
        return self.rng.random(size)


class UniformSamplerPresample(NumpySampler):
    """
    A class for generating uniformly distributed random samples using NumPy's random number generator. This class presamples `size` samples.

    Attributes:
        seed (int): The seed value for the random number generator.
        size (int): The number of samples to presample.
        actual (int): The actual number of samples remaining.
        iter (iterator): An iterator over samples of uniform(0,1).
    """
    def __init__(self, seed, size=1000):
        super().__init__(seed)
        self.size = self.actual = int(size)
        # keep an iterator over samples of uniform(0,1)
        self.iter = iter(self.rng.random(self.size))
        
    def get_next_sample(self):
        """
        Get the next sample from the iterator.

        Returns:
            sample (float): The next sample from the iterator.
        """
        try:
            sample = next(self.iter)
            self.actual -= 1
        except StopIteration:
            self.iter = iter(self.rng.random(self.size))
            sample = next(self.iter)
            self.actual = self.size - 1
        return sample
        
    def get_next_samples(self, size=10):
        """
        Get the next `size` samples from the iterator.

        Args:
            size (int): The number of samples to get.

        Returns:
            Iterable: An iterable over the next `size` samples.
        """
        sliced = itertools.islice(self.iter, size)
        if self.actual < size:
            missing = size - self.actual
            replacing_sample = self.rng.random(missing + self.size)
            sample = itertools.chain(sliced, replacing_sample[:missing])
            self.iter = iter(replacing_sample[missing:])
            self.actual = self.size
            return sample
        self.actual -= size
        return sliced
        
        
def get_sampler(type='exp', presample_size=1000, scale_one=False, seed=None):
    """
    Returns an instance of the ExpSampler class with optional presampling and scaling.

    Args:
        presample_size (int, optional): The number of samples to presample. Defaults to 1000.
        scale_one (bool, optional): If True, the samples are scaled from Exp(1). 
            If False, the samples are sampled directly from Exp(lamda). Defaults to False.
        seed (int, optional): The random seed to use for sampling. Defaults to None.

    Returns:
        An instance of an exponential sampler.
    """
    if type == 'exp':
        if presample_size:
            if scale_one:
                return ExpSamplerPresampleScaleOne(seed, presample_size)
            else:
                return ExpSamplerPresample(seed, presample_size)
        else:
            return ExpSampler()
    else:
        if presample_size:
            return UniformSamplerPresample(seed, presample_size)
        else:
            return UniformSampler(seed)