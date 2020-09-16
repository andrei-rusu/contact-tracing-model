import random
import math
import tqdm
import inspect
import re
import pickle

from sys import stdout
from contextlib import contextmanager
from cProfile import Profile
from pstats import Stats

def get_z_for_overlap(k=10, overlap=.08):
    return k * (1 - overlap) / (1 + overlap)

def get_overlap_for_z(k=10, z_add=5, z_rem=5):
    return (k - z_rem) / (k + z_add)
    
def decode_pair(i):
    """
    Returns triangular root and the difference between the root and the number
    """
    # triangular root is the second item in the pair
    second = math.floor((1+math.sqrt(1+8*i))/2)
    # first element is difference from root to number i
    first = i - second*(second-1)//2 
    return first, second

def rand_pairs(n, m):
    """
    Returns m random pairs of integers up to n by using triangular root decoding
    """
    return [decode_pair(i) for i in random.sample(range(n*(n-1)//2),m)]

def rand_pairs_excluding(n, m, to_exclude):
    """
    Returns m random pairs of integers up to n by using triangular root decoding
    Also excludes to_exclude elements
    """
    # decoding pairs range
    full_range = list(range(n*(n-1)//2))
    # this will hold all final pairs + to_exclude prior to removing to_exclude (sets will take care of adding with no repetition)
    pairs = to_exclude.copy()
    desired_len = len(to_exclude) + m
    # add until we have desired_len elements (extra m elements compared to original edge set)
    while len(pairs) != desired_len:
        pairs.add(decode_pair(random.sample(full_range, 1)[0]))
    return pairs - to_exclude

def exp(lamda=1):
    """
    Sample from an Exponential
    """
    if lamda:
        return -(math.log(random.random()) / lamda)
    return float('inf')
    
def expFactorTimesCount(net, nid, state='I', lamda=1, base=0):
    """
    Sample from an Exponential with Neighbour counts
    """
    exp_param = base + lamda * net.node_counts[nid][state]
    if exp_param:
        return -(math.log(random.random()) / exp_param)
    return float('inf')


### JSON-like class for holding up Events

class Event(dict):
    """
    General JSON-like class for recording information about various Events
    """
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Profiler(Profile):
    """ Custom Profile class with a __call__() context manager method to
        enable profiling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable()  # Profiling initially off.

    @contextmanager
    def __call__(self):
        self.enable()
        yield  # Execute code to be profiled.
        self.disable()
        
    def stat(self, stream=None):
        if not stream:
            stream = stdout
        stats = Stats(self, stream=stream)
        stats.strip_dirs().sort_stats('cumulative', 'time')
        stats.print_stats()
        
        
### Hack for TQDM to allow print together with the progress line

@contextmanager
def redirect_to_tqdm():
    # Store builtin print
    old_print = print
    try:
        # Globaly replace print with tqdm.write
        inspect.builtins.print = tqdm.tqdm.write
        yield
    finally:
        inspect.builtins.print = old_print
        
def tqdm_redirect(*args, **kwargs):
    with redirect_to_tqdm():
        for x in tqdm.tqdm(*args, file=stdout, **kwargs):
            yield x
            
# dump pickle
def pkl(obj, filename=None):
    if not filename:
        frame = inspect.currentframe().f_back
        s = inspect.getframeinfo(frame).code_context[0]
        filename = re.search(r"\((.*)\)", s).group(1)
    filename = 'saved/' + filename + '.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def get_pkl(obj=None):
    # Load data (deserialize)
    if type(obj) != str:
        frame = inspect.currentframe().f_back
        s = inspect.getframeinfo(frame).code_context[0]
        obj = re.search(r"\((.*)\)", s).group(1)
    filename = 'saved/' + obj + '.pkl'
    with open(filename, 'rb') as handle:
        return pickle.load(handle)