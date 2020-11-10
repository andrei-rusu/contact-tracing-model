import random
import math
import tqdm
import inspect
import re
import pickle
import json
import glob
import importlib
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sys import stdout
from contextlib import contextmanager
from cProfile import Profile
from pstats import Stats
from collections import defaultdict

def get_z_for_overlap(k=10, overlap=.08, include_add=False):
    if include_add:
        z = k * (1 - overlap) / (1 + overlap)
        return z, z
    else:
        z = k * (1 - overlap)
        return 0, z

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

def expFactorTimesTimeDif(net, nid, current_time=100, lamda=1):
    """
    Sample from an Exponential with tracing time difference
    """
    lamda *= current_time - net.traced_time[nid]
    if lamda:
        return -(math.log(random.random()) / lamda)
    return float('inf')    
    
def expFactorTimesCount(net, nid, state='I', lamda=1):
    """
    Sample from an Exponential with Neighbour counts
    """
    lamda *= net.node_counts[nid][state]
    if lamda:
        return -(math.log(random.random()) / lamda)
    return float('inf')

def expFactorTimesCountImportance(net, nid, state='T', base=0):
    """
    Sample from an Exponential with Neighbour counts weighted by network count_importance
    """
    lamda = base + net.count_importance * net.node_counts[nid][state]
    if lamda:
        return -(math.log(random.random()) / lamda)
    return float('inf')    

def expFactorTimesCountMultiState(net, nid, states=['I'], lamda=1, rel_states=[], rel=1, debug=False):
    """
    lamda : multiplicative factor of exponential
    rel : relative importance
    states : the number of these states will be multiplied with lamda
    rel_states : the number fo these states will be multiplied with lamda * rel
    """
    exp_param_states = exp_param_rel_states = 0
    counts = net.node_counts[nid]
    for state in states:
        exp_param_states += counts[state]
    for state in rel_states:
        exp_param_rel_states += counts[state]
    lamda *= exp_param_states + rel * exp_param_rel_states
    
    if lamda:
        return -(math.log(random.random()) / lamda)
    return float('inf')

def get_boxplot_statistics(data, axis=0):
    for_data = np.array(data)
    if axis:
        for_data = for_data.T
    shape_dat = for_data.shape
    # this list will hold the statistic dict results
    results = []
    # compute standard dev - if more than 1 element ddof=1 ; if ndim = 1 output array with one element
    if for_data.ndim == 1:
        stds = [np.std(for_data, ddof=1) if shape_dat[0] > 1 else 0]
    else:
        stds = np.std(for_data, axis=0, ddof=1) if shape_dat[0] > 1 else [0] * shape_dat[1]
    # boxplot computes mean, quartiles and whiskers
    B = plt.boxplot(for_data, showmeans=True)
    plt.close()
    # Record means and medians for each axis
    means = B['means']
    meds = B['medians']
    # there are 2 whiskers (low, high) per axis !!!
    whisks = B['whiskers']
    # iterate through the statistics
    for i in range(len(means)):
        dct = {}
        dct['mean'] = means[i].get_ydata()[0]
        dct['std'] = stds[i]
        whisk1 = whisks[2*i].get_ydata()
        whisk2 = whisks[2*i+1].get_ydata()
        dct['whislo'] = whisk1[1]
        dct['q1'] = whisk1[0]
        dct['med'] = meds[i].get_ydata()[0]
        dct['q3'] = whisk2[0]
        dct['whishi'] = whisk2[1]
        results.append(dct)
    return results


### Useful classes ###

# JSON-like class for holding up Events
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


# Class to use for profiling
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
        
# Class for JSON-encoding dicts containing numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

# List subclass which propagates method calls to the elements in the list
class ListDelegator(list):
    
    def __init__(self, *args, args_list=()):
        all_args = args + tuple(args_list)
        list.__init__(self, all_args)        

    def __getattr__(self, method_name):
        if self and hasattr(self[0], method_name):
            def delegator(self, *args, **kw):
                results = ListDelegator()
                for element in self:
                    results.append(getattr(element, method_name)(*args, **kw))
                # return the list of results only if there's any not-None value (equivalent to ignoring void returns)
                return results if results != [None] * len(results) else None
            # add the method to the class s.t. it can be called quicker the second time
            setattr(ListDelegator, method_name, delegator)
            # returning the attribute rather than the delegator func directly due to "self" inconsistencies
            return getattr(self, method_name)
        else:
            error = "Could not find '" +  method_name + "' in the attributes list of this ListDelegator's elements"
            raise AttributeError(error)
            
    def draw(self, *args, ax=[None, None], **kw):
        self[0].draw(*args, ax=ax[0], **kw)
        self[1].draw(*args, ax=ax[1], **kw)
    
### Hack for TQDM to allow print together with the progress line in the same STDOUT

@contextmanager
def redirect_to_tqdm():
    """Context manager to allow tqdm.write to replace the print function"""
    
    def tqdm_print(*args, **kwargs):
        for arg in args:
            tqdm.tqdm.write(str(arg), **kwargs)
            
    # Store builtin print
    old_print = print
    try:
        # Globaly replace print with tqdm.write
        inspect.builtins.print = tqdm_print
        yield
    finally:
        inspect.builtins.print = old_print
        
def tqdm_redirect(*args, **kwargs):
    with redirect_to_tqdm():
        for x in tqdm.tqdm(*args, file=stdout, **kwargs):
            yield x
            
@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
        
    # Store builtin completion callback
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    
    try:
        # Globaly replace the completion callback
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        yield tqdm_object
    finally:
        # restore completion callback
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        # close tqdm_object
        tqdm_object.close()
            

# Debugging methods
            
# reloads all modules specified
def rel(*modules):
    for module in modules:
        importlib.reload(module)
        
# prints all variables specified and their value in the current frame
def pvar(*var):
    for i, x in enumerate(var):
        if x == '\n':
            print()
        else:
            frame = inspect.currentframe().f_back
            s = inspect.getframeinfo(frame).code_context[0]
            r = re.search(r"\((.*)\)", s).group(1)
            fi = r.split(', ')
            print("{} = {}, ".format(fi[i],x), end="", flush=True)
    print('\n')
            
            
# Pickle and JSON dump to file and retrieve functions

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
    
def get_json(json_or_path):
    # Can load from either a json string or a file path
    try:
        return json.loads(json_or_path)
    except:
        with open(json_or_path, 'r') as handle:
            return json.loads(handle.read())
        
def process_json_results(path=None, print_id_fail=True):
    if path is None:
        path = 'data/run/*.json'
    else:
        path += '*.json'
    nested_dict = lambda: defaultdict(nested_dict)
    all_sim_res = nested_dict()
    for file in glob.glob(path):
        try:
            json_file = get_json(file)
            # the json files has keys: 'args' and the single 'taut' value chosen
            args = json_file['args']
            taut = args['taut']
            taur = args['taur']
            over = float(round(args['overlap'], 2))
            # The following try block is needed since sometimes the string key took the int version of the taut var
            try:
                results = json_file[str(taut)]
            except:
                results = json_file[str(int(taut))]
            all_sim_res[over][taut][taur] = results
        except json.JSONDecodeError:
            if print_id_fail:
                print(int(re.findall('id(.*?)_', file)[0]), end=",")
    return all_sim_res
    