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

from multiprocess.context import Process
from multiprocess.pool import Pool


def pad_2d_list_variable_len(a, pad=0):
    max_len = len(max(a, key=len))
    return [i + [pad] * (max_len - len(i)) for i in a]

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

def rand_pairs(n, m, seed=None):
    """
    Returns m random pairs of integers up to n by using triangular root decoding
    """
    rand = random if seed is None else random.Random(seed)
    return [decode_pair(i) for i in rand.sample(range(n*(n-1)//2),m)]

def rand_pairs_excluding(n, m, to_exclude, seed=None):
    """
    Returns m random pairs of integers up to n by using triangular root decoding
    Also excludes to_exclude elements
    """
    rand = random if seed is None else random.Random(seed)
    # decoding pairs range
    full_range = list(range(n*(n-1)//2))
    # this will hold all final pairs + to_exclude prior to removing to_exclude (sets will take care of adding with no repetition)
    pairs = to_exclude.copy()
    desired_len = len(to_exclude) + m
    # add until we have desired_len elements (extra m elements compared to original edge set)
    while len(pairs) != desired_len:
        pairs.add(decode_pair(rand.sample(full_range, 1)[0]))
    return pairs - to_exclude


def get_stateless_sampling_func(lamda, exp=True):
    if not exp:
        return (lambda net, nid, time=None: lamda)
    return (lambda net, nid, time=None: (-math.log(1. - random.random()) / lamda))

def get_stateful_sampling_func(sampler_type='expFactorTimesCountMultiState', exp=True, **kwargs):
    if not exp: 
        sampler_type += '_rate'
    func = globals()[sampler_type]
    return (lambda net, nid, time=None: func(net, nid, current_time=time, **kwargs))


def expFactorTimesCount(net, nid, state='I', lamda=1, **kwargs):
    """
    Sample from an Exponential with Neighbour counts
    """
    exp_param = lamda * net.node_counts[nid][state]
    if exp_param:
        return -(math.log(1. - random.random()) / exp_param)
    return float('inf')

def expFactorTimesCount_rate(net, nid, state='I', lamda=1, **kwargs):
    """
    Sample from an Exponential with Neighbour counts
    """
    exp_param = lamda * net.node_counts[nid][state]
    return exp_param

def expFactorTimesTimeDif(net, nid, lamda=1, current_time=100, **kwargs):
    """
    Sample from an Exponential with tracing time difference
    """
    exp_param = lamda * (current_time - net.traced_time[nid])
    if exp_param:
        return -math.log(1. - random.random()) / exp_param
    return float('inf')

def expFactorTimesTimeDif_rate(net, nid, current_time=100, lamda=1, **kwargs):
    """
    Sample from an Exponential with tracing time difference
    """
    exp_param = lamda * (current_time - net.traced_time[nid])
    return exp_param
    
def expFactorTimesCountImportance(net, nid, state='T', base=0, **kwargs):
    """
    Sample from an Exponential with Neighbour counts weighted by network count_importance
    """
    exp_param = base + net.count_importance * net.node_counts[nid][state]
    if exp_param:
        return -math.log(1. - random.random()) / exp_param
    return float('inf')

def expFactorTimesCountImportance_rate(net, nid, state='T', base=0, **kwargs):
    """
    Sample from an Exponential with Neighbour counts weighted by network count_importance
    """
    exp_param = base + net.count_importance * net.node_counts[nid][state]
    return exp_param

def expFactorTimesCountMultiState(net, nid, states=['I'], lamda=1, rel_states=[], rel=1, **kwargs):
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
    exp_param = lamda * (exp_param_states + rel * exp_param_rel_states)
    if exp_param:
        return -math.log(1. - random.random()) / exp_param
    return float('inf')

def expFactorTimesCountMultiState_rate(net, nid, states=['I'], lamda=1, rel_states=[], rel=1, **kwargs):
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
    exp_param = lamda * (exp_param_states + rel * exp_param_rel_states)
    return exp_param



def get_boxplot_statistics(data, axis=0, avg_without_idx=None, round_to=2):
    for_data = np.array(data)
    if axis:
        for_data = for_data.T
    shape_dat = for_data.shape
    # compute means and standard devs - if more than 1 element ddof=1, otherwise just 0
    # if ndim = 1 output array with one element
    if len(shape_dat) == 1:
        # note that the full mean is computed during the boxplot routine, so we skip that
        stds = [np.std(for_data, ddof=1) if shape_dat[0] > 1 else 0]
    else:
        # note that the full mean is computed during the boxplot routine, so we skip that
        stds = np.std(for_data, axis=0, ddof=1) if shape_dat[0] > 1 else [0] * shape_dat[1]
    
    # also select data without certain indexes
    if avg_without_idx:
        data_without_idx = np.delete(for_data, avg_without_idx, axis=0)
        shape_dat_without = data_without_idx.shape
        if len(shape_dat_without) == 1:
            means_without = [np.mean(data_without_idx) if shape_dat_without[0] > 0 else 0]
            stds_without = [np.std(data_without_idx, ddof=1) if shape_dat_without[0] > 1 else 0]
        else:
            means_without = np.mean(data_without_idx, axis=0) if shape_dat_without[0] > 0 else [0] * shape_dat_without[1]
            stds_without = np.std(data_without_idx, axis=0, ddof=1) if shape_dat_without[0] > 1 else [0] * shape_dat_without[1]
    
    # this list will hold the statistic dict results
    results = []
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
        dct['mean'] = round(means[i].get_ydata()[0], round_to)
        dct['std'] = round(stds[i], 2)
        whisk1 = whisks[2*i].get_ydata()
        whisk2 = whisks[2*i+1].get_ydata()
        dct['whislo'] = round(whisk1[1], round_to)
        dct['q1'] = round(whisk1[0], round_to)
        dct['med'] = round(meds[i].get_ydata()[0], round_to)
        dct['q3'] = round(whisk2[0], round_to)
        dct['whishi'] = round(whisk2[1], round_to)
        if avg_without_idx is not None:
            # Note: if avg_without_idx = [] it means the option was selected from args, but no simulation early stopped
            # In this case, we report the alternative = actual mean/std in order to be consistent across runs
            dct['mean_wo'] = round(means_without[i], round_to) if avg_without_idx else dct['mean']
            dct['std_wo'] = round(stds_without[i], round_to) if avg_without_idx else dct['std']
        results.append(dct)
    return results

    
### Hacks for TQDM to allow printing together with the progress line in the same STDOUT + joblib update per iter

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
            

### Debugging methods
            
# reloads all modules specified
def rel(*modules):
    for module in modules:
        importlib.reload(module)
        
# prints all variables specified in the current frame and their value
def pvar(*var, owners=True):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    fi = r.split(', ')
    last_var = len(var) - 1
    for i, x in enumerate(var):
        if x == '\n':
            print()
        else:
            if not owners:
                fi[i] = fi[i].split('.')[-1]
            format_to_print = "{} = {}, "
            if i == last_var:
                format_to_print = format_to_print[:-2]
            print(format_to_print.format(fi[i],x), end="", flush=True)
    print()
            
            
### Pickle and JSON dump to file and retrieve functions

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
        with open(json_or_path, 'r', encoding="utf8") as handle:
            return json.loads(handle.read())
        
def process_json_results(path=None, print_id_fail=True):
    """
    Note this method only processes the results for the first taut value in the JSON file
    """
    if path is None:
        path = 'data/run/batch1_pbs/*.json'
    else:
        path += '*.json'
    nested_dict = lambda: defaultdict(nested_dict)
    all_sim_res = nested_dict()
    for file in glob.glob(path):
        try:
            json_file = get_json(file)
            # the json files has keys: 'args' and the single 'taut' value chosen
            args = json_file['args']
            # ignore results for taur=0 as we ignore the case where no testing is done
            if args['taur'] == 0:
                continue
            taut = np.atleast_1d(args['taut'])[0]
            taur = args['taur']
            over = float(round(args['overlap'], 2))
            uptake = float(round(args.get('uptake', 1.), 2))
            pa = args['pa']
            try:
                # If only a single taut value present in running args, the results key is 'res'
                results = json_file['res']
            except:
                # Backwards compatibility except block - runs when mutliple taut values have been selected
                # The following try block is needed to cover for the case the supplied taut was an int
                try:
                    results = json_file[str(taut)]
                except:
                    results = json_file[str(int(taut))]
            all_sim_res[pa][uptake][over][taut][taur] = results
        except json.JSONDecodeError:
            if print_id_fail:
                print(int(re.findall('id(.*?)_', file)[0]), end=",")
    return all_sim_res


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


class NoDaemonProcess(Process):
    """Monkey-patch process to ensure it is never daemonized"""
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass
    
class NoDaemonPool(Pool):
    def Process(self, *args, **kwds):
        proc = super().Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc
    