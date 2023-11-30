import random
import math
import tqdm
import inspect
import re
import pickle
import json
import glob
import importlib
import numpy as np
import sys
import io
import matplotlib.pyplot as plt
import multiprocessing

from sys import stdout
from contextlib import contextmanager
from cProfile import Profile
from pstats import Stats
from collections import defaultdict
from PIL import Image
from base64 import b64decode


@contextmanager
def no_std_context(enabled=False):
    """
    Disable STDOUT/STDERR within this context

    Args:
        enabled (bool): If True, STDOUT/STDERR will be disabled. Default is False.

    Returns:
        None
    """
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    if enabled:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr

        
def is_not_empty(lst):
    """
    Check if a given list-like object is not empty. This works for both None, lists and numpy arrays.

    Args:
        lst: A list-like object to check.

    Returns:
        True if the list-like object is not empty, False otherwise.
    """
    return lst is not None and getattr(lst, 'size', len(lst))


def float_defaultdict():
    """
    Returns a defaultdict object with default value of 0.0 for any missing keys.
    This function is a replacement for `lambda: defaultdict(float)` which cannot be pickled.
    """
    return defaultdict(float)


def r_from_growth(growth, method='exp', t=7, mean=6.6, shape=1.87, inv_scale=0.28):
    """
    Get R_e from growth rate (NOT exponential growth) assuming Gamma distribution of infection generation time.

    Args:
        growth (float): The growth rate.
        method (str): The method used to calculate R_e. Can be 'exp' or 'jrc'.
        t (float): The time interval.
        mean (float): The mean of the Gamma distribution of generation time.
        shape (float): The shape of the Gamma distribution of generation time.
        inv_scale (float): The inverse scale of the Gamma distribution of generation time.

    Returns:
        float: The value of R_e.
    """
    if not growth:
        return 0
    if method == 'exp':
        if inv_scale is None:
            inv_scale = shape / mean
        return (1 + math.log(growth) / (inv_scale * t)) ** shape
    elif method == 'jrc':
        return 1 + math.log(growth) / t * mean

    
def pad_2d_list_variable_len(a, pad=0):
    """
    Pads a 2D list with variable length sublists to make all sublists have the same length.
    
    Args:
        a (list): The 2D list to pad.
        pad (int, optional): The value to use for padding. Defaults to 0.
    
    Returns:
        list: The padded 2D list.
    """
    if a and isinstance(a[0], list):
        max_len = len(max(a, key=len))
        return [i + [pad] * (max_len - len(i)) for i in a]
    return a


def get_z_for_overlap(k=10, overlap=.08, include_add=0):
    """
    Calculates the value of z for a given overlap and average degree. This assumes z_add = 0.

    Args:
        k (int, optional): The average degree of the graph.
        overlap (float, optional): The overlap between adjacent nodes. Defaults to 0.08.
        include_add (int, optional): Whether to include an additional value of z. Defaults to 0.

    Returns:
        tuple: A tuple containing one or two values of z, depending on the value of include_add.
    """
    if include_add:
        z = k * (1 - overlap) / (1 + overlap)
        return z, z
    else:
        z = k * (1 - overlap)
        return 0, z

    
def get_overlap_for_z(k, z_add, z_rem):
    """
    Calculates the overlap between the original graph and the traced graph for a given average degree and the average number of connections added or removed during tracing.

    Args:
        k (int): The average degree of the graph.
        z_add (int): The average number of connections added during tracing.
        z_rem (int): The average number of connections removed during tracing.

    Returns:
        float: The overlap between the original graph and the traced graph.
    """
    return (k - z_rem) / (k + z_add)
  

def decode_pair(i):
    """
    Returns the difference between a number i and its triangular root, as well as the triangular root itself. This can be used to bijectively map integers to pairs of integers.

    Args:
        i (int): The number to decode.

    Returns:
        tuple: A tuple containing two integers. The first integer is the difference between the triangular root and i, and the second integer is the triangular root itself.
    """
    # triangular root is the second item in the pair
    second = math.floor((1+math.sqrt(1+8*i))/2)
    # first element is difference from root to number i
    first = i - second*(second-1)//2
    return first, second


def rand_pairs(n, m, seed=None):
    """
    Returns m random pairs of integers up to n by using triangular root decoding
    
    Args:
        n (int): The maximum integer value for the pairs.
        m (int): The number of pairs to generate.
        seed (int, optional): The seed value for the random number generator. Defaults to None.
    
    Returns:
        list: A list of m random pairs of integers up to n.
    """
    rand = random if seed is None else random.Random(seed)
    return [decode_pair(i) for i in rand.sample(range(n*(n-1)//2),m)]


def rand_pairs_excluding(n, m, to_exclude, seed=None):
    """
    The function returns a set of m random pairs of distinct integers from 0 to n-1, excluding the pairs specified by to_exclude. 
    It uses a triangular root decoding technique to map a single integer to a pair of integers.

    Args:
        n (int): an integer specifying the upper bound (exclusive) of the integers in the pairs. It must be positive.
        m (int): an integer specifying the number of pairs to return. It must be non-negative and less than or equal to n*(n-1)//2 - len(to_exclude).
        to_exclude (set): a set of pairs of integers that are not allowed to be in the result. Each pair must be a tuple of two distinct integers from 0 to n-1.
        seed (int): an optional integer or None (default) specifying the seed for the random number generator. If None, the default random module is used. 
            If an integer, a new Random instance with the given seed is used.

    Returns:
        pairs (set): a set of m random pairs of distinct integers from 0 to n-1, excluding the pairs in to_exclude. Each pair is a tuple of two integers.

    Example:
    >>> rand_pairs_excluding(5, 3, {(0, 1), (2, 3)}, seed=42)
    {(0, 2), (1, 3), (3, 4)}
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


### Functions for sampling from various Exponential distributions ###

def get_stateless_sampling_func(lamda, exp=True):
    """
    Returns a sampling function that can be used to sample inter-event times of stateless transitions for a given node in a network.
    Stateless transitions do not depend on the connections of the network.

    Args:
        lamda (float): The base rate at which events occur.
        exp (bool): If True, the rate is exponentially sampled at this stage. Otherwise, the base rate is used.

    Returns:
        A function that takes in a network, a node ID, and an optional time, and returns the inter-event time for that node.
    """
    if not exp:
        return (lambda net, nid, time=None: lamda)
    return (lambda net, nid, time=None: (-math.log(1. - random.random()) / lamda))


def get_stateful_sampling_func(sampler_type='expFactorTimesCountMultiState', exp=True, **kwargs):
    """
    Returns a sampling function that can be used to sample inter-event times of stateful transitions for a given node in a network.
    Stateful transitions depend on the present connections of the network.

    Args:
        sampler_type (str): The type of sampler to use. Defaults to 'expFactorTimesCountMultiState'.
        exp (bool): Whether to exponentially sample the rate at this stage or leave it as a base rate. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the sampler function.

    Returns:
        A function that takes a network, node ID, and optional current time, and returns a sample from the network.
    """
    if not exp: 
        sampler_type += '_rate'
    func = globals()[sampler_type]
    return (lambda net, nid, time=None: func(net, nid, current_time=time, **kwargs))


def expFactorTimesCount(net, nid, state='I', lamda=1, **kwargs):
    """
    Sample from an Exponential distribution with a rate parameter that is proportional to the number of neighbors of a given node in a given state.

    Args:
        net (tracing.Network): The network object.
        nid (int): The ID of the node.
        state (str, optional): The state of the node to consider. Defaults to 'I'.
        lamda (float, optional): The multiplicative factor of the rate parameter. Defaults to 1.
        **kwargs: Additional keyword arguments.

    Returns:
        float: A sample from the Exponential distribution.
    """
    exp_param = lamda * net.node_counts[nid][state]
    if exp_param:
        return -(math.log(1. - random.random()) / exp_param)
    return float('inf')


def expFactorTimesCount_rate(net, nid, state='I', lamda=1, **kwargs):
    """
    Returns the rate parameter of an Exponential distribution proportional to the number of neighbors of a given node in a given state.

    Args:
        net (tracing.Network): The network object
        nid (int): The ID of the node.
        state (str): The state of the neighbors to consider. Default is 'I'.
        lamda (float): The multiplicative factor of the rate parameter. Default is 1.
        **kwargs: Additional keyword arguments.

    Returns:
        float: The rate parameter for the Exponential distribution.
    """
    exp_param = lamda * net.node_counts[nid][state]
    return exp_param


def expFactorTimesTimeDif(net, nid, lamda=1, current_time=100, **kwargs):
    """
    Sample from an Exponential distribution with a rate parameter that is proportional to the time difference between the current time and the last time the node was traced.

    Args:
        net (tracing.Network): The network object.
        nid (int): The ID of the node.
        lamda (float, optional): The multiplicative factor of the rate parameter. Defaults to 1.
        current_time (float, optional): The current time. Defaults to 100.
        **kwargs: Additional keyword arguments.

    Returns:
        float: A random sample from the exponential distribution.
    """
    exp_param = lamda * (current_time - net.traced_time[nid])
    if exp_param:
        return -math.log(1. - random.random()) / exp_param
    return float('inf')


def expFactorTimesTimeDif_rate(net, nid, current_time=100, lamda=1, **kwargs):
    """
    Returns the rate parameter of an Exponential distribution proportional to the time difference between the current time and the last time the node was traced.

    Args:
        net (tracing.Network): The network object.
        nid (int): The ID of the node.
        lamda (float, optional): The multiplicative factor of the rate parameter. Defaults to 1.
        current_time (float, optional): The current time. Defaults to 100.
        **kwargs: Additional keyword arguments.

    Returns:
        float: The rate parameter for the Exponential distribution.
    """
    exp_param = lamda * (current_time - net.traced_time[nid])
    return exp_param


def expFactorTimesCountImportance(net, nid, state='T', base=0, **kwargs):
    """
    Sample from an Exponential distribution with rate parameter calculated as `base` + `counts` * `importance`.
    As such, the neighbour counts get weighted by the Network-specific `count_importance`.

    Args:
        net (tracing.Network): A network object.
        nid (int): The ID of the node.
        state (str): The state of the neighbors to consider. Default is 'I'.
        base (float): The base value. Default is 0.
        **kwargs: Additional keyword arguments.

    Returns:
        float: A random sample from the Exponential distribution.
    """
    exp_param = base + net.count_importance * net.node_counts[nid][state]
    if exp_param:
        return -math.log(1. - random.random()) / exp_param
    return float('inf')


def expFactorTimesCountImportance_rate(net, nid, state='T', base=0, **kwargs):
    """
    Returns the rate parameter of an Exponential distribution calculated as `base` + `counts` * `importance`.
    As such, the neighbour counts get weighted by the Network-specific `count_importance`.

    Args:
        net (tracing.Network): A network object.
        nid (int): The ID of the node.
        state (str): The state of the neighbors to consider. Default is 'I'.
        base (float): The base value. Default is 0.
        **kwargs: Additional keyword arguments.

    Returns:
        float: The rate parameter for the Exponential distribution.
    """
    exp_param = base + net.count_importance * net.node_counts[nid][state]
    return exp_param


def expFactorTimesCountMultiState(net, nid, states=['I'], lamda=1, rel_states=[], rel=1, **kwargs):
    """
    Sample from an Exponential distribution with rate parameter being a factor times count of multiple neighboring states.
    The states can be organized into two categories: states and relative states, each having a different influence on the rate.
    For example, state 'Is' can be more infectious than 'I' and 'Ia'.
    The rate parameter is calculated as `lamda` * (`counts` * `states` + `rel` * `counts` * `rel_states`).

    Args:
        net (tracing.Network): The network object.
        nid (int): The ID of the node.
        states (list of str, optional): List of states for neighbors to be considered. Defaults to ['I'].
        lamda (float, optional): The multiplicative factor of the rate parameter. Defaults to 1.
        rel_states (list of str, optional): List of states for neighbors to be considered for relative importance. Defaults to [].
        rel (float, optional): Relative importance. Defaults to 1.
        **kwargs: Additional keyword arguments.

    Returns:
        float: A random sample from the Exponential distribution.

    Raises:
        ValueError: If `exp_param` is zero.

    Example:
        >>> net = Network()
        >>> nid = 0
        >>> states = ['I', 'S']
        >>> lamda = 0.5
        >>> rel_states = ['R']
        >>> rel = 0.1
        >>> exp_sample = expFactorTimesCountMultiState(net, nid, states, lamda, rel_states, rel)
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
    else:
        raise ValueError("Exponential parameter is zero.")


def expFactorTimesCountMultiState_rate(net, nid, states=['I'], lamda=1, rel_states=[], rel=1, **kwargs):
    """
    Returns the rate parameter of an Exponential distribution calculated as a factor times count of multiple neighboring states.
    The states can be organized into two categories: states and relative states, each having a different influence on the rate.
    For example, state 'Is' can be more infectious than 'I' and 'Ia'.
    The rate parameter is calculated as `lamda` * (`counts` * `states` + `rel` * `counts` * `rel_states`).

    Args:
        net (tracing.Network): The network object.
        nid (int): The ID of the node.
        states (list of str, optional): List of states for neighbors to be considered. Defaults to ['I'].
        lamda (float, optional): The multiplicative factor of the rate parameter. Defaults to 1.
        rel_states (list of str, optional): List of states for neighbors to be considered for relative importance. Defaults to [].
        rel (float, optional): Relative importance. Defaults to 1.
        **kwargs: Additional keyword arguments.

    Returns:
        float: The rate parameter for the Exponential distribution.

    Raises:
        ValueError: If `exp_param` is zero.

    Example:
        >>> net = Network()
        >>> nid = 0
        >>> states = ['I', 'S']
        >>> lamda = 0.5
        >>> rel_states = ['R']
        >>> rel = 0.1
        >>> exp_param = expFactorTimesCountMultiState_rate(net, nid, states, lamda, rel_states, rel)
    """
    exp_param_states = exp_param_rel_states = 0
    counts = net.node_counts[nid]
    for state in states:
        exp_param_states += counts[state]
    for state in rel_states:
        exp_param_rel_states += counts[state]
    exp_param = lamda * (exp_param_states + rel * exp_param_rel_states)
    return exp_param


### Methods for computing various statistics from the Simulation results ###

def get_statistics(data, compute='mean', axis=0, avg_without_idx=None, round_to=2):
    """
    The function takes an array of data and returns a list of dictionaries containing various statistics of each column (or row, depending on axis). 
    The statistics can be mean, standard deviation, boxplot parameters, or a combination of them. 
    Optionally, it can also exclude some rows from the calculation, transpose the data, and round the results to a specified number of decimal places.

    Args:
        data: a list or a numpy array of numerical data. It can be one-dimensional or two-dimensional.
        compute: a string specifying what statistics to compute. It can be one of the following values:
            - 'mean': compute the mean and standard deviation of each column.
            - 'mean+wo': compute the mean and standard deviation of each column, as well as the mean and standard deviation without the rows specified by avg_without_idx.
            - 'boxplot': compute the boxplot parameters (minimum, first quartile, median, third quartile, maximum, outliers) of each column.
            - 'boxplot+wo': compute the boxplot parameters of each column, as well as the boxplot parameters without the rows specified by avg_without_idx.
            - 'mean_boxplot': compute the mean, standard deviation, and boxplot parameters of each column.
            - 'mean+wo_boxplot': compute the mean, standard deviation, and boxplot parameters of each column, as well as the mean, standard deviation, and boxplot parameters 
                without the rows specified by avg_without_idx.
            - 'mean_boxplot+wo': compute the mean, standard deviation, and boxplot parameters of each column, as well as the boxplot parameters without the rows specified 
                by avg_without_idx.
            - 'all': compute the mean, standard deviation, and boxplot parameters of each column, as well as the mean, standard deviation, and boxplot parameters without the rows 
                specified by avg_without_idx.
        axis: an integer specifying the axis along which to compute the statistics. If 0 (default), the statistics are computed for each column of data. 
            If 1, the statistics are computed for each row of data.
        avg_without_idx: a list of indices to be excluded from the calculation. If None (default), all rows are included. 
            If an empty list, the option is ignored and the same results as for_data are returned.
        round_to: an integer specifying the number of decimal places to round the results to. Default is 2.

    Returns:
        result_list (list[dict]): a list of dictionaries, one for each column (or row, depending on axis) of data. 
            Each dictionary has the keys corresponding to `get_means_and_std` and `get_boxplot_statistics`.
    """
    for_data = np.array(data)
    if axis:
        for_data = for_data.T
        
    result_list = []
    
    if compute == 'mean':
        result_list = get_means_and_std(for_data, None, round_to)
    elif compute == 'mean+wo':
        result_list = get_means_and_std(for_data, avg_without_idx, round_to)
    elif compute == 'boxplot':
        result_list = get_boxplot_statistics(for_data, None, round_to)
    elif compute == 'boxplot+wo':
        result_list = get_boxplot_statistics(for_data, avg_without_idx, round_to)
    else:
        if compute == 'mean_boxplot':
            result_list = get_means_and_std(for_data, None, round_to)
            boxplot_stats_list = get_boxplot_statistics(for_data, None, round_to)
        elif compute == 'mean+wo_boxplot':
            result_list = get_means_and_std(for_data, avg_without_idx, round_to)
            boxplot_stats_list = get_boxplot_statistics(for_data, None, round_to)
        elif compute == 'mean_boxplot+wo':
            result_list = get_means_and_std(for_data, None, round_to)
            boxplot_stats_list = get_boxplot_statistics(for_data, avg_without_idx, round_to)
        elif compute == 'all':
            result_list = get_means_and_std(for_data, avg_without_idx, round_to)
            boxplot_stats_list = get_boxplot_statistics(for_data, avg_without_idx, round_to)
        
        for i in range(len(result_list)):
            result_list[i].update(boxplot_stats_list[i])
            
    return result_list


def get_means_and_std(for_data, avg_without_idx=None, round_to=2):
    """
    The function takes an array of data and returns a list of dictionaries containing the mean and standard deviation of each column. 
    Optionally, it can also exclude some rows from the calculation and round the results to a specified number of decimal places.

    Args:
        for_data: a numpy array of numerical data. It can be one-dimensional or two-dimensional.
        avg_without_idx: a list of indices to be excluded from the calculation. If None (default), all rows are included. 
            If an empty list, the option is ignored and the same results as for_data are returned.
        round_to: an integer specifying the number of decimal places to round the results to. Default is 2.

    Returns:
        results (list[dict]): a list of dictionaries, one for each column of for_data. Each dictionary has the following keys:
            - mean: the mean of the column, rounded to round_to decimal places.
            - std: the standard deviation of the column, rounded to round_to decimal places. If the column has only one element, the standard deviation is zero.
            - mean_wo: the mean of the column without the rows specified by avg_without_idx, rounded to round_to decimal places. Only present if avg_without_idx is not None and not empty.
            - std_wo: the standard deviation of the column without the rows specified by avg_without_idx, rounded to round_to decimal places. 
                Only present if avg_without_idx is not None and not empty. If the column has only one element after excluding the rows, the standard deviation is zero.
    """
    shape_dat = for_data.shape
    # compute means and standard devs - if more than 1 element ddof=1, otherwise just 0
    # if ndim = 1 output array with one element
    if len(shape_dat) == 1:
        # note that the full mean is computed during the boxplot routine, so we skip that
        means = [np.mean(for_data)]
        stds = [np.std(for_data, ddof=1) if shape_dat[0] > 1 else 0]
    else:
        # note that the full mean is computed during the boxplot routine, so we skip that
        means = np.mean(for_data, axis=0)
        stds = np.std(for_data, axis=0, ddof=1) if shape_dat[0] > 1 else [0] * shape_dat[1]
    
    # record 'wo' only if avg_without_idx exists and it does not cover the whole range of for_data in the first dim
    is_record_avg_without = (avg_without_idx is not None and len(for_data) > len(avg_without_idx))

    # Record statistics wihtout indexes covered by avg_without_idx if not all indexes are to be removed
    if is_record_avg_without:
        if avg_without_idx:
            data_without_idx = np.delete(for_data, avg_without_idx, axis=0)
            shape_dat_without = data_without_idx.shape
            if len(shape_dat_without) == 1:
                means_without = [np.mean(data_without_idx)]
                stds_without = [np.std(data_without_idx, ddof=1) if shape_dat_without[0] > 1 else 0]
            else:
                means_without = np.mean(data_without_idx, axis=0)
                stds_without = np.std(data_without_idx, axis=0, ddof=1) if shape_dat_without[0] > 1 else [0] * shape_dat_without[1]
        # Note: if avg_without_idx = [] it means the option was selected from args, but no simulation early stopped
        # In this case, we report the alternative: actual mean/std in order to be consistent across runs
        else:
            means_without = means
            stds_without = stds
            
    # this list will hold the statistic dict results
    results = []
    # iterate through the statistics
    for i in range(len(means)):
        dct = {}
        dct['mean'] = round(means[i], round_to)
        dct['std'] = round(stds[i], round_to)
        if is_record_avg_without:
            dct['mean_wo'] = round(means_without[i], round_to)
            dct['std_wo'] = round(stds_without[i], round_to)
        results.append(dct)
        
    return results


def get_boxplot_statistics(for_data, avg_without_idx=None, round_to=2):
    """
    Computes the mean, quartiles and whiskers for a given dataset and returns the statistics as a list of dictionaries.
    
    Args:
        for_data (numpy.ndarray): the dataset for which the statistics are to be computed.
        avg_without_idx (list or None): a list of indices to exclude from the computation of statistics. Default is None.
        round_to (int): the number of decimal places to round the statistics to. Default is 2.
    
    Returns:
        results (list[dict]): a list of dictionaries, one for each column of for_data, containing the statistics for that column. 
            Each dictionary has the following keys:
            - q1: the first quartile (25th percentile) of the column, rounded to round_to decimal places.
            - med: the median (50th percentile) of the column, rounded to round_to decimal places.
            - q3: the third quartile (75th percentile) of the column, rounded to round_to decimal places.
            - whislo / whishi: low and high outlier limits of the column, rounded to round_to decimal places. Outliers are values that are more than 1.5 times 
                the interquartile range (q3 - q1) away from the nearest quartile.
            - q1_wo: the first quartile (25th percentile) of the column without the rows specified by avg_without_idx, rounded to round_to decimal places. 
                Only present if compute contains '+wo' and avg_without_idx is not None and not empty.
            - med_wo: the median (50th percentile) of the column without the rows specified by avg_without_idx, rounded to round_to decimal places. 
                Only present if compute contains '+wo' and avg_without_idx is not None and not empty.
            - q3_wo: the third quartile (75th percentile) of the column without the rows pecified by avg_without_idx, rounded to round_to decimal places. 
                Only present if compute contains '+wo' and avg_without_idx is not None and not empty.
            - whihlo_wo / whishi_wo: low and high outlier limits of the column without the rows specified by avg_without_idx, rounded to round_to decimal places. 
                Only present if compute contains '+wo' and avg_without_idx is not None and not empty.
    """
    # boxplot computes mean, quartiles and whiskers
    B = plt.boxplot(for_data)
    # Record medians for each axis
    meds = B['medians']
    # there are 2 whiskers (low, high) per axis !!!
    whisks = B['whiskers']
    
    # record 'wo' only if avg_without_idx exists and it does not cover the whole range of for_data in the first dim
    is_record_avg_without = (avg_without_idx is not None and len(for_data) > len(avg_without_idx))
    
    # Record statistics wihtout indexes covered by avg_without_idx if not all indexes are to be removed
    if is_record_avg_without:
        if avg_without_idx:
            data_without_idx = np.delete(for_data, avg_without_idx, axis=0)
            if data_without_idx.size:
                B = plt.boxplot(data_without_idx)
                # Record medians for each axis
                meds_wo = B['medians']
                # there are 2 whiskers (low, high) per axis !!!
                whisks_wo = B['whiskers']
        # we also deal with the case in which avg_without_idx = [], in which case the "wo" values preserve the "with" values
        else:
            meds_wo = meds
            whisks_wo = whisks
            
    # close the plots objects
    plt.close()
    
    # this list will hold the statistic dict results
    results = []
    # iterate through the statistics
    for i in range(len(meds)):
        dct = {}
        whisk1 = whisks[2*i].get_ydata()
        whisk2 = whisks[2*i+1].get_ydata()
        dct['whislo'] = round(whisk1[1], round_to)
        dct['q1'] = round(whisk1[0], round_to)
        dct['med'] = round(meds[i].get_ydata()[0], round_to)
        dct['q3'] = round(whisk2[0], round_to)
        dct['whishi'] = round(whisk2[1], round_to)
        
        if is_record_avg_without:
            whisk1 = whisks_wo[2*i].get_ydata()
            whisk2 = whisks_wo[2*i+1].get_ydata()
            dct['whislo_wo'] = round(whisk1[1], round_to)
            dct['q1_wo'] = round(whisk1[0], round_to)
            dct['med_wo'] = round(meds_wo[i].get_ydata()[0], round_to)
            dct['q3_wo'] = round(whisk2[0], round_to)
            dct['whishi_wo'] = round(whisk2[1], round_to)
        
        results.append(dct)
        
    return results

    
### Workaround for TQDM to allow printing together with the progress line in the same STDOUT ###

def tqdm_print(*args, sep=' ', **kwargs):
    """
    Prints the given arguments to the console using tqdm's write method.

    Args:
        *args: The arguments to print.
        sep (str): The separator to use between arguments. Defaults to ' '.
        **kwargs: Additional keyword arguments to pass to tqdm's write method.
    """
    to_print = sep.join(str(arg) for arg in args)
    tqdm.tqdm.write(to_print, **kwargs)


@contextmanager
def redirect_to_tqdm():
    """
    Context manager to allow tqdm.write to replace the print function
    """   
    # Store builtin print
    old_print = print
    try:
        # Globaly replace print with tqdm.write
        inspect.builtins.print = tqdm_print
        yield
    finally:
        inspect.builtins.print = old_print

        
def tqdm_redirect(*args, **kwargs):
    """
    A generator function that redirects the output of tqdm progress bar to stdout.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Yields:
        x: The next value in the iterable.
    """
    with redirect_to_tqdm():
        for x in tqdm.tqdm(*args, file=stdout, **kwargs):
            yield x


### Methods for converting figures to black-white for journal printing ###

def set_ax_bw(ax, colors=None, markersize=3.):
    """
    Take each Line2D in the axes in `ax` and converts its line style to be suitable for black and white viewing.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to adjust.
        colors (list, optional): A list of colors to be mapped to black and white styles.
            If None, `plt.rcParams['axes.prop_cycle'].by_key()['color']` will be used.
        markersize (float, optional): The size of the markers to use for the lines.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    if colors:
        if len(colors) < 8:
            colors += [None] * (8 - len(colors))
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
    color_map = {
        colors[5]: {'marker': None, 'dash': (None,None)},
        colors[2]: {'marker': None, 'dash': [5,5]},
        colors[7]: {'marker': None, 'dash': [5,3,1,3]},
        colors[4]: {'marker': None, 'dash': [1,3]},
        colors[3]: {'marker': None, 'dash': [5,2,5,2,5,10]},
        colors[1]: {'marker': None, 'dash': [5,3,1,2,1,10]},
        colors[0]: {'marker': None, 'dash': [1,2,1,10]},
        colors[6]: {'marker': 'o', 'dash': (None,None)},
    }

    lines_to_adjust = ax.get_lines()
    try:
        lines_to_adjust += ax.get_legend().get_lines()
    except AttributeError:
        pass

    for line in lines_to_adjust:
        orig_color = line.get_color()
        if orig_color != 'black':
            try:
                line.set_color('black')
                line.set_dashes(color_map[orig_color]['dash'])
                line.set_marker(color_map[orig_color]['marker'])
                line.set_markersize(markersize)
            except KeyError:
                pass
            
    boxplots = ax.artists
    for box in boxplots:
        box.set_edgecolor('black')
        box.set_facecolor('white')
    
    # Added code to change the color of the mean indicator
    means = ax.findobj(match=lambda x: type(x) == plt.Line2D)
    for mean in means:
        mean.set_markerfacecolor('darkgray')


def set_fig_bw(fig, colors=None, markersize=3.):
    """
    Take each axes in the figure, and apply `set_ax_bw` to give it a black and white view.

    Args:
        fig (matplotlib.figure.Figure): The figure object to modify.
        colors (list, optional): A list of colors to be mapped to black and white styles.
            If None, `plt.rcParams['axes.prop_cycle'].by_key()['color']` will be used.
        markersize (float, optional): The size of the markers to use for scatter plots.

    Returns:
        None
    """
    for ax in fig.get_axes():
        set_ax_bw(ax, colors, markersize)


### Debugging methods ###
            
def rel(*modules):
    """
    Reloads the specified modules.

    Args:
        *modules: One or more modules to reload.
    """
    for module in modules:
        importlib.reload(module)
     

def pvar(*var, owners=True, nan_value=None):
    """
    Prints the names and values of the variables specified in the current frame. Owning frames can also be displayed. 
    Not-set values (=nan_value) are overwritten with `NotYetDefined`.

    Args:
        *var (any): The variables to be printed. If '\n' is passed, a new line is printed.
        owners (bool, optional): If True, the owning frames of the variables are also displayed. 
            If False, only the last part of the names are displayed. The default is True.
        nan_value (any, optional): The value to be used for variables that are not set. The default is None.

    Returns:
        None

    Examples:
        >>> x = 1
        >>> y = 2
        >>> z = 3
        >>> pvar(x, y, z)
        x = 1, y = 2, z = 3
        >>> pvar(x, y, z, owners=False)
        x = 1, y = 2, z = 3
        >>> pvar(x, y, z, nan_value=0)
        x = 1, y = 2, z = 3
        >>> pvar(x, '\n', y, '\n', z)
        x = 1
        y = 2
        z = 3
    """
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    fi = r.split(', ')
    last_var = len(var) - 1
    for i, x in enumerate(var):
        if x == '\n':
            print()
        else:
            if x == nan_value:
                x = 'NotYetDefined'
            if not owners:
                fi[i] = fi[i].split('.')[-1]
            format_to_print = "{} = {}, "
            if i == last_var:
                format_to_print = format_to_print[:-2]
            print(format_to_print.format(fi[i], x), end="", flush=True)
    print()   
            

def animate_cell_capture(capture, filename=None, fps=1, quality=10.):
    """
    Create an animation from a cell capture object and save it to a file.

    Args:
        capture (cell_capture.CellCapture): The cell capture object to animate.
        filename (str, optional): The name of the file to save the animation to. Defaults to 'fig/simulation.gif'.
        fps (int, optional): The frames per second of the animation. Defaults to 1.
        quality (float, optional): The quality of the animation. Only used if the file format is not '.gif'. Defaults to 10.0.
    """
    # import imageio here to avoid dependency if no animation outputting is performed
    try:
        from imageio import mimsave
    except ImportError:
        raise ImportError("The `imageio` module is needed to perform this operation. Please install it using pip or conda.")
    kwargs_write = {'fps': fps}
    if not filename:
        filename = 'fig/simulation.gif'
    if filename.endswith('.gif'):
        kwargs_write.update({'quantizer': 2})
    else:
        kwargs_write.update({'quality': quality})
    if not filename.startswith('fig/'):
        filename = 'fig/' + filename
    images_data = []
    for img_data in capture.outputs:
        png = img_data._repr_png_()
        if isinstance(png, str):
            png = b64decode(png)
        images_data.append(Image.open(io.BytesIO(png)))
    mimsave(filename, images_data, **kwargs_write)
    

def pkl(obj, filename=None):
    """
    Pickles the given object and saves it to a file with the given filename in the 'saved' folder.
    If no filename is provided, the filename is inferred from the calling function's name.
    The pickled object is saved in the 'saved' directory with a '.pkl' extension.
    """
    if not filename:
        frame = inspect.currentframe().f_back
        s = inspect.getframeinfo(frame).code_context[0]
        filename = re.search(r"\((.*)\)", s).group(1)
    filename = 'saved/' + filename + '.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

def get_pkl(obj=None):
    """
    Load and return a pickled object from a file located in the 'saved' folder.

    Args:
        obj (str): The name of the object to load from the file. If None, the name is inferred from the calling code.

    Returns:
        The unpickled object.

    Raises:
        FileNotFoundError: If the file for the given object name does not exist.
        pickle.UnpicklingError: If there is an error unpickling the object from the file.
    """
    if not isinstance(obj, str):
        frame = inspect.currentframe().f_back
        s = inspect.getframeinfo(frame).code_context[0]
        obj = re.search(r"\((.*)\)", s).group(1)
    filename = 'saved/' + obj + '.pkl'
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
    
    
def get_json(json_or_path):
    """
    Load JSON data from either a JSON string or a file path.

    Args:
        json_or_path (str): A JSON string or a file path to a JSON file.

    Returns:
        dict: A dictionary containing the parsed JSON data.

    Raises:
        JSONDecodeError: If the input is not valid JSON.
        FileNotFoundError: If the input file path does not exist.
    """
    try:
        return json.loads(json_or_path)
    except (json.JSONDecodeError, FileNotFoundError):
        with open(json_or_path, 'r', encoding="utf8") as handle:
            return json.loads(handle.read())
        
        
def process_json_results(path='tracing_runs/default_id/', keys=('pa', 'dual'), bottom_keys=('uptake', 'overlap', 'taut', 'taur'), print_id_fail=True, round_to=2):
    """
    Processes the JSON results of simulation runs and returns a nested dictionary containing the aggregated statistics.

    Args:
        path (str, optional): The folder path of the JSON files.
        keys (tuple, optional): The keys to use for the top level of the nested dictionary. Defaults to ('pa', 'dual').
        bottom_keys (tuple, optional): The keys to use for the bottom level of the nested dictionary. Defaults to ('uptake', 'overlap', 'taut', 'taur'). 
            We recommend changing this rarely, if ever, to maintain consistency across runs at the bottom level.
        print_id_fail (bool, optional): Whether to print the ID of any failed JSON files. Defaults to True.

    Returns:
        dict: A nested dictionary containing any simulation results that could be processed correctly (without raising JSON.DecodeError).

    Notes:
        The JSON files expect the following keys: 
            - 'args' containing the arguments used to run the experiment
            - 'res' containing the results for a single `taut` value OR the `taut` values themselves (if more than one was used in the runs)
        This method does not fail upon encountering a JSON file that cannot be processed. Instead, it prints the ID of the file, if `print_id_fail`,
        and continues processing the rest of the files.
    """
    if not path.endswith('*.json'):
        path += '*.json'
    files = glob.glob(path)
    if not files:
        print('No JSON files found in the specified path.')
        return None
    nested_dict = lambda: defaultdict(nested_dict)
    all_sim_res = nested_dict()
    for file in files:
        try:
            json_file = get_json(file)
            args = json_file['args']
            # note, `taut` may be a collection of values, so we need to handle it differently
            taut_vals = np.atleast_1d(args['taut'])
            top_keys = tuple(key if key == 'taut' else (round(args[key], round_to) if isinstance(args[key], float) else args[key])
                            for key in keys if key in args)
            btm_keys = tuple(key if key == 'taut' else (round(args[key], round_to) if isinstance(args[key], float) else args[key])
                            for key in bottom_keys if key in args)
            
            for taut in taut_vals:
                # if only one taut value was provided, results in newer versions are found under key 'res'
                try:
                    results = json_file['res']
                # for multiple taut values or older versions, the results are under keys str(taut)
                except KeyError:
                    # The following try block is needed to cover for the case in which the supplied taut is either float/int
                    try:
                        results = json_file[str(taut)]
                    except KeyError:
                        results = json_file[str(int(taut))]
                taut = round(taut, round_to)
                current_level = all_sim_res
                for key in top_keys:
                    if key == 'taut':
                        current_level = current_level[taut]
                    else:
                        current_level = current_level[key]
                for key in btm_keys:
                    if key == 'taut':
                        current_level = current_level[taut]
                    else:
                        current_level = current_level[key]
                current_level.update(results)
                                    
        except json.JSONDecodeError:
            if print_id_fail:
                print(int(re.findall('id(.*?)_', file)[0]), end=',')
                
    return all_sim_res


### General purpose classes for Event recording, profiling code, encoding numpy arrays into JSON, ### 
### delegating functions to list members and creating non-daemon processes. ###

class Event(dict):
    """
    General JSON-like class for recording information about various Events. Supports both simple dict API and OOP dot notation.
    """
    def __init__(self, **kwargs):
        self.update(kwargs)
        
    def __repr__(self):
        """
        Returns a string representation of the object.

        The returned string is of the form '<ClassName>(key1=value1, key2=value2, ...)'.
        """
        items = (f"{k}={v!r}" for k, v in self.items())
        return "{}({})".format(type(self).__name__, ", ".join(items))
            
    def __getattr__(self, item):
        """
        Get the value of the attribute with the given name.

        If the attribute is not found in the object's dictionary, a KeyError is initially raised, caught, and then an AttributeError is raised.
        If the attribute is found in the object's dictionary, its value is returned.

        Args:
            item (str): The name of the attribute to get.

        Returns:
            The value of the attribute with the given name.

        Raises:
            AttributeError: If the attribute is not found in the object's dictionary.
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    # The following methods are copied over from dict to support the OOP dot notation API.      
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Profiler(Profile):
    """ 
    Custom Profile class with a __call__() context manager method to enable profiling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable()  # Profiling initially off.

    @contextmanager
    def __call__(self):
        """
        Enables the profiler, executes the code to be profiled, and then disables the profiler.

        Usage:
            profiler = MyProfiler()
            with profiler():
                # Code to be profiled goes here.
        """
        self.enable()
        yield  # Execute code to be profiled.
        self.disable()
        
    def stat(self, stream=None):
        """
        Prints statistics for the current profiling session.

        Args:
            stream: The output stream to print the statistics to. Defaults to stdout.
        """
        if not stream:
            stream = stdout
        stats = Stats(self, stream=stream)
        stats.strip_dirs().sort_stats('cumulative', 'time')
        stats.print_stats()
     
    
class NumpyEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that can handle NumPy arrays and sets.

    This encoder extends the default JSON encoder to handle NumPy arrays and sets,
    which are not natively supported by the encoder. NumPy arrays are converted to
    lists using the `tolist()` method, while sets are converted to lists using the
    built-in `list()` function.

    Usage:
    ```
    import json
    import numpy as np

    data = {
        'array': np.array([1, 2, 3]),
        'set': set([4, 5, 6])
    }

    json_data = json.dumps(data, cls=NumpyEncoder)
    ```
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)
    

class ListDelegator(list):
    """
    A list subclass that delegates method calls to its elements.

    This class allows one to call a method on all elements of the list at once.
    If any of the elements do not have the method, an AttributeError is raised.

    Attributes:
        args_list (tuple): A tuple of additional arguments to be passed to the list constructor.

    Methods:
        __getattr__(self, method_name): Delegates method calls to the elements of the list.
        draw(self, *args, ax=[None, None], **kw): Calls the 'draw' method on the first two elements of the list.
    """    
    def __init__(self, *args, args_list=()):
        all_args = args + tuple(args_list)
        list.__init__(self, all_args)

    def __getattr__(self, method_name):
        if self and hasattr(self[0], method_name):
            def delegator(self, *args, **kw):
                results = ListDelegator()
                for element in self:
                    results.append(getattr(element, method_name)(*args, **kw))
                # return the list of results only if there's any non-None value (equivalent to ignoring void returns)
                return results if results != [None] * len(results) else None
            # add the method to the class s.t. it can be called quicker the second time
            setattr(ListDelegator, method_name, delegator)
            # returning the attribute rather than the delegator func directly due to "self" inconsistencies
            return getattr(self, method_name)
        else:
            error = "Could not find '" + method_name + "' in the attributes list of this ListDelegator's elements"
            raise AttributeError(error)
            
    def draw(self, *args, ax=[None, None], **kwargs):
        for i in range(len(self)):
            self[i].draw(*args, ax=ax[i], **kwargs)


class NoDaemonProcess(multiprocessing.Process):
    """A process class that is never daemonized.

    This class is a monkey-patch of the `multiprocessing.Process` class to ensure that it is never daemonized. 
    It also includes a workaround to allow subprocesses to be spawned without the `AuthenticationString` raising an error.

    Attributes:
        daemon (bool): Whether or not the process is a daemon. Always returns False.
    """
    @property
    def daemon(self):
        """
        Mark process as always non-daemon.
        """
        return False

    @daemon.setter
    def daemon(self, val):
        """
        Disable setting daemon status.
        """
        pass
    
    def __getstate__(self):
        """called when pickling - this workaround allows subprocesses to 
           be spawned without the AuthenticationString raising an error"""
        state = self.__dict__.copy()
        conf = state['_config']
        if 'authkey' in conf: 
            conf['authkey'] = bytes(conf['authkey'])
        return state

    def __setstate__(self, state):
        """for unpickling"""
        state['_config']['authkey'] = multiprocessing.process.AuthenticationString(state['_config']['authkey'])
        self.__dict__.update(state)

        
def get_pool(pytorch=False, daemon=True, set_spawn=False):
    """
    Returns a multiprocessing pool object with or without daemon processes.

    Args:
        pytorch (bool, optional): If True, returns a PyTorch multiprocessing pool object. Defaults to False.
        daemon (bool, optional): If True, returns a pool object with daemon processes. Defaults to True.
        set_spawn (bool, optional): If True and pytorch is True, sets the start method to 'spawn'. Defaults to False.

    Returns:
        multiprocessing.pool.Pool: A multiprocessing pool object.
    """
    if pytorch:
        from torch import multiprocessing as mp
        if set_spawn:
            mp.set_start_method('spawn', force=True)
        if daemon:
            return mp.Pool
    else:
        if daemon:
            return multiprocessing.Pool
        
    from multiprocessing.pool import Pool    
    class NoDaemonPool(Pool):
        def Process(self, *args, **kwds):
            proc = super().Process(*args, **kwds)
            proc.__class__ = NoDaemonProcess
            return proc

    return NoDaemonPool