import numpy as np
from random import random
from collections import defaultdict

from lib.utils import Event
from lib.exp_sampler import get_sampler

class SimEvent(Event):
    pass

class Simulation():
    
    def __init__(self, net, trans, isolate_S=True, trace_once=False, already_exp=True, presample=1000):
        self.net = net
        self.trans = trans
        self.isolate_S = isolate_S
        self.trace_once = trace_once
        self.time = 0
        # Create Adapter to allow the rate returned by rate_func to be either a base rate (which shall be exponentially sampled) 
        # or an exponential rate directly (which shall be left unchanged)
        if already_exp:
            self.sampling = lambda rate: rate
        else:
            sampler = get_sampler(presample_size=presample, scale_one=True)
            self.sampling = lambda rate: (sampler.get_next_sample(rate) if rate else float('inf'))
        # variables for Gillespie sampling
        # node that gets invalidated after an update
        self.last_updated = -1
        # the current base rates for each of the node
        self.lamdas = defaultdict(list)

    def get_next_event(self):
        """
        Sample next event using CT Monte Carlo
        This is used to sample infection events and trace events if args.separate_traced=False; otherwise only infection events
        """
        
        id_next = None
        from_next = None
        to_next = None
        best_time = float('inf')
        
        # making local vars for efficiency
        net = self.net
        trans = self.trans
        time = self.time
        sampling = self.sampling
        isolate_S = self.isolate_S
        node_list = net.node_list
        node_states = net.node_states
        node_traced = net.node_traced
                
        for nid in node_list:
            current_state = node_states[nid]
            # if the current node is traced, it should not be possible to move from 'S'
            if isolate_S and node_traced[nid] and current_state == 'S':
                continue
            for to, rate_func in trans[current_state]:
                # rate_func is a lambda expression waiting for a net and a node id
                rate = sampling(rate_func(net, nid))
                # increment time with the current rate
                trans_time = time + rate
                if trans_time < best_time:
                    best_time = trans_time
                    from_next = current_state
                    to_next = to
                    id_next = nid
                
        if id_next is None:
            return None
        
        return SimEvent(node=id_next, fr=from_next, to=to_next, time=best_time)
    
    def get_next_trace_event(self):
        """
        Sample trace event using CT Monte Carlo - this gets executed only if args.separate_traced=True
        """
        
        id_next = None
        from_next = None
        to_next = None
        best_time = float('inf')
        
        # local vars for efficiency
        net = self.net
        trans = self.trans
        time = self.time
        sampling = self.sampling
        trace_once = self.trace_once
        node_list = net.node_list
        node_traced = net.node_traced
        node_states = net.node_states
        traced_time = net.traced_time
        
        traceable_states = ['S', 'E', 'I', 'Ia', 'Is']
        traced_state = 'T'
        noncompliant_state = 'N'
        
        # Lazily initialize a dict with the correct transition functions for tracing (we need all 'state -> T -> func' entries)
        # This is a bit convoluted because the 'items' of the nested dict 'state -> state -> func' are cached for efficiency reasons
        # Hence the entries become 'state -> (state, func)'
        try:
            trace_funcs = self.trace_funcs
            noncompliance_rate_func = self.noncompliance
        except:
            # collect only the tracing function that actually exist in self.trans based on each traceable state
            trace_funcs = {
                s : dict(trans[s])[traced_state] for s in traceable_states if traced_state in dict(trans[s])
            }
            self.trace_funcs = trace_funcs
            
            # get the noncompliance function if one exists in transitions
            # also takes into account whether no transition to T has been defined, in which case no transition to N can happen
            noncompliance_rate_func = \
                dict(trans[traced_state]).get(noncompliant_state, None) if traced_state in trans else None
            self.noncompliance = noncompliance_rate_func
            
            
        # Filter out nodes that can actually be traced - i.e. not traced yet and from the traceable_states
        not_traced_inf = []
        # Collect here a list of the traced nodes that are still important if non-compliant (i.e. traceable_states)
        traced_inf = []
        for nid in node_list:
            # check if there is any tracing function instantiated before collecting non-traced points
            # Note: tracing multiple times can be disallowed through trace_once
            if trace_funcs and not node_traced[nid] and (not trace_once or not nid in traced_time) \
            and node_states[nid] in trace_funcs:
                not_traced_inf.append(nid)
            # check if there is a noncompliance rate func before collecting the traced points that are actually "dangerous"
            elif noncompliance_rate_func and node_traced[nid] and node_states[nid] in trace_funcs:
                traced_inf.append(nid)
                
        
        # look for tracing events
        for nid in not_traced_inf:
            current_state = node_states[nid]
            if current_state in trace_funcs:
                # rate_func is a lambda expression waiting for a net and a node id
                rate = sampling(trace_funcs[current_state](net, nid))
                # increment time with the current rate
                trans_time = time + rate
                if trans_time < best_time:
                    best_time = trans_time
                    from_next = current_state
                    to_next = traced_state
                    id_next = nid
        
        # look for noncompliance events       
        for nid in traced_inf:
            # rate_func is a lambda expression waiting for a net and a node id
            rate = sampling(noncompliance_rate_func(net, nid, time))
            # increment time with the current rate
            trans_time = time + rate
            if trans_time < best_time:
                best_time = trans_time
                from_next = traced_state
                to_next = noncompliant_state
                id_next = nid
                
        if id_next is None:
            return None

        return SimEvent(node=id_next, fr=from_next, to=to_next, time=best_time)
    
    
    def get_next_event_sample_only_minimum(self, trans_know, tracing_nets):
        """
        Sampling using Gillespie's Algorithm
        """
        
        # making local vars for efficiency
        net = self.net
        trans = self.trans
        time = self.time
        sampling = self.sampling
        trace_once = self.trace_once
        isolate_S = self.isolate_S
        node_list = net.node_list
        node_states = net.node_states
        node_traced = net.node_traced
        traced_time = net.traced_time

        traceable_states = ['S', 'E', 'I', 'Ia', 'Is']
        traced_state = 'T'
        noncompliant_state = 'N'
        
        # Lazily initialize a dict with the correct transition functions for tracing (we need all 'state -> T -> func' entries)
        # This is a bit convoluted because the 'items' of the nested dict 'state -> state -> func' are cached for efficiency reasons
        # Hence the entries are actually stored as 'state -> (state, func)'
        try:
            trace_funcs = self.trace_funcs
            noncompliance_rate_func = self.noncompliance
        except:
            # collect only the tracing function that actually exist in self.trans based on each traceable state
            trace_funcs = {
                s : dict(trans_know[s])[traced_state] for s in traceable_states if traced_state in dict(trans_know[s])
            }
            self.trace_funcs = trace_funcs
            
            # get the noncompliance function if one exists in transitions
            # also takes into account whether no transition to T has been defined, in which case no transition to N can happen
            noncompliance_rate_func = \
                dict(trans_know[traced_state]).get(noncompliant_state, None) if traced_state in trans_know else None
            self.noncompliance = noncompliance_rate_func
            
        
        last_updated = self.last_updated
        lamdas = self.lamdas
        # update propensity base rates only for the last updated and its neighbors; if first iter update for all
        nodes_for_updating = [last_updated] + list(net.neighbors(last_updated)) if last_updated >= 0 else node_list
        
        # loop through the list of 'invalidated' nodes - i.e. for which the rates need to be updated
        for nid in nodes_for_updating:
            # remove previous rates and possible transitions for the invalidated node
            lamdas.pop(nid, None)
        
            current_state = node_states[nid]
            current_traced = node_traced[nid]
            
            # update infection possible transitions lists
            if not (isolate_S and current_traced and current_state == 'S'):
                for to, rate_func in trans[current_state]:
                    lamdas[nid].append((rate_func(net, nid), to))
            
            # check if there is any possible tracing before updating the tracing possible transitions lists
            if trace_funcs and not current_traced and (not trace_once or not nid in traced_time) \
            and current_state in trace_funcs:
                for trace_net in tracing_nets:
                    lamdas[nid].append((trace_funcs[current_state](trace_net, nid), traced_state))
                
            # check if there is a noncompliance rate func before collecting the possible noncompliant transitions lists
            elif noncompliance_rate_func and current_traced and current_state in trace_funcs:
                for trace_net in tracing_nets:
                    lamdas[nid].append((noncompliance_rate_func(trace_net, nid, time), noncompliant_state))
        
        base_rates = []
        sum_lamdas = 0
        nodes_and_trans = []
        for nid, sublist in lamdas.items():
            for rate, to in sublist:
                sum_lamdas += rate
                base_rates.append(rate)
                nodes_and_trans.append((nid,to))
        
        # this corresponds to no more event possible
        if not sum_lamdas:
            return None
                
        # sampling from categorical distribution based on each lambda propensity func
        index_min = ((np.array(base_rates) / sum_lamdas).cumsum() >= random()).argmax()
        # select the corresponding node id and transition TO for the sampled categorical
        id_next, to_next = nodes_and_trans[index_min]
        from_next = node_states[id_next]
        # sampling next time of event from the sum of rates (Gillespie algorithm)
        best_time = time + sampling(sum_lamdas)
                
        return SimEvent(node=id_next, fr=from_next, to=to_next, time=best_time)
    
    
    def get_event_with_config(self, **kwargs):
        return SimEvent(**kwargs)
        
    def run_event(self, e):
        self.net.change_state_fast_update(e.node, e.to)
        self.time = e.time
        self.last_updated = e.node
        
    def run_event_no_update(self, e):
        self.net.change_state(e.node, e.to, update=False)
        self.time = e.time
        self.last_updated = e.node
        
    def run_trace_event(self, e, to_traced=True):
        self.net.change_traced_state_fast_update(e.node, to_traced, e.time)
        self.time = e.time
        self.last_updated = e.node        