import numpy as np
from random import random
from collections import defaultdict

from lib.utils import Event
from lib.exp_sampler import get_sampler


class SimEvent(Event):
    pass


class Simulation():
    
    def __init__(self, net, trans, isolate_s=True, trace_after=14, already_exp=True, presample=1000, **kwargs):
        self.net = net
        self.trans = trans
        self.isolate_S = isolate_s
        self.trace_after = trace_after
        self.time = 0
        # Create Adapter to allow the rate returned by rate_func to be either a base rate (which shall be exponentially sampled) 
        # or an exponential rate directly (which shall be left unchanged)
        if already_exp:
            self.sampling = lambda rate: rate
        else:
            sampler = get_sampler(presample_size=presample, scale_one=True)
            self.sampling = lambda rate: (sampler.get_next_sample(rate) if rate else float('inf'))
        # variables for Gillespie sampling
        # nodes that get invalidated after an update
        self.last_updated = []
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
        trace_after = self.trace_after
        node_list = net.node_list
        node_traced = net.node_traced
        node_states = net.node_states
        traced_time = net.traced_time
        noncomp_time = net.noncomp_time
        
        traceable_states = ['S', 'E', 'I', 'Ia', 'Is']
        traced_state = 'T'
        noncompliant_state = 'N'
        
        # Lazily initialize a dict with the correct transition functions for tracing (we need all 'state -> T -> func' entries)
        # This is a bit convoluted because the 'items' of the nested dict 'state -> state -> func' are cached for efficiency reasons
        # Hence the entries become 'state -> (state, func)'
        try:
            trace_funcs = self.trace_funcs
            noncompliance_rate_func = self.noncompliance
        except AttributeError:
            # collect only the tracing function that actually exist in self.trans based on each traceable state
            trace_funcs = {
                s: dict(trans[s])[traced_state] for s in traceable_states if traced_state in dict(trans[s])
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
            # if trace_after == -1 -> trace a node only if its NOT present in traced_time: aka never traced or legally exited isolation
            # if trace_after != -1 -> make tracing same node possible only after a time delay trace_after since becoming noncompliant N
            if trace_funcs and not node_traced[nid] and node_states[nid] in trace_funcs \
            and ((trace_after != -1 and time - noncomp_time.get(nid, time) > trace_after) or not nid in traced_time):
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
    
    
    def get_next_event_sample_only_minimum(self, trans_know, tracing_nets, **kwargs):
        """
        Sampling using Gillespie's Algorithm
        
        The Simulation object usually runs over one type of network (INFECTION or TRACING) based on one transition object;
        However, in Gillespie we need access to both the INFECTION and the TRACING rates in one place at the same time!
        To keep the class signature the same for all sampling methods, we pass the TRACING transition object and networks at 
        this method's invocation time on a Simulation object which already contains the INFECTION network and transition object.
        
        Regardless of the implementation, one should think of both the INFECTION and the TRACING objects as global variables here
        """
        
        # making local vars for efficiency
        net = self.net
        trans = self.trans
        time = self.time
        sampling = self.sampling
        trace_after = self.trace_after
        isolate_S = self.isolate_S
        node_list = net.node_list
        node_states = net.node_states
        node_traced = net.node_traced
        traced_time = net.traced_time
        noncomp_time = net.noncomp_time

        traceable_states = ['S', 'E', 'I', 'Ia', 'Is']
        traced_state = 'T'
        noncompliant_state = 'N'
        
        # Lazily initialize a dict with the correct transition functions for tracing (we need all 'state -> T -> func' entries)
        # This is a bit convoluted because the 'items' of the nested dict 'state -> state -> func' are cached for efficiency reasons
        # Hence the entries are actually stored as 'state -> (state, func)'
        try:
            trace_funcs = self.trace_funcs
            noncompliance_rate_func = self.noncompliance
        except AttributeError:
            # collect only the tracing function that actually exist in self.trans based on each traceable state
            trace_funcs = {
                s: dict(trans_know[s])[traced_state] for s in traceable_states if traced_state in dict(trans_know[s])
            }
            self.trace_funcs = trace_funcs
            
            # get the noncompliance function if one exists in transitions
            # also takes into account whether no transition to T has been defined, in which case no transition to N can happen
            noncompliance_rate_func = \
                dict(trans_know[traced_state]).get(noncompliant_state, None) if traced_state in trans_know else None
            self.noncompliance = noncompliance_rate_func
            
        
        lamdas = self.lamdas
        last_updated = self.last_updated
        # update propensity base rates only for the last updated and their neighbors; if first time here, update for all
        nodes_for_updating = last_updated + [neigh for update_id in last_updated for neigh in net.neighbors(update_id)] \
                                    if last_updated else node_list
        # clear the list of invalidated nodes for next update
        last_updated.clear()
        
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
            # if trace_after == -1 -> trace a node only if its NOT present in traced_time: aka never traced or legally exited isolation
            # if trace_after != -1 -> make tracing same node possible only after a time delay trace_after since becoming noncompliant N
            if trace_funcs and not current_traced and current_state in trace_funcs \
            and ((trace_after != -1 and time - noncomp_time.get(nid, time) > trace_after) or not nid in traced_time):
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
                nodes_and_trans.append((nid, to))
        
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
        self.last_updated.append(e.node)
        
    def run_event_no_update(self, e):
        self.time = e.time
        self.last_updated.append(e.node)
        
    def run_trace_event_for_trace_net(self, e, to_traced=True, legal_isolation_exit=False):
        self.net.change_traced_state_update_tracing(e.node, to_traced, e.time, legal_isolation_exit)
        self.time = e.time
        self.last_updated.append(e.node)
        
    def run_trace_event_for_infect_net(self, e, to_traced=True, legal_isolation_exit=False):
        self.net.change_traced_state_update_infectious(e.node, to_traced, e.time, legal_isolation_exit)
        self.time = e.time
        self.last_updated.append(e.node)