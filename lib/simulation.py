from collections import defaultdict
import numpy as np

from lib.utils import Event

class SimEvent(Event):
    pass

class Simulation():
    
    def __init__(self, net, trans):
        self.net = net
        self.trans = trans
        self.time = 0
        
    def get_next_event(self):
        id_next = None
        from_next = None
        to_next = None
        best_time = float('inf')
        
        # making local vars for efficiency
        net = self.net
        trans = self.trans
        time = self.time
        node_list = net.node_list
        node_states = net.node_states
        node_traced = net.node_traced
        
        for nid in node_list:
            current_state = node_states[nid]
            # if the current node is traced, it should not be possible to move from 'S'
            if node_traced[nid] and current_state == 'S':
                continue
            for to, rate_func in trans[current_state]:
                # rate_func is a lambda expression waiting for a net and a node id
                rate = rate_func(net, nid)
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
        id_next = None
        from_next = None
        to_next = None
        best_time = float('inf')
        
        # local vars for efficiency
        net = self.net
        trans = self.trans
        time = self.time
        node_list = net.node_list
        node_traced = net.node_traced
        node_states = net.node_states
        
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
            # collect only the tracing function that actually exist in self.trans based on each state
            trace_funcs = {
                s : dict(trans[s])[traced_state] for s in traceable_states if traced_state in dict(trans[s])  
            }
            self.trace_funcs = trace_funcs
            
            # get the noncompliance function if one exists in transitions
            noncompliance_rate_func = dict(self.trans[traced_state])[noncompliant_state] if traced_state in self.trans else None
            self.noncompliace = noncompliance_rate_func

            
        # Filter out nodes that can actually be traced - i.e. not traced yet and from the traceable_states
        not_traced_inf = []
        # Also collect a list of the traced nodes that are still dangerous (E, I,)
        traced_inf = []
        for nid in node_list:
            # check if there is any tracing function instantiated before collection non-traced points
            if trace_funcs and not node_traced[nid] and node_states[nid] in traceable_states:
                not_traced_inf.append(nid)
            # check if there is a noncompliance rate func before collecting the traced points that are actually dangerous
            elif noncompliance_rate_func and node_traced[nid] and node_states[nid] in traceable_states:
                traced_inf.append(nid)
        
        # look for tracing events
        for nid in not_traced_inf:
            current_state = node_states[nid]
            # rate_func is a lambda expression waiting for a net and a node id
            rate = trace_funcs[current_state](net, nid)
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
            rate = noncompliance_rate_func(net, nid, time)
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
                
        
    def run_event(self, e):
        self.net.change_state_fast_update(e.node, e.to)
        self.time = e.time
        
    def run_trace_event(self, e, to_traced=True):
        self.net.change_traced_state_fast_update(e.node, to_traced, e.time)
        self.time = e.time
        
    def run_event_separate_traced(self, e):
        if e.to != 'T':
            self.net.change_state_fast_update(e.node, e.to)
        else:
            self.net.change_traced_state_fast_update(e.node)
        self.time = e.time
        
    def run_until(self, time_limit):
        if self.time >= time_limit:
            raise ValueError("WARNING: run_until called with a passed time")
        e = get_next_event()
        if e.time < time_limit:
            run_event(e)
        
        
    
        