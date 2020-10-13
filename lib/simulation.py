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
        for nid, current_state in enumerate(self.net.node_states):
            for to, rate_func in self.trans[current_state]:
                # rate_func is a lambda expression waiting for a net and a node id
                rate = rate_func(self.net, nid)
                # increment time with the current rate
                trans_time = self.time + rate
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
        
        traceable_states = ['S', 'E', 'I', 'Ia', 'Is']
        
        # Lazily initialize a dict with the correct transition functions for tracing (we need all state -> T -> func entries)
        # This is a bit convoluted because the nested dict state -> state -> func is cached for efficiency reasons
        # Hence the items are actually state -> (state, func) 
        try:
            trace_funcs = self.trace_funcs
        except:
            trace_funcs = {
                s : dict(self.trans[s])['T']
                for s in traceable_states
            }
            self.trace_funcs = trace_funcs

        # Filter out nodes that can actually be traced
        traced_nodes = np.array(self.net.node_traced)
        correct_state_nodes = np.isin(self.net.node_states, traceable_states)
        not_traced = np.where(~traced_nodes & correct_state_nodes)[0]
        
        for nid in not_traced:
            current_state = self.net.node_states[nid]
            # rate_func is a lambda expression waiting for a net and a node id
            rate = trace_funcs[current_state](self.net, nid)
            # increment time with the current rate
            trans_time = self.time + rate
            if trans_time < best_time:
                best_time = trans_time
                from_next = current_state
                to_next = 'T'
                id_next = nid
                
        if id_next is None:
            return None

        return SimEvent(node=id_next, fr=from_next, to=to_next, time=best_time)        
                
        
    def run_event(self, e):
        self.net.change_state_fast_update(e.node, e.to)
        self.time = e.time
        
    def run_trace_event(self, e):
        self.net.change_traced_state_fast_update(e.node)
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
        
        
    
        