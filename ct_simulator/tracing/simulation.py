import numpy as np
from random import random
from collections import defaultdict
from bisect import insort

from .utils import Event
from .exp_sampler import get_sampler


class SimEvent(Event):
    """
    A simulation event that represents a change in state of a node in the simulation.

    Attributes:
        node (int): The ID of the node that the event is associated with.
        fr (int): The ID of the node that the event originates from.
        to (int): The ID of the node that the event transitions to.
    """

    def __repr__(self):
        return f'{self.node}: {self.fr} -> {self.to}'


class Simulation():
    """
    A class representing a simulation of a network with a given transition knowledgebase.
    
    Attributes:
        net (network.Network): a Network object representing the network to simulate
        trans (dict): a dictionary representing the transition knowledgebase
        violations (int): an integer representing the number of violations of the counts
        isolate_s (bool): a boolean indicating whether to isolate S nodes that are traced
        trace_after (int): an integer representing the time after which tracing a node can occur
        time (float): a float representing the current time of the simulation
        sampling (lambda): an Adapter lambda function used to sample rates
        last_updated (list): a list of nodes that were last updated
        lamdas (defaultdict): a defaultdict of lists representing the current base rates for each node
    """    
    def __init__(self, net, trans, isolate_s=True, trace_after=14, already_exp=True, presample=1000, seed=None, **kwargs):
        self.net = net
        self.trans = trans
        self.violations = 0
        self.isolate_s = isolate_s
        self.trace_after = trace_after
        self.time = 0
        # Create Adapter `sampling` to allow the rate returned by rate_func to be either a base rate (which shall be exponentially sampled) 
        # or an exponential rate directly (which shall be left unchanged)
        if already_exp:
            self.sampling = lambda rate: rate
        else:
            sampler = get_sampler(type='exp', presample_size=presample, scale_one=True, seed=seed)
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
        
        Returns:
            SimEvent: A SimEvent object containing the node id, the current state, the next state and the time of the next event.
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
        isolate_s = self.isolate_s
        node_list = net.node_list
        node_states = net.node_states
        node_traced = net.node_traced
                
        for nid in node_list:
            current_state = node_states[nid]
            # The following commented out block can be used to validate the dynamically-updated neighbor counts of susceptibles correspond to the true counts.
            # Executing these checks slow down the Simulator, hence the exclusion. If more than 5 violations are found, this block of code will raise a ValueError.
            # if current_state == 'S':
            #     inf_neigh, inf_degree, inf = net.infected_neighbors(nid)
            #     if net.node_counts[nid]['I'] != inf_degree:
            #         self.violations += 1
            #         print(f'Issue with {nid=}: {inf_neigh=}, {inf_degree=}, but counts: {net.node_counts[nid]["I"]}')
            #         print('Infected neighbors, weight:', inf)
            #         print('Neighbor states:', [(neigh, net.node_states[neigh], net.node_traced[neigh]) for neigh in net.neighbors(nid)])
            #         if self.violations == 5:
            #             raise ValueError('More than 5 violations of the counts have been found!')
            
            # if the current node is traced, it should not be possible to move from 'S'
            if isolate_s and node_traced[nid] and current_state == 'S':
                continue
            for to, rate_func in trans[current_state]:
                # rate_func is a lambda expression waiting for a net and a node id
                rate = sampling(rate_func(net, nid))
                # a rate < 0 means a significant error has occurred in the sampling, so debug messages are printed and the process stops with a ValueError
                if rate < 0:
                    print(f'{nid=}, {current_state=}, {to=}, counts: {net.node_counts[nid].items()}')
                    print(f'Edges of {nid=}: {net[nid]}')
                    inf_neigh, inf_degree, _ = net.infected_neighbors(nid)
                    print(f'{inf_neigh=}, {inf_degree=}')
                    raise ValueError('Node counts are probably negative for one or multiple susceptible nodes!')
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
        
        Returns:
            SimEvent: A SimEvent object containing the node id, the current state, the next state and the time of the event.
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
        
        traceable_states = {'S', 'E', 'I', 'Ia', 'Is'}
        traced_state = 'T'
        noncompliant_state = 'N'
        
        # Lazily initialize a dict with the correct transition functions for tracing (we need all 'state -> T -> func' entries)
        # This is a bit convoluted because the 'items' of the nested dict 'state -> state -> func' are cached beforehand for efficiency reasons
        # Hence the entries available at this stage are 'state -> (state, func)'
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

        # Filter out nodes that can actually be traced or noncompliant
        for nid in node_list:
            # check if there is any tracing function instantiated
            if trace_funcs and node_states[nid] in trace_funcs:
                # check if the node is not traced yet and from the traceable_states
                if not node_traced[nid] and ((trace_after is not None and time - noncomp_time.get(nid, time) > trace_after) or nid not in traced_time):
                    # rate_func is a lambda expression waiting for a net and a node id
                    rate = sampling(trace_funcs[node_states[nid]](net, nid))
                    # increment time with the current rate
                    trans_time = time + rate
                    if trans_time < best_time:
                        best_time = trans_time
                        from_next = node_states[nid]
                        to_next = traced_state
                        id_next = nid
                # check if there is a noncompliance rate func and the node is traced
                elif noncompliance_rate_func and node_traced[nid]:
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
        Sample next event using Gillespie's Algorithm. This can be used to sample infection AND trace events when args.separate_traced = True.
        
        Args:
            trans_know (dict): a dictionary representing the transition knowledgebase
            tracing_nets (list): list containing tracing network(s) (one for each tracing type selected)

        Returns:
            SimEvent: A SimEvent object containing the node id, the current state, the next state and the time of the next event.

        Notes:
            The Simulation object usually runs over one type of network (INFECTION or TRACING) based on one transition object.
            However, in Gillespie we need access to both the INFECTION and the TRACING rates at the same time in order to sample the next event occurrence.
            To keep the class signature the same for all sampling methods, the TRACING transition knowledgebase and networks are supplied as parameters here. 
            This comes in addition to the `self` Simulation object, which should be able to reference the INFECTION network and transition knowledgebase.
            Regardless of the implementation, one should consider the INFECTION and the TRACING objects as global variables here.
        """      
        # making local vars for efficiency
        net = self.net
        trans = self.trans
        time = self.time
        sampling = self.sampling
        trace_after = self.trace_after
        isolate_s = self.isolate_s
        node_list = net.node_list
        node_states = net.node_states
        node_traced = net.node_traced
        traced_time = net.traced_time
        noncomp_time = net.noncomp_time

        traceable_states = {'S', 'E', 'I', 'Ia', 'Is'}
        traced_state = 'T'
        noncompliant_state = 'N'
        
        # Lazily initialize a dict with the correct transition functions for tracing (collects all 'state -> T -> func' entries).
        # This is a bit convoluted because the 'items' of the nested dict 'state -> state -> func' get cached as 'state -> (state, func)' for efficiency purposes.
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
            # The following commented out block can be used to validate the dynamically-updated neighbor counts of susceptibles correspond to the true counts.
            # Executing these checks slow down the Simulator, hence the exclusion. If more than 5 violations are found, this block of code will raise a ValueError.
            # if current_state == 'S':
            #     inf_neigh, inf_degree, inf = net.infected_neighbors(nid)
            #     if net.node_counts[nid]['I'] != inf_degree:
            #         self.violations += 1
            #         print(f'Issue with {nid=}: {inf_neigh=}, {inf_degree=}, but counts: {net.node_counts[nid]["I"]}')
            #         print('Infected neighbors, weight:', inf)
            #         print('Neighbor states:', [(neigh, net.node_states[neigh], net.node_traced[neigh]) for neigh in net.neighbors(nid)])
            #         if self.violations == 5:
            #             raise ValueError('More than 5 violations of the counts have been found!')
            
            # update infection possible transitions lists
            if not (isolate_s and current_traced and current_state == 'S'):
                for to, rate_func in trans[current_state]:
                    lamdas[nid].append((rate_func(net, nid), to))
                    
            # check if there is a tracing or noncompliance transition possible before updating the transition knowledgebase lists
            if trace_funcs and current_state in trace_funcs:
                if current_traced:
                    if noncompliance_rate_func:
                        for trace_net in tracing_nets:
                            lamdas[nid].append((noncompliance_rate_func(trace_net, nid, time), noncompliant_state))
                else:
                    # if trace_after is None -> trace a node only if its NOT present in traced_time: i.e. never traced or legally exited isolation
                    # if trace_after not None -> make tracing same nodes possible only after a time delay trace_after since becoming noncompliant N
                    if nid not in traced_time or (trace_after is not None and time - noncomp_time.get(nid, time) > trace_after):
                        for trace_net in tracing_nets:
                            lamdas[nid].append((trace_funcs[current_state](trace_net, nid), traced_state))                    
        
        base_rates = []
        sum_lamdas = 0
        nodes_and_trans = []
        for nid, sublist in lamdas.items():
            for rate, to in sublist:
                # a rate < 0 means a significant error has occurred in the sampling, so debug messages are printed and the process stops with a ValueError
                if rate < 0:
                    print(f'{nid=}, {to=}, counts: {net.node_counts[nid].items()}')
                    print(f'Edges of {nid=}: {net[nid]}')
                    inf_neigh, inf_degree, _ = net.infected_neighbors(nid)
                    print(f'{inf_neigh=}, {inf_degree=}')
                    raise ValueError('Node counts are probably negative for one or multiple susceptible nodes!')
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
        """
        Creates a new SimEvent object with the given configuration.

        Args:
            **kwargs: Keyword arguments specifying the configuration for the new SimEvent object.

        Returns:
            A new SimEvent object with the specified configuration.
        """
        return SimEvent(**kwargs)
        
    def run_event(self, e):
        """
        Runs the given event by changing the state of the node, updating the time, and adding the node to the last_updated list.

        Args:
            e (Event): The event to be run.

        Returns:
            None
        """
        self.net.change_state_fast_update(e.node, e.to, time=e.time)
        self.time = e.time
        self.last_updated.append(e.node)
        
    def run_trace_event_for_infect_net(self, e, to_traced=True, legal_isolation_exit=False, order_nid=False):
        """
        Runs a tracing event on the infection network in a separate_traced setup.

        Args:
            e (Event): The tracing event to run.
            to_traced (bool): What to change the node traced status to (default True).
            legal_isolation_exit (bool): Whether the node is legally exiting isolation (default False).
            order_nid (bool): Whether to insert the node ID in the last_updated list in ascending order or append it (default False).

        Returns:
            None
        """
        self.net.change_traced_state_update_infectious(e.node, to_traced, e.time, legal_isolation_exit)
        self.time = e.time
        insort(self.last_updated, e.node) if order_nid else self.last_updated.append(e.node)
         
    def run_trace_event_for_trace_net(self, e, to_traced=True, legal_isolation_exit=False):
        """
        Runs a tracing event on a tracing network in a separate_traced setup.

        Args:
            e (Event): The tracing event to run.
            to_traced (bool): What to change the node traced status to (default True).
            legal_isolation_exit (bool): Whether the node is legally exiting isolation (default False).
        
        Returns:
            None
        """
        self.net.change_traced_state_update_tracing(e.node, to_traced, e.time, legal_isolation_exit)
        self.time = e.time