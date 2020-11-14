import networkx as nx

import itertools
import random
from copy import deepcopy

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from lib.utils import rand_pairs, rand_pairs_excluding, get_z_for_overlap, get_overlap_for_z
from lib.simulation import Simulation
from lib.exp_sampler import ExpSampler

STATES_COLOR_MAP = {
    'S': 'darkgreen',
    'E': 'yellow',
    'I': 'orange',
    'Ia': 'pink',
    'Is': 'red',
    'H': 'cyan',
    'T': 'blue',
    'R': 'lime',
    'D': 'gray',
    'N': 'purple'
}

class Network(nx.Graph):
    
    def get_simulator(self, trans):
        return Simulation(self, trans)
    
    def __init__(self, **kwds):
        # Atomic Counter for node ids
        self.cont = itertools.count()
        # Initialize pos to None - a new spring layout is created when first drawn
        self.pos = None
        # Overlap after noising - initially perfect overlap, changes after calling 'noising_links'
        self.overlap = 1
        # Maintain 'active' (not orphans) node list + neighbor counts; and inf + traced states for all nodes
        self.node_list = []
        self.node_states = []
        self.node_traced = []
        self.node_counts = defaultdict(dict)
        self.traced_time = {}
        self.noncomp_time = {}
        # relative counts importance (will be multiplied with the counts to obtain their importance in the rate)
        self.count_importance = 1
        
        super().__init__(**kwds)
        
    def init_random(self, n=200, k=10, rem_orphans=False, presample_size=0, weighted=False):
        # Add nodes in state 'S'
        self.add_mult(n, 'S')
        # Add random links with degree k
        remaining_links = int(n * k / 2)
        # Generate random pairs without replacement
        links_to_create = rand_pairs(n, remaining_links)
        if weighted:
            # Create random weights from 1 to 10 and append them to the list of edge tuples
            weights = random.choices(range(1,10), k=remaining_links)
            links_to_create = [links_to_create[i] + (weights[i],) for i in range(remaining_links)]
        # Add the random edges with/without weights depending on 'weighted' parameter
        # This also updates the active node list and the traced array based on orphan encounter
        self.add_links(links_to_create, update=False, rem_orphans=rem_orphans)
        
        # exponential sampler associated with this network
        self.sampler = ExpSampler(size=presample_size) if presample_size else None
                             
        return self
        
    def noising_links(self, overlap=None, z_add=0, z_rem=5, keep_nodes_percent=1, maintain_overlap=True, update=True):
        # Recover total num of nodes
        n = self.number_of_nodes()
        # Recover average degree
        k = self.avg_degree()
        
        # Current edge set (prior to noising, nodes in each edge tuple are sorted)
        current_edges = {tuple(sorted(edge)) for edge in self.edges}
        
        # links to add and rem lists
        links_to_add, links_to_rem = [], []
        remaining_links_rem = 0
        
        # If we are interested in maintaining the overlap, then the correct z_add and z_rem will be calculated based on the params
        if maintain_overlap:
            # If no overlap value, use z_add & z_rem
            if overlap is None:
                # Average degree k, z_add, z_rem used to calculate overlap after noising
                self.overlap = get_overlap_for_z(k, z_add, z_rem)
            # If overlap given, calculate z_add and z_rem; SUPPLIED z_add and z_rem are mostly IGNORED
            else:
                self.overlap = overlap
                # if z_add = 0, z_add will stay 0 while z_rem = z; OTHERWISE z_add = z_rem = z
                z_add, z_rem = get_z_for_overlap(k, overlap, include_add = z_add)
                
            # Random adding z_add on average - this should be 0 in case of digital tracing networks
            remaining_links_add = int(n * z_add / 2)
            links_to_add = rand_pairs_excluding(n, remaining_links_add, to_exclude=current_edges) if z_add else []

            # Random removing z_rem on average - this will also cover the case keep_nodes_percent < 1
            remaining_links_rem = int(n * z_rem / 2)
            #
            # The logic for removing edges is done below (need to account for keep_nodes_percent value)
        
        
        # Simple case, no node's edges are removed completely from the network unless the randomness does it
        if keep_nodes_percent == 1:
            links_to_rem = random.sample(current_edges, remaining_links_rem)
            
        # The links of certain nodes will be completely removed if keep_nodes_percent is below 1 (<100%)
        else:
            untraceable_nodes = random.sample(self.node_list, int(round(1 - keep_nodes_percent, 2) * n))
            untraceable_edges = {tuple(sorted(edge)) for edge in self.edges(untraceable_nodes)}
            len_untraceable_edges = len(untraceable_edges)
            
            # update the total links_to_rem list
            links_to_rem += untraceable_edges
            
            # maintain_overlap tries to ensure the overlap value is also met in the keep_nodes_percent < 1 case
            # if this value cannot be met, an error will be thrown
            if maintain_overlap:
                remaining_links_rem -= len_untraceable_edges
                if remaining_links_rem >= 0:
                    # The rest of the edges until the overlap value is met are to be removed at random
                    links_to_rem += random.sample(current_edges - untraceable_edges, remaining_links_rem)
                else:
                    raise ValueError("The value of keep_nodes_percent is too small for maintaining the selected overlap!")
            # in the case the supplied overlap gets ultimately ignored, a new value needs to be calculated for it
            else:
                actual_z_add = 0
                actual_z_rem = len_untraceable_edges * 2 / n
                self.overlap = get_overlap_for_z(k, actual_z_add, actual_z_rem)

                
        self.add_edges_from(links_to_add)
        self.remove_edges_from(links_to_rem)

        if update:
            self.update_counts()
        
        return self
    
    def reinit_for_another_iter(self, first_inf_nodes):
        self.init_states('S')
        self.change_state(first_inf_nodes, state='I', update=True)
    
    def init_states(self, state='S'):
        # Set all nodes back to state
        n = self.number_of_nodes()
        self.node_states = [state] * n
        # the boolean tracing array must reflect that only the active nodes are NOT traced
        self.node_traced = np.isin(list(self), self.node_list, invert=True).tolist()

    def add(self, state='S', traced=False):
        current = next(self.cont)
        self.add_node(current)
        self.node_states.append(state)
        self.node_traced.append(traced)
        
    def add_mult(self, n=200, state='S', traced=False):
        current = next(self.cont)
        self.add_nodes_from(range(current, current + n))
        self.node_states += [state] * n
        self.node_traced += [traced] * n
        # increment count to reflect n additions
        self.cont = itertools.count(current + n)
        
    def update_counts(self, nlist=None):
        # local for efficiency
        counts = self.node_counts
        states = self.node_states
        # if nlist not given, update_counts for all 'active' nodes
        if nlist is None:
            nlist = self.node_list
        for nid in nlist:
            # get the list of neighbour states and num of traced
            neigh_states_for_node = [states[i] for i in self.neighbors(nid)]
            # update counts
            counts[nid] = Counter(neigh_states_for_node)
            
    def update_counts_with_traced(self, nlist=None):
        # if nlist not given, update_counts for all 'active' nodes
        if nlist is None:
            nlist = self.node_list
        for nid in nlist:
            # get the list of neighbour states and num of traced
            neigh_states_for_node = []
            num_traced_for_node = 0
            for i in self.neighbors(nid):
                neigh_states_for_node.append(self.node_states[i])
                if self.node_traced[i]:
                    num_traced_for_node += 1
            # update neighbor count but give priority to self.node_traced for 'T' count
            counter = Counter(neigh_states_for_node)
            counter['T'] = num_traced_for_node
            self.node_counts[nid] = counter
            
    def get_count(self, nid, state='I'):
        return self.node_counts[nid][state]
            
    def generate_layout(self, seed=43):
        self.pos = nx.spring_layout(self, seed=seed)
        
    def get_state(self, nid):
        return self.node_states[nid]
        
    def change_state(self, nids, state='I', update=True):
        for nid in np.atleast_1d(nids):
            self.node_states[nid] = state
        if update:
            self.update_counts()
                
    def change_state_fast_update(self, nid, state):
        # local vars for efficiency
        states = self.node_states
        # update counts if NOT traced, NOT hospitalized, NOT exposed without being infectious:
        old_state = states[nid]
        if state != 'E' and not self.node_traced[nid] and old_state != 'H':
            counts = self.node_counts
            for neigh in self.neighbors(nid):
                counts_neigh = counts[neigh]
                counts_neigh[old_state] -= 1
                counts_neigh[state] += 1
        states[nid] = state
            
    def change_traced_state_fast_update(self, nid, to_traced, time_of_trace):
        """
        When using separate_traced, the T state is independent of all the other states and represented via a flag
        Counts are updated as-if this would be an actual state change: 
         - node_counts for neighbor node -> for current inf state of nid decrease/increase, for 'T' increment/decremenet
         based on to_traced True/False
        """
        # local vars for efficiency
        counts = self.node_counts
        # get this node's current infection network state
        inf_state = self.node_states[nid]
        # update counts only if the infection network state of this node is an infectious state
        update_infectious_counts = (inf_state in ['I', 'Ia', 'Is'])
        
        # switch traced flag for the current node
        self.node_traced[nid] = to_traced
        
        # If this is a traced event, count_val = 1, update time of tracing
        if to_traced:
            count_val = 1
            self.traced_time[nid] = time_of_trace
        # for noncomploance, count_val = -1
        else:
            count_val = -1
            self.noncomp_time[nid] = time_of_trace
        
        for neigh in self.neighbors(nid):
            neigh_counts = counts[neigh]
            neigh_counts['T'] += count_val
            if update_infectious_counts:
                neigh_counts[inf_state] -= count_val
                
    def add_link(self, nid1, nid2, weight=1, update=False):
        self.add_edge(nid1, nid2, weight=weight)
        if update:
            self.update_counts([nid1, nid2])
            
    def rem_link(self, nid1, nid2, update=False):
        self.remove_edge(nid1, nid2)
        if update:
            self.update_counts([nid1, nid2])
            
    def add_links(self, lst, weight=1, update=False, rem_orphans=True):
        len_elem = len(lst[0])
        # If only source + target nodes provided, add specified 'weight' parameter as weight
        if len_elem == 2:
            self.add_edges_from(lst, weight=weight)
        # If weights provided alognside nodes, ignore weight parameter
        elif len_elem == 3:
            self.add_weighted_edges_from(lst)
        
        # Set the active node list (if rem_orphans=False, these are all the nodes)
        self.node_list = list(self)
        if rem_orphans:
            for nid in self.node_list:
                if self.degree(nid) == 0:
                    self.node_list.remove(nid)
                    self.node_traced[nid] = True     
        if update:
            self.update_counts()
            
    def avg_degree(self):
        return np.mean(list(dict(self.degree()).values()))
    
    def compute_efforts_and_check_end_config(self):
        # local for efficiency
        node_traced = self.node_traced
        node_states = self.node_states
        node_counts = self.node_counts
        # if this var changes to False, the network is not yet in the ending configuration (all end states)
        end_config = True
        # compute random tracing effort -> we only care about the number of traceable_states ('S', 'E', 'I', 'Ia', 'Is')
        randEffortAcum = 0
        # compute active tracing effort -> we only care about the neighs of nodes in traceable_states ('S', 'E', 'I', 'Ia', 'Is')
        tracingEffortAccum = 0
        for nid in self.node_list:
            current_state = node_states[nid]
            if not node_traced[nid] and current_state in ['S', 'E', 'I', 'Ia', 'Is']:
                randEffortAcum += 1
                tracingEffortAccum += node_counts[nid]['T']
            # if one state is not a final configuration state, continue simulation
            if current_state not in ['S', 'R', 'D']:
                end_config = False
                
        return randEffortAcum, tracingEffortAccum, end_config
    
        
    def draw(self, pos=None, show=True, ax=None, seed=43):
        # for the true network, colors for all nodes are based on their state
        if self.overlap == 1:
            colors = list(map(lambda x: STATES_COLOR_MAP[x], self.node_states))
        # for the dual network, when no explicit T state set, give priority to node_traced == True as if 'T' state
        # also give priority to node_traced == False as if 'N' state IF the node has been traced before (has a traced_time)
        else:
            colors = [None] * self.number_of_nodes()
            for nid, (state, traced) in enumerate(zip(self.node_states, self.node_traced)):
                if traced:
                    colors[nid] = STATES_COLOR_MAP['T']
                elif nid in self.traced_time:
                    colors[nid] = STATES_COLOR_MAP['N']
                else:
                    colors[nid] = STATES_COLOR_MAP[state]

        # by doing this we avoid overwriting self.pos with pos when we just want a different layout for the drawing
        pos = pos if pos else self.pos
        # if both were None, generate a new layout with the seed and use it for drawing
        if not pos:
            self.generate_layout(seed)
            pos = self.pos
            
        # sometimes an iterable of axis may be supplied instead of one Axis object, so get the first element
        if isinstance(ax, Iterable): ax = ax[0]
        # draw graph
        nx.draw(self, pos=pos, node_color=colors, ax=ax, with_labels=True)
        # create color legend
        plt.legend(handles=[mpatches.Patch(color=color, label=state) for state, color in STATES_COLOR_MAP.items()])
        
        if show:
            plt.show()
            
    
def get_random(n=200, k=10, rem_orphans=False, presample_size=0, weighted=False):
    G = Network()
    return G.init_random(n, k, rem_orphans, presample_size, weighted)
       
def get_dual(G, overlap=None, z_add=0, z_rem=5, keep_nodes_percent=1, maintain_overlap=True, count_importance=1):
    N = deepcopy(G)
    N.count_importance = count_importance
    return N.noising_links(overlap, z_add, z_rem, keep_nodes_percent, maintain_overlap, update=True)